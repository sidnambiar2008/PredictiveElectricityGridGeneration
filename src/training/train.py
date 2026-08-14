import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import os
import joblib
from src.regions import REGIONS
from src.model_config import HIDDEN_SIZE, NUM_LAYERS

from src.models.GridPulseLSTM import GridPulseLSTM
from src.preprocessing.data_loader import GridDataLoader
import logging

logger = logging.getLogger(__name__)

# Initialize hyperparameters related to training to simplify tuning
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15


def train_model(region_str):
    """
    Train the LSTM model for a specified region

    Args:
        region_str (string): The region name
    Returns:
        None. Saves model weights and scaler to the saved_models directory
    Side Effects:
        Logs Hardware detected
        Logs Epoch Number
        Logs Mean Training Loss
        Logs Validation Loss
        Logs Scaler/Model Save Path
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"💻 Training hardware detected: {device.type.upper()}")

    dataset = GridDataLoader(grid_history_csv=f"grid_data/raw/grid_history_{region_str.lower()}_v4.csv")

    # Ensures we have some data for training and validation
    train_size = int(len(dataset) * 0.85)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(dataset = train_subset, batch_size = BATCH_SIZE, shuffle = False)
    val_loader = DataLoader(dataset = val_subset, batch_size = BATCH_SIZE, shuffle = False)

    model = GridPulseLSTM(input_size=len(dataset.feature_cols), hidden_size=HIDDEN_SIZE, num_layers= NUM_LAYERS, forecast_horizon=dataset.forecast_horizon).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logger.info("\n⚡ Setup complete. Ready to begin training loops...")

    for epoch in range(1, EPOCHS+1):
        model.train()

        running_loss = 0.0

        for (x_batch, y_batch) in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        logger.info(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Mean Training Loss: {epoch_loss:.6f}")

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(x_batch)

                if epoch == 1:
                    logger.info(f"Input: {x_batch.shape}")
                    logger.info(f"Target: {y_batch.shape}")
                    logger.info(f"Prediction: {predictions.shape}")

                loss = loss_fn(predictions, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        logger.info(f"| Validation Loss: {val_loss:.6f}")

    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", f"lstm_grid_pulse_24h_{region_str.lower()}_v2.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"\n Training complete! Model weights successfully stored at: {save_path}")

    # Persists the exact scaler this model was trained against, so later scripts
    # transform live/validation data the same way regardless of what the CSV
    # on disk looks like by the time they run
    scaler_path = os.path.join("saved_models", f"scaler_{region_str.lower()}_v2.pkl")
    joblib.dump(dataset.scaler, scaler_path)
    logger.info(f" Scaler saved alongside model at: {scaler_path}")


if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)
    regions = REGIONS
    for region in regions:
        try:
            logger.info(f"\n========== TRAINING {region} ==========")
            train_model(region)
        except Exception as error:
            logger.exception(f"Training failed for {region}: {error}")


