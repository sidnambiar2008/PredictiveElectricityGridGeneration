import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import os

from src.models.EnergyLSTM import GridPulseLSTM
from src.ingestion.data_loader import GridDataLoader

# Initialize hyperparameters related to training to simplify tuning
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15

def train_model():
    # Hardware Anchors
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 Training hardware detected: {device.type.upper()}")

    # Pipeline Processing
    dataset = GridDataLoader(grid_history_csv="grid_data/raw/grid_history.csv")

    train_size = int(len(dataset) * 0.85)
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, len(dataset)))

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(dataset = train_subset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(dataset = val_subset, batch_size = BATCH_SIZE, shuffle = False)

    # Model Instantiation
    model = GridPulseLSTM(input_size=9, hidden_size=64).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n⚡ Setup complete. Ready to begin training loops...")

    for epoch in range(1, EPOCHS+1):
        model.train()

        # Variable that helps logging the loss
        running_loss = 0.0

        for (x_batch, y_batch) in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # 5 step sequence to ensure that 
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            print(f"   Epoch [{epoch:02d}/{EPOCHS:02d}] | Mean Training Loss: {epoch_loss:.6f}")

            os.makedirs("saved_models", exist_ok=True)
            save_path = os.path.join("saved_models", "lstm_grid_pulse.pt")
            torch.save(model.state_dict(), save_path)
            print(f"\n Training complete! Model weights successfully stored at: {save_path}")


if __name__ == "__main__":
    train_model()
