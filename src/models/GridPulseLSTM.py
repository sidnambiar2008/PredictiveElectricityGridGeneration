from torch import nn


class GridPulseLSTM(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, num_layers: int = 1, forecast_horizon: int = 24):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, forecast_horizon * input_size)

    def forward(self, x):
        # 'output' shape is 3D: [batch_size, sequence_len, hidden_size]
        output, (h_n, c_n) = self.lstm(x)

        # Uses the final hidden state from the 168-hour memory window
        # to predict the next 24 hourly fuel-mix states
        predictions = self.fc(output[:, -1, :])

        # Converts (batch, 216) into (batch, 24, 9)
        predictions = predictions.view( -1, self.forecast_horizon, self.input_size
        )
        return predictions


