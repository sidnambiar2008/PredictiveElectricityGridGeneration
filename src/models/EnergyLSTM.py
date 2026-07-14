from torch import nn


class GridPulseLSTM(nn.Module):
    def __init__(self, input_size: int = 9, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 9)

    def forward(self, x):
        # 'output' shape is 3D: [batch_size, sequence_len, hidden_size]
        output, (h_n, c_n) = self.lstm(x)

        # Sliced to predict the most recent time step, and we keep all the hidden nodes
        predictions = self.fc(output[:, -1, :])
        return predictions


