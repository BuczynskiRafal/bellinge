import torch
import torch.nn as nn


class FlashFloodLSTM(nn.Module):
    def __init__(self, input_size=47, hidden_size=19, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, apply_sigmoid=False):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        output = self.fc(last_hidden)
        if apply_sigmoid:
            output = torch.sigmoid(output)
        return output.squeeze(1)
