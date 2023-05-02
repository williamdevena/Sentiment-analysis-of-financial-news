import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            self.num_layers,
                            batch_first=True)
        self.linear = nn.Linear()
        self.final_act = nn.Softmax()


    def forward(self, x):
        x = self.lstm(x)
        x = self.linear(x)
        out = self.final_act(x)

        return out
