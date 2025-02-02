import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fcs = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)

        x = self.fcs(x[:, -1, :])

        return x

class SimpleRNNModel(nn.Module):
    def __init__(self, hidden_dim, input_dim, num_layers, num_classes):
        super(SimpleRNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)

        self.fcs = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x)

        x = self.fcs(x[:, -1, :])

        return x