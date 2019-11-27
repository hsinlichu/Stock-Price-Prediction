import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class StockModel(BaseModel):
    def __init__(self, input_size=47, hidden_size=128, num_layers=1, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        print("num_layer", self.num_layers)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        #print("GRU output", output.size())
        last_output = output[:, -1, :]
        x = F.relu(last_output)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        #print("FC1 output", x.size())
        predict = self.fc2(x)

        return predict

class VAElstm(BaseModel):
    def __init__(self, input_size=47, hidden_size=128, num_layers=2, batch_first=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, input_size - 10)
        self.fc21 = nn.Linear(input_size - 10, 10)
        self.fc22 = nn.Linear(input_size - 10, 10)
        self.fc3 = nn.Linear(10, input_size - 10)
        self.fc4 = nn.Linear(input_size - 10, input_size)

        self.lstm_model = StockModel(10, hidden_size, num_layers, batch_first)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.lstm_model(z)
        return (self.decode(z), x, mu, logvar, output)


