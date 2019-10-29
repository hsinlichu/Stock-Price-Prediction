import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

class StockModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.gru = nn.GRU(input_size=5, hidden_size=128, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        #print("GRU output", output.size())
        x = F.relu(output[:, -1, :])
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        #print("FC1 output", x.size())
        predict = self.fc2(x)
        return predict
