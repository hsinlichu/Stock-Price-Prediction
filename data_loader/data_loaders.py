from torchvision import datasets, transforms
import os
cwd = os.getcwd()
print(cwd)
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.append('./base')
sys.path.append('.')
sys.path.append('./data')
from base import BaseDataLoader
from torch.utils.data import Dataset

import pandas as pd
from tqdm import tqdm



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class StockDataLoader(BaseDataLoader):
    """
    Stock data loading demo using BaseDataLoader
    """
    def __init__(self, data_path, howManyDays, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = StockDataset(self, data_path, howManyDays)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)




class StockDataset(Dataset):
    def __init__(self, data_path, howManyDays):
        self.data_path = data_path
        stockdf = pd.read_csv(self.data_path)
        self.howManyDays = howManyDays
        print("How many days per entry:", self.howManyDays)
        self.loaddata(stockdf)
        
    def loaddata(self, raw_data):
        self.dataset = []
        for i in tqdm(range(self.howManyDays, len(raw_data) - 1)):
            data = {"data": raw_data[i - self.howManyDays : i],
                    "label": raw_data.iloc[i + 1]}
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

