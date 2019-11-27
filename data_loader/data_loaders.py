import torch
from torchvision import datasets, transforms
import os
from base import BaseDataLoader
from torch.utils.data import Dataset

import pandas as pd
from tqdm import tqdm
from math import floor
import pickle

class StockDataLoader(BaseDataLoader):
    """
    Stock data loading demo using BaseDataLoader
    """
    def __init__(self, data_path, howManyDays, batch_size, training, normalize_info_path, shuffle=True, validation_split=0.0, testing_split=0.0, num_workers=1):
        self.dataset = StockDataset(data_path, howManyDays, training, testing_split, normalize_info_path)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class StockDataset(Dataset):
    def __init__(self, data_path, howManyDays, training, testing_split, normalize_info_path):
        self.data_path = data_path
        stockdf = pd.read_csv(self.data_path)
        self.howManyDays = howManyDays
        self.training = training
        self.testing_split = testing_split
        self.normalize_info_path = normalize_info_path

        print("How many days per entry:", self.howManyDays)
        self.loaddata(stockdf)
        print("Dataset length", len(self.dataset))
        
    def loaddata(self, raw_data):
        print("Origin Raw data length", len(raw_data))
        print("cutting date(testing start): {}".format(raw_data.iloc[floor((1 - self.testing_split) * len(raw_data))].date), floor((1 - self.testing_split) * len(raw_data)))
        del raw_data["date"]

        stat = {}
        if self.training:
            stat["mean"] = raw_data.mean()
            stat["std"] = raw_data.std()
            with open( self.normalize_info_path, 'wb') as f:
                pickle.dump( stat, f)
        else:
            with open( self.normalize_info_path, 'rb') as f:
                stat = pickle.load( f)

        raw_data = (raw_data - stat["mean"]) / stat["std"] # mean normalization

        if self.training:
            raw_data = raw_data[: floor((1 - self.testing_split) * len(raw_data))] 
        else:
            #raw_data = raw_data[floor((1 - self.testing_split) * len(raw_data)) - self.howManyDays : len(raw_data)][['open', 'high', 'low', 'volume', 'close']]
            pass
        self.dataset = []

        print("Raw data length {} | is training: {}".format(len(raw_data), self.training))
        for i in tqdm(range(0, len(raw_data) - self.howManyDays - 1)):
            if i + self.howManyDays < len(raw_data):
                data = {"data": torch.FloatTensor(raw_data[i : i + self.howManyDays].values.tolist()),
                        "label": torch.FloatTensor(raw_data[["close"]].iloc[i + self.howManyDays].values.tolist())}
            self.dataset.append(data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index] 
        return data["data"], data["label"]
