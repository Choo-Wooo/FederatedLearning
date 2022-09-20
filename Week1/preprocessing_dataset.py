import pandas as pd
import numpy as np
import torch

from torchvision import transforms
from torch.utils.data import Dataset

train_dataframe = pd.read_csv('dataset/train.csv')/255
test_dataframe = pd.read_csv('dataset/test.csv')/255
test_label_dataframe = pd.read_csv('dataset/sample_submission.csv')

train_label = train_dataframe.label
train_input = train_dataframe.drop('label', axis=1)
train_input = train_input.values.reshape(-1, 1, 28, 28)

test_idx = test_dataframe.index
test_dataset = test_dataframe.values.reshape(-1, 1, 28, 28)
test_label = test_label_dataframe.Label

# Image Dataset Transform
transforms = transforms.Compose([
    transforms.RandomRotation(degrees=30),
    transforms.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.2), ratio=(1, 1))
])

class Dataset(Dataset):
    def __init__(self, task='train', transform=None):
        super().__init__()
        self.task = task
        self.transform = transform

        if self.task == 'train':
            self.input = torch.from_numpy(train_input.astype(np.float32))
            self.label = torch.from_numpy(train_label.values)

        elif self.task == 'test':
            self.input = torch.from_numpy(test_dataset.astype(np.float32))
            self.label = torch.from_numpy(test_label.values)

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        if self.task == 'train':
            input_data = self.input[idx]
            input_data = self.transform(input_data)

        elif self.task == 'test':
            input_data = self.input[idx]
        return input_data, self.label[idx]

train_set = Dataset(task='train', transform=transforms)
test_set = Dataset(task='test')

print(test_set)