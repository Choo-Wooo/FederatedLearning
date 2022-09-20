import torch
from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, conv1_channels=64, conv2_channels=128, linear1_size=256, linear2_size=128, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=conv1_channels, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=3, stride=1, padding=1)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(in_features=conv2_channels*7*7, out_features=linear1_size)
        self.fc2 = nn.Linear(in_features=linear1_size, out_features=linear2_size)
        self.fc3 = nn.Linear(in_features=linear2_size, out_features=10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.flat(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.drop(out)

        out = self.fc3(out)

        return out