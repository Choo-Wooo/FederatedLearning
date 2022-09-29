import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    # def __init__(self) -> None:
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 10)
    #
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     return self.fc3(x)


class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv2d(3, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=(1, 1))  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv2d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=(1, 1))  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv2d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=(1, 1))  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv2d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7
        # view

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)  batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7

        #
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)
        return out

def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    return loss / len(testloader.dataset), correct / total


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = VGG16().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(server_address="0.0.0.0:8080",client=FlowerClient(),)