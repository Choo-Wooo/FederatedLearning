# Week3 Assignment

### 기존에 사용하던 NET과 다르게 VGG16 모델을 사용하여 실습함

기존 모델
    
    class Net(nn.Module):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

VGG16 pretrainde 모델

    use_pretrained=True
    trainloader, testloader, _ = load_data()
    net=models.vgg16(pretrained=use_pretrained)
    net.eval()

Result

![VGG16_result](https://user-images.githubusercontent.com/90769598/192919020-b3d0781c-1feb-4e21-a784-1cc56c5e022d.PNG)
