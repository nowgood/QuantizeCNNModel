# coding=utf-8
import torch
import torch.nn as nn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=0, stride=1, bias=False)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1, bias=False)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16, 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        x = x.view(-1, num_features)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    net = Net()
    print(net)
