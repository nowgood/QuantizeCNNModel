# coding=utf-8
import torch
from net.simple_net import Net
from quantize.quantize_method import QuantizeWeightOrActivation
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def test_quantize_weight():
    qw = QuantizeWeightOrActivation()

    net = Net()
    qw.info(net, "初始化权重")

    net.apply(qw.quantize)
    qw.info(net, "量化权重")

    # 网络输入, 输出
    input_ = torch.ones(1, 1, 6, 6, requires_grad=True)
    lable = torch.ones(1, 2)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    output = net(input_)
    loss = criterion(output, lable)
    optimizer.zero_grad()
    loss.backward()
    print("\nMSE LOSS ", loss, "\n")

    net.apply(qw.restore)
    qw.info(net, "恢复全精度权重")

    net.apply(qw.update_grad)

    print("now")
    optimizer.step()
    qw.info(net, "更新全精度权重")


def test_quantize_weight_update():
    qw = QuantizeWeightOrActivation()

    net = Net()
    input_ = torch.rand(1, 1, 6, 6, requires_grad=True)
    label = torch.ones(1, 2)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.99)
    criterion = nn.MSELoss()
    log = {}
    for step in torch.arange(5000):
        net.apply(qw.quantize)
        output = net(input_)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        # print("loss ", loss.data)
        net.apply(qw.restore)
        net.apply(qw.update_grad)
        optimizer.step()

        log[step] = loss

    plt.axis([0, 5000, 0, 0.1])
    plt.plot(log.values(), "r-")
    plt.show()


if __name__ == "__main__":
    test_quantize_weight_update()