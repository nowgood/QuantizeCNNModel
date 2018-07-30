# coding=utf-8
"""
论文中 scalar layer 层设计 (多个 GPU )
"""

import torch
import torch.nn as nn


class Scalar(nn.Module):

    def __init__(self):
        super(Scalar, self).__init__()  # 这一行很重要
        # 第1种错误
        # self.scalar = torch.tensor([0.01], requires_grad=True)
        # RuntimeError: Expected object of type torch.FloatTensor
        # but found type torch.cuda.FloatTensor for argument

        # 第2种错误
        # self.scalar = torch.tensor([0.01], requires_grad=True).cuda()
        # RuntimeError: arguments are located on different GPUs

        # 第3种错误
        # self.scalar = nn.Parameter(torch.tensor(0.01, requires_grad=True))
        # RuntimeError: slice() cannot be applied to a 0-dim tensor,
        #  而加了方括号正确为 1-dim tensor

        # 第4中错误
        #  scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        #  self.register_buffer("scalar", scalar)
        #  scalar没有梯度更新(全是None), register_buffer 用于存储非训练参数, 如bn的平均值存储

        # 第1种方法, 可以使用
        # self.scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))

        # 第2种方法, 可以使用
        # scalar = nn.Parameter(torch.tensor([0.01], requires_grad=True))
        # self.register_parameter("scalar", scalar)

        # 根据训练经验, 设为 2.5
        self.scalar = nn.Parameter(torch.tensor([1.0], requires_grad=True, dtype=torch.float))

    def forward(self, i):
        return self.scalar * i


if __name__ == "__main__":
    # 多 GPU
    CUDA_VISIBLE_DEVICES = 0, 3
    net = Scalar()
    pnet = torch.nn.DataParallel(net, [0, 1]).cuda()

    print("start training")
    for _ in range(10):
        x = torch.rand((3, 600, 3, 3))
        y = pnet(x).view(1, -1).squeeze().sum()
        y.backward()
        print(net.sim.scalar.grad)