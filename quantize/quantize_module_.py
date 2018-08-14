# coding=utf-8
import torch
import torch.nn as nn
from quantize.quantize_function import quantize_weights_bias, quantize_activations
import torch.nn.functional as F


class QWConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QWConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input):
        """
        关键在于使用函数 F.conv2d, 而不是使用模块 nn.ConV2d
        """
        qweight = quantize_weights_bias(self.weight)
        return F.conv2d(input, qweight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QWAConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QWAConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                       padding, dilation, groups, bias)
        # nn.init.xavier_normal_(self.weight, 1)
        # nn.init.constant_(self.weight, 1)

    def forward(self, input):
        qweight = quantize_weights_bias(self.weight)
        qinput = quantize_activations(input)
        return F.conv2d(qinput, qweight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class QWLinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False):
        super(QWLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qweight = quantize_weights_bias(self.weight)

        if self.bias is not None:
            qbias = quantize_weights_bias(self.bias)
        else:
            qbias = None

        return F.linear(input, qweight, qbias)


class QWALinear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(QWALinear, self).__init__(in_features, out_features, bias)

    def forward(self, input):
        qinput = quantize_activations(input)
        qweight = quantize_weights_bias(self.weight)

        if self.bias is not None:
            qbias = quantize_weights_bias(self.bias)
        else:
            qbias = None

        return F.linear(qinput, qweight, qbias)

"""
论文中 scalar layer 层设计 (多个 GPU )
"""


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
    qconv = QWConv2D(1, 1, 3)
    qconv.zero_grad()
    x = torch.ones(1, 1, 3, 3, requires_grad=True).float()
    y = qconv(x)
    y.backward()
    print("QConv2D 权重梯度", qconv.weight.grad)

    # 直接求梯度
    a = torch.ones(3, 3, requires_grad=True).float()
    w = nn.init.constant_(torch.empty(3, 3, requires_grad=True), 1)
    qw = quantize_weights_bias(w)

    z = (qw * a).sum()
    z.backward()
    print("求权重梯度", w.grad)

    # 验证量化梯度
    qa = quantize_weights_bias(a).sum()
    qa.backward()
    print("直接求量化权重梯度", a.grad)