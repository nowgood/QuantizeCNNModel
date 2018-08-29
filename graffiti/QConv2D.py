# coding=utf-8
import torch
import torch.nn as nn
from quantize.quantize_method import QuantizeWeightOrActivation
import torch.nn.functional as F
quantize = QuantizeWeightOrActivation()


class QConv2D(torch.nn.Conv2d):
    def __init__(self, n_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(QConv2D, self).__init__(n_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, bias)
        nn.init.constant_(self.weight, 1)

    def forward(self, x):
        qweight = quantize.quantize_weights_bias(self.weight)
        x = F.conv2d(x, qweight)
        return x


if __name__ == "__main__":
    qconv = QConv2D(1, 1, 3)
    qconv.zero_grad()
    x = torch.ones(1, 1, 3, 3, requires_grad=True).float()
    y = qconv(x)
    y.backward()
    print(qconv.weight.grad)

    a = torch.ones(3, 3, requires_grad=True).float()
    w = torch.nn.init.constant_(torch.empty(3, 3, requires_grad=True), 1)
    qw = quantize.quantize_weights_bias(w)

    z = (qw * a).sum()
    z.backward()
    print(w.grad)

    qa = quantize.quantize_weights_bias(a).sum()
    qa.backward()
    print(a.grad)