# coding=utf-8
import torch

a = torch.ones(2, 2, requires_grad=True).cuda(1)
b = torch.rand(2, 2, requires_grad=True).cuda(2)
c = a + b

print(c)