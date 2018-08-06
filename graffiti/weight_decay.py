# coding=utf-8
import torch
import torchvision.models as models


model = models.resnet18()
l2_loss = 0
for i in model.parameters():
    l2_loss += i.norm(p=2)

print(l2_loss)
print(l2_loss * 1e-4)

"""
random init:
l2_loss:            (668.1154)
l2_loss * 1e-4:     (0.066812)

pre-trained
l2_loss:            (517.5516)
l2_loss * 1e-4:     (0.051755)
"""