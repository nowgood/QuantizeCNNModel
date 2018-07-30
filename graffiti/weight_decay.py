# coding=utf-8
import torch
import torchvision.models as models


model = models.resnet50(pretrained=True)
l2_loss = 0
for i in model.parameters():
    l2_loss += i.norm(p=2)

print(l2_loss)
print(l2_loss * 0.0001)

"""

"""