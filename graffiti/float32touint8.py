import torch
from net import net_quantize_guide
from torchvision import models

# coding=utf-8
model = net_quantize_guide.resnet18()
print(model.state_dict().keys())
model = models.resnet18(pretrained=True)
state_dict = model.state_dict()
state_dict = {k: v.to(torch.uint8) for k, v in state_dict.items()}
torch.save(state_dict, "nowgood.pth")
