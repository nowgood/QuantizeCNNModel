# coding=utf-8
import torch
import torchvision.models as models

CUDA_VISIBLE_DEVICES = 0, 3
model = models.resnet18(pretrained=True)
model = torch.nn.DataParallel(model, [0]).cuda()

state_dict = model.state_dict()

second_last_convlayer_weight = state_dict['module.layer4.1.conv1.weight']
last_convlayer_weight = state_dict['module.layer4.1.conv2.weight']
print(second_last_convlayer_weight)
print(last_convlayer_weight)
print(last_convlayer_weight.norm(p=2))
l1 = torch.norm(last_convlayer_weight, p=2)
print(l1)

print(len(list(model.modules())), type(model.modules))
print(state_dict.keys())