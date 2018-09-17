# coding=utf-8
import torch
from torchvision import models
import numpy as np
import os
from net import net_bn_conv_merge, net_bn_conv_merge_quantize
from utils.data_loader import load_val_data
from utils.train_val import validate

epsilon = 1e-5
data = "/media/wangbin/8057840b-9a1e-48c9-aa84-d353a6ba1090/ImageNet_ILSVRC2012/ILSVRC2012"

model = models.resnet18(pretrained=True)
# merge_model = net_bn_conv_merge.resnet18()
merge_model = net_bn_conv_merge_quantize.resnet18()
state_dict = model.state_dict()
merge_state_dict = merge_model.state_dict()

# for name in state_dict:
#     print(name)

merge_state_dict.update({"fc.weight": state_dict["fc.weight"],
                        "fc.bias": state_dict["fc.bias"]})
del state_dict["fc.weight"]
del state_dict["fc.bias"]
params = np.array(list(state_dict.keys()))

params = params.reshape((-1, 5))
for index in range(params.shape[0]):
    weight = state_dict[params[index][0]]
    gamma = state_dict[params[index][1]]
    beta = state_dict[params[index][2]]
    running_mean = state_dict[params[index][3]]
    running_var = state_dict[params[index][4]]
    delta = gamma/(torch.sqrt(running_var+epsilon))
    weight = weight * delta.view(-1, 1, 1, 1)
    bias = (0-running_mean) * delta + beta
    merge_state_dict.update({params[index][0]: weight,
                             params[index][0][:-6] + "bias": bias})
merge_model.load_state_dict(merge_state_dict)
merge_model_name = "resnet18_merge_bn_conv.pth.tar"
torch.save(merge_model.state_dict(), merge_model_name)

"""
    conv1.weight
    bn1.weight
    bn1.bias
    bn1.running_mean
    bn1.running_var
    layer1.0.conv1.weight
    layer1.0.bn1.weight
    layer1.0.bn1.bias
    layer1.0.bn1.running_mean
    layer1.0.bn1.running_var
"""

# print("bn1.weight: \n", len(state_dict["bn1.weight"]), state_dict["bn1.weight"])
# print("bn1.bias: \n", len(state_dict["bn1.bias"]), state_dict["bn1.bias"])
# print("bn1.running_mean: \n", state_dict["bn1.running_mean"])
# print("bn1.running_val: \n", state_dict["bn1.running_var"])

val_loader = load_val_data(data)
evaluate = merge_model_name
if os.path.isfile(evaluate):
    print("Loading evaluate model '{}'".format(evaluate))
    checkpoint = torch.load(evaluate)
    merge_model.load_state_dict(checkpoint)
    print("Loaded evaluate model '{}'".format(evaluate))
else:
    print("No evaluate mode found at '{}'".format(evaluate))

merge_model.cuda()
merge_model.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()
validate(merge_model, val_loader, criterion)
