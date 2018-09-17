import torch
from net import net_quantize_guide
from torchvision import models


x = torch.ones(5, 3)
bias = torch.ones(5, 1)
bias[0][0] = 4
bias[3][0] = 3
y = x * bias
print(y)