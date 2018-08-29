# coding=utf-8
import torch
import torchvision.models as models
from quantize.quantize_method import quantize_weights_bias_tanh
import numpy as np


checkpoint = "/home/wangbin/Desktop/uisee/model_quantize/AandW_lr1e-3_step10_epoch35/checkpoint.pth.tar"


def weight_decay():

    """
       random init:
       l2_loss:            (668.1154)
       l2_loss * 1e-4:     (0.066812)

       pre-trained
       l2_loss:            (517.5516)
       l2_loss * 1e-4:     (0.051755)
    """

    model = models.resnet18()
    l2_loss = 0
    for i in model.parameters():
        l2_loss += i.norm(p=2)

    print(l2_loss)
    print(l2_loss * 1e-4)


def quantize_weight_distribute():
    model_checkpoint = torch.load(checkpoint)
    state_dict = model_checkpoint['state_dict']

    for k, v in state_dict.items():
        if k == "module.layer1.1.conv2.weight":
            cnts = [0 for _ in range(26)]
            v = v.view(-1)
            print(v)
            v = (quantize_weights_bias_tanh(v) + 1) / 2 * (256 - 1)
            print(v.size())
            for ele in v:
                cnts[np.abs(int(ele)//10)] += 1
            for i in range(26):
                print(i, " ", '{:.4f}'.format(cnts[i]/len(v)))

    # 权值越在深层, 方差越小, 越底层, 分布范围越大, 方差越大
    """
    conv4.1_layer
    0   0.0000
    1   0.0000
    2   0.0000
    3   0.0000
    4   0.0000
    5   0.0000
    6   0.0000
    7   0.0000
    8   0.0000
    9   0.0000
    10   0.0009
    11   0.0717
    12   0.5933
    13   0.3055
    14   0.0257
    15   0.0022
    16   0.0003
    17   0.0001
    18   0.0000
    19   0.0000
    20   0.0000
    21   0.0000
    22   0.0000
    23   0.0000
    24   0.0000
    25   0.0000
    """

    '''
    conv1.1_layer
    0   0.0001
    1   0.0000
    2   0.0001
    3   0.0002
    4   0.0004
    5   0.0007
    6   0.0019
    7   0.0032
    8   0.0084
    9   0.0204
    10   0.0566
    11   0.1618
    12   0.3274
    13   0.2621
    14   0.1029
    15   0.0341
    16   0.0116
    17   0.0050
    18   0.0019
    19   0.0005
    20   0.0004
    21   0.0002
    22   0.0001
    23   0.0000
    24   0.0000
    25   0.0000

    '''


if __name__ == "__main__":
    quantize_weight_distribute()