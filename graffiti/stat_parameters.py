# coding=utf-8
"""
resnet-18:
    layer1.0.conv1.weight                    0.003
    layer1.0.conv2.weight                    0.003
    layer1.1.conv1.weight                    0.003
    layer1.1.conv2.weight                    0.003
    layer2.0.conv1.weight                    0.006
    layer2.0.conv2.weight                    0.013
    layer2.1.conv1.weight                    0.013
    layer2.1.conv2.weight                    0.013
    layer3.0.conv1.weight                    0.025
    layer3.0.conv2.weight                    0.050
    layer3.0.downsample.0.weight             0.003
    layer3.1.conv1.weight                    0.050
    layer3.1.conv2.weight                    0.050
    layer4.0.conv1.weight                    0.101
    layer4.0.conv2.weight                    0.202
    layer4.0.downsample.0.weight             0.011
    layer4.1.conv1.weight                    0.202
    layer4.1.conv2.weight                    0.202
    fc.weight                                0.044
"""
import torchvision.models as models


def num_features(shape):
    feature = 1
    for dim in shape:
        feature *= dim
    return feature


def total_parameters(state_dict):
    count = 0
    for value in state_dict.values():
        count += num_features(value.size())
    return count


if __name__ == "__main__":
    model = models.resnet50()
    total = total_parameters(model.state_dict())
    for k, v in model.state_dict().items():
        rate = num_features(v.size())/total
        if rate > 0.001:
            print("{: <30} {:.3f}".format(k, rate))
