# coding=utf-8

import torch
import torch.nn as nn
from net import simple_net
import torch.optim as optim
from quantize.quantize_function import QuantizeWeightOrActivation
import queue

qw = QuantizeWeightOrActivation()


class MyFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):

        tanh_i = torch.tanh(i)
        max_w = torch.max(torch.abs(tanh_i)).data
        out = tanh_i / max_w
        ctx.save_for_backward(tanh_i, max_w)
        return out

    @staticmethod
    def backward(ctx, grad_outputs):
        by, bm, = ctx.saved_tensors
        return grad_outputs*((1-torch.pow(by, 2.0))/bm)


def modify_weights(weight):
    fn = MyFunction.apply
    return fn(weight)


def weights_update():
    feature_map = torch.ones(1, 1, 3, 3, requires_grad=True)
    kernel = nn.Conv2d(1, 1, kernel_size=3, bias=False)

    # start
    print("\n自动求导求量化梯度")
    # w = Variable(kernel.weight.data.clone(), requires_grad=True)
    w = kernel.weight
    y = torch.tanh(w)/torch.max(torch.abs(torch.tanh(w)))
    z = y.sum()
    z.backward()
    print(w.grad)
    kernel.zero_grad()
    # end

    print("权重初始化\n", kernel.weight.data, "\n")

    tanh_w = torch.tanh(kernel.weight)
    max_w = torch.max(torch.abs(tanh_w))
    hand_grad = (1 - torch.pow(kernel.weight, 2.0)) / max_w
    print("手动求梯度\n", hand_grad, "\n")   # 卷积核的面积=3x3=9, y=(x*x).mean(), y'=2x/9

    # fn_w = modify_weights(kernel.weight)
    fn_w = qw.quantize_weights_bias(kernel.weight)
    fn_w.sum().backward()

    square_weight_grad = kernel.weight.grad.data.clone()
    print("自动求梯度\n", square_weight_grad, "\n")  # 只需要在原本的梯度上乘以卷积核的面积就好

    print("量化前权重\n", kernel.weight.data, "\n")

    # 这种方式没法更新模型的权重, 看 state_dict 函数可以看出, 返回的是一个新建的有序字典,
    # 更新的其实是新字典, 而不是模型参数, 使用 load_state_dict 方法
    # kernel.state_dict().update(weight=fn_w)

    # state_dict = kernel.state_dict()  # 第 1 种方法更新权重
    # state_dict.update(weight=square)
    # kernel.load_state_dict(state_dict)

    # kernel.weight = nn.Parameter(square)  # 第 2 种方法更新权重

    kernel.weight.data.copy_(fn_w.data)  # 第 3 种方法更新权重

    print("量化后权重\n", kernel.weight.data, "\n")

    # 权重的另一个计算图
    other_graph = kernel(feature_map)
    other_graph.backward()

    print("不使用 Module.zer_grad(), 卷积后权重梯度\n", kernel.weight.grad, "\n")

    kernel.zero_grad()
    other_graph = kernel(feature_map)
    other_graph.backward()

    print("使用 Module.zer_grad(), 卷积后权重梯度\n", kernel.weight.grad, "\n")
    print("手动计算梯度更新(加法)\n", kernel.weight.grad + square_weight_grad, "\n")
    print("手动计算梯度更新(乘法)\n", kernel.weight.grad * square_weight_grad, "\n")


def module_apply():
    saved_param = queue.Queue()
    saved_grad = queue.Queue()

    def info(s):
        print("\n---{}---\n".format(s))

        for k, v in net.state_dict().items():
            print(k, v, "\n")
            break

    def square(module):
        if type(module) == nn.Conv2d:
            saved_param.put(module.weight.data.clone())  # 第一步, 保存全精度权重
            quantize_w = modify_weights(module.weight)  # 第二步, 量化权重
            quantize_w.sum().backward()
            saved_grad.put(module.weight.grad.data.clone())  # 第三步, 保存量化梯度
            module.weight.data.copy_(quantize_w.data)  # 第四步, 使用量化权重代替全精度权重

    def restore(module):
        if type(module) == nn.Conv2d:
            module.weight.data.copy_(saved_param.get())  # 第四步, 使用量化权重代替全精度权重

    def update_weight(module):
        if type(module) == nn.Conv2d:
            module.weight.grad.data.mul_(saved_grad.get())  # 第四步, 使用量化权重代替全精度权重

    net = simple_net.Net()
    info("初始化权重")

    # net.zero_grad()  # optimizer.zero_grad() is enough
    # 网络输入, 输出
    input_ = torch.ones(1, 1, 6, 6, requires_grad=True)
    lable = torch.ones(1, 2)

    optimizer = optim.SGD(net.parameters(), lr=1)
    criterion = nn.MSELoss()

    print("\n\n")

    print(net.state_dict().keys(), "\n")
    print(optimizer.param_groups)
    print(optimizer.state_dict())

    print("\n\n")

    for _ in range(5):

        net.apply(square)
        info("量化权重\n")
        print("net.conv1.weight.grad\n", net.conv1.weight.grad)
        output = net(input_)
        loss = criterion(output, lable)
        optimizer.zero_grad()  # very important!

        print("\nnet.conv1.weight.grad after optimizer.zero_grad()\n", net.conv1.weight.grad)

        loss.backward()

        net.apply(restore)
        info("恢复全精度权重")

        net.apply(update_weight)
        print(net.state_dict().keys(), "\n")

        optimizer.step()
        info("更新全精度权重")
        print(net.state_dict().keys(), "\n")

    torch.save(net.state_dict(), "../model/model_name_changed.pkl")
    xx = torch.load("../model/model_name_changed.pkl")
    print(xx.keys())


if __name__ == "__main__":
    module_apply()