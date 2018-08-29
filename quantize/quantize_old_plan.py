# coding=utf-8
import torch
import torch.nn as nn
import queue
from quantize.quantize_method import QuantizeTanh


class QuantizeWeightOrActivation(object):
    def __init__(self):
        self.saved_param = queue.Queue()
        self.saved_grad = queue.Queue()
        self.quantize_fn = QuantizeTanh.apply  # 量化函数

    def quantize_weights_bias(self, weight):
        tanh_w = torch.tanh(weight)
        """
        torch 关于 y = w/max(|w|) 函数在max(|w|)处梯度行为怪异该如何解释?
        tensor w ([[ 0.1229,  0.2390],
                 [ 0.8703,  0.6368]])

        tensor y ([[ 0.2873,  0.2873],
                 [-0.3296,  0.2873]])
        由于没有搞清楚 torch 在 max(|w|) 处如何处理的, 
        不过, 从上面看出梯度为负数, y = w/max(|w|) w>0时, 梯度为负数, 我认为是不正确的.
        为了便于处理, 这里求梯度过程中, 我们把 max(|w|) 当成一个常量来处理,
        代码中通过 Tensor.data 这样求 max(|w|) 的过程就不会加入到计算图中,
        可以看出, max_abs_w 就是一个一个常量
        """
        max_abs_w = torch.max(torch.abs(tanh_w)).data
        norm_weight = ((tanh_w / max_abs_w) + 1) / 2

        return 2 * self.quantize_fn(norm_weight) - 1

    def quantize_activations(self, activation):
        activation = torch.clamp(activation, 0.0, 1.0)
        return self.quantize_fn(activation)

    def quantize(self, m):
        # isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            self.saved_param.put(m.weight.data.clone())  # 第1步, 保存全精度权重
            quantize_w = self.quantize_weights_bias(m.weight)  # 第2步, 量化权重
            quantize_w.sum().backward()
            self.saved_grad.put(m.weight.grad.data.clone())  # 第3步, 保存量化梯度
            m.weight.data.copy_(quantize_w.data)  # 第4步, 使用量化权重代替全精度权重
            # m.zero_grad() # 不需要, 因为后面调用 optimizer.zero_grad() 会把所有 m 的梯度清零

        if type(m) == nn.Linear:  # 量化 bias
            self.saved_param.put(m.bias.data.clone())
            quantize_b = self.quantize_weights_bias(m.bias)
            quantize_b.sum().backward()
            self.saved_grad.put(m.bias.grad.data.clone())
            m.bias.data.copy_(quantize_b.data)

    def restore(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            m.weight.data.copy_(self.saved_param.get())  # 第5步, 使用全精度权重代替量化权重

        if type(m) == nn.Linear:
            m.bias.data.copy_(self.saved_param.get())

    def update_grad(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            m.weight.grad.data.mul_(self.saved_grad.get())  # 第6步, 使用量化权重更新全精度权重

        if type(m) == nn.Linear:
            m.bias.grad.data.mul_(self.saved_grad.get())

    @staticmethod
    def info(net, s):
        print("\n-----------{}--------\n".format(s))
        for k, v in net.state_dict().items():
            print(k, "\n", v)
