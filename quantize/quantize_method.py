# coding=utf-8
"""
"""
import torch
import math
import numpy as np


# 量化比特
QUANTIZE_BIT = 8


class QuantizeTanh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        n = math.pow(2.0, QUANTIZE_BIT) - 1
        return torch.round(i * n) / n

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class QuantizeGEMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        n = math.pow(2.0, QUANTIZE_BIT) - 1
        v_max = torch.max(i)
        v_min = torch.min(i)
        scale = (v_max - v_min)/n
        scale = max(scale, 1e-8)
        zero_point = torch.round(torch.clamp(-v_min/scale, 0, n))
        quantize_val = torch.clamp(torch.round(i/scale + zero_point), 0, n)
        return (quantize_val-zero_point) * scale

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


quantize_tanh = QuantizeTanh.apply
quantize_gemm = QuantizeGEMM.apply


def quantize_weights_bias_tanh(weight):
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

    return 2 * quantize_tanh(norm_weight) - 1


def quantize_activations_tanh(activation):
    activation = torch.clamp(activation, 0.0, 1.0)
    return 2 * quantize_tanh(activation) - 1


def quantize_weights_bias_gemm(weight):
    return quantize_gemm(weight)


def quantize_activations_gemm(activation):
    return quantize_gemm(activation)

