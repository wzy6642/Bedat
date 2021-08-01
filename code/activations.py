# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
这个文件对所有可能用到的激活函数进行统一整理，如果有更好的激活函数在这个文件添加即可

reference：https://github.com/huggingface/transformers/blob/master/src/transformers/activations.py
示例：
    >>> from activations import ACT2FN
    >>> import torch
    
    # 选定激活函数
    >>> act = ACT2FN["gelu_new"]
    
    # 计算结果
    >>> act(torch.tensor([1., 2., 3.]))
@author: Zhenyu Wu
"""
import math
import torch
import torch.nn.functional as F


def swish(x):
    return x * torch.sigmoid(x)

    
def _gelu_python(x):
    """
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """
    Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    
def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


# 版本控制，在torch==1.6中有gelu函数，而在之前的版本中并没有这个函数
if torch.__version__ < "1.4.0":
    gelu = _gelu_python
else:
    gelu = F.gelu


# 激活函数列表，如果用到新的激活函数，直接在这个字典中添加即可
ACT2FN = {
    "relu": F.relu, 
    "swish": swish, 
    "gelu": gelu, 
    "tanh": torch.tanh, 
    "gelu_new": gelu_new, 
    "gelu_fast": gelu_fast, 
}


def get_activation(activation_string):
    """
    作用：
        获取要用的激活函数，如果模型指定的激活函数在本文件中，则返回该激活函数，如果不在本文件中将给出提示语
    参数：
        activation_string：str型，必要参数
            指定要用的激活函数名称
    返回：
        要用的激活函数
    """
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))