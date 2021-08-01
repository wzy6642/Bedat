"""
Created on Fri Sep 4 11:25:40 2020
定义GPT-2模型中所有用到的层

reference：https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_gpt2.py
           https://github.com/graykode/gpt-2-Pytorch/blob/master/GPT2/model.py

备注：
    1. LayerNorm使用PyTorch自带的层归一化，eps参数来自于config.py文件：nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
    
@author: Zhenyu Wu
"""
from activations import ACT2FN
import torch.nn as nn
import torch
import math


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        # 随机初始化的 nx 行，nf 列的张量
        w = torch.empty(nx, nf)
        # 利用正态分布 - N(mean, std) 对张量进行初始化
        nn.init.normal_(w, std=0.02)
        # 利用Parameter指定可以优化的参数
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        # https://pytorch.org/docs/master/generated/torch.addmm.html
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    """
    GPT-2中的注意力层
    Args:
        nx：输入特征的维度
        n_ctx：位置掩码维度
        config：模型配置文件
    """
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        # self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e4 * (1 - b)
        
        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask
        # 这里的计算结果即为注意力权重矩阵
        w = nn.Softmax(dim=-1)(w)
        
        visualize=False
        if visualize:
            # 每次迭代都会显示12个结果代表12个解码器注意力输出
            # 每一个输出中又包含12个矩阵表示12个注意力头的输出
            # 每一个输出的后两维表示注意力矩阵的计算结果
            # 前12个为训练样本的注意力权重计算结果
            # 后边的为目前生成的token与之前内容的依赖程度（权重）
            # 可视化就是针对w[0,0,:,:]进行分析，从而得到：目前的模型是否能通过pisition embedding学到距离更近，影响更大这层含义？这个问题的答案
            # print(w[0,0,:,:])
            from matplotlib import pyplot as plt  
            plt.figure(figsize=(25, 14))
            plt.imshow(w[0,0,:,:].cpu().detach().numpy(), cmap='Blues')  
            # plt.colorbar()
            plt.show() 

        w = self.attn_dropout(w)
        
        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask
        
        return [torch.matmul(w, v)]

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        
        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)


class MLP(nn.Module):
    """
    GPT-2中的多层感知器
    Args:
        n_state：输入特征的维度
        config：模型配置文件
    """
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Block(nn.Module):
    """
    GPT-2中的解码器模块
    Args:
        n_ctx：因果掩码维度
        config：模型配置文件
    """
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):
        output_attn = self.attn(self.ln_1(x), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask)
        a = output_attn[0]  # output_attn: a, present, (attentions)
        
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        
        outputs = [x] + output_attn[1:]
        return outputs  # x, present, (attentions)