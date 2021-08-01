# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
Highlight：
    1、设定推理方法以及结果的选取机制

@author: Zhenyu Wu
"""
import torch
from torch.nn import functional as F


# 从模型的输出中选取top_k元素，将其余的元素用filter_value替换
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def fast_sample_sequence(model, context, date_stamp, length, temperature=1.0, top_k=30, top_p=0.0, device='cpu', cluster=None):
    # 将输入的行为序列转为张量作为inputs，同时会把变量inputs加载到GPU中，时间跨度序列不需要此操作
    inputs = torch.LongTensor(context).view(1, -1).to(device)
    cluster_inputs = [cluster%2038]*len(date_stamp[1][:-1])
    cluster_inputs = torch.tensor([cluster_inputs]).long().to(device)
    # 如果我们输入的多于一个单词，那么以输入的最后一个单词开始续写
    if len(context) > 1:
        # _标示模型的输出，torch.Size([1, 126, 85806])
        # 返回的past是一个元组，存储着12个decoder中每一个decoder的权重(注意力头分裂，每一个注意力头的权重信息)
        # model的返回值为元组
        output = model(inputs[:, :-1], past=None, time_stamp=date_stamp[1][:-1], cluster=cluster_inputs)
        _ = output[0]
        past = output[1]
        time_log = output[-1]
        # 输入语句中的最后一个单词
        prev = inputs[:, -1].view(1, -1)
        # 这里需要注意的是，我们整个过程并不能推理得到时间间隔信息，只能得到分窗信息，这一部分和行为预测模型是不一样的
        prev_date = [date_stamp[1][-1]]
        prev_date_log = [time_log[-1]]
    # 如果我们输入的只有一个单词，那么就以该单词开始续写
    else:
        past = None
        prev = inputs
        prev_date = date_stamp[1]
        ############################ 初始化部分是对的，所以着重考虑之后的部分##################################
    # 保存生成序列，把目前已知的内容进行保存（不保存，也就是生成的内容都是推理产生的结果）
    generate = []
    # 反向传播时都不会自动求导，volatile可以实现一定速度的提升，并节省一半的显存，因为其不需要保存梯度
    with torch.no_grad():
        # 逐步生成每一个token
        for i in range(length):
            # print('时间ID差分值预测结果：{}'.format(prev_date))
            # print('时间ID预测结果：{}'.format(prev_date_log))
            output = model(prev, past=past, time_stamp=prev_date, state='Test', date_log=prev_date_log, cluster=torch.LongTensor([cluster]).view(1, -1).to(device))
            # output存储的是模型最后一层softmax的输出
            # past存储的是各层的权重
            prev_date = [output[-2]]
            prev_date_log = [output[-1]]
            output, past = output[:2]
            # output张量
            output = output[-1].squeeze(0) / temperature
            # 选取最大的top_k个元素，其他的均被置为-inf
            filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
            # 作用是对input的每一行做n_samples次取值，输出的张量是每一次取值时input张量对应行的下标
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            # 将最新生成的添加进去
            generate.append(next_token.item())
            # 更新输入语句中的最后一个单词
            prev = next_token.view(1, 1)
    return generate


def sample_sequence(model, context, length, date_stamp, n_ctx, tokenizer, temperature=1.0, top_k=30, top_p=0.0, repitition_penalty=1.0,
                    device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    # 改变维度
    context = context.unsqueeze(0)
    generated = context
    final_text = []
    with torch.no_grad():
        for _ in range(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            for id in set(generated):
                next_token_logits[id] /= repitition_penalty
            next_token_logits = next_token_logits / temperature
            # next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            final_text.append(next_token.item())
    # return generated.tolist()[0]
    return final_text


# 为样本生成后续内容
def generate(n_ctx, model, context, length, date_stamp, tokenizer, temperature=1, top_k=0, top_p=0.0, repitition_penalty=1.0, device='cpu',
             is_fast_pattern=False, cluster=None):
    if is_fast_pattern:
        # 快速生成样本
        return fast_sample_sequence(model, context, date_stamp, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                    device=device, cluster=cluster)
    else:
        # 正常生成样本
        return sample_sequence(model, context, date_stamp, length, n_ctx, tokenizer=tokenizer, temperature=temperature, top_k=top_k, top_p=top_p,
                               repitition_penalty=repitition_penalty, device=device)