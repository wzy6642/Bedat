# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
Highlight：
    1、定义模型的推理过程，我们的模型需要先对转发时间ID进行推理，然后对推理结果做embedding并加入到行为预测的过程中
    2、注意time_window变量与训练过程中保持一致
@author: Zhenyu Wu
"""
#################################
## 导入必要的包
#################################
# 系统设定部分
import os
import json
import regex as re
from tqdm import tqdm

# 模型设定部分
import torch
import config
from model import GPT2LMHeadModel
import torch.nn.functional as F
import utils


# 设置使用哪些显卡
device = '1'
# 生成长度，-1表示model.config.n_ctx，也即2048
length = 100
# 生成的batch size
batch_size = 1
# 生成几个样本
nsamples = 1
# 生成温度,越大随机性越强，越小确定性越高
temperature = 1
# 最高几选一,默认模型中这里是40
topk = 8
# 最高积累概率
topp = 0
# 选择模型参数，这个可以自定义，需要修改vocab_size这个参数适应于自己的数据集，应该是encoder.json文件中元素个数+1
model_config = 'config/model_config.json'
# 时间部分的小模型超参定义
time_submodel_config = 'config/model_config_time.json'
# 选择词库，我们在这里使用基于数据集生成的vocab_processed.txt
tokenizer_path = 'data/vocab_processed.txt'
# 用于推理的模型路径
model_path = 'model_save/model_epoch8/pytorch_model.bin'
# 采用更加快的方式生成文本
# 原来这里需要赋值为False才可以
fast_pattern = True
# 保存产生的样本
save_samples = True
# 保存样本的路径
save_samples_path = 'inference'
# 重复惩罚值
repetition_penalty = 1.0
# encoder.json
encoder_json = 'data/encoder.json'
# 原始训练语料，我们这里选择test_encode.json
raw_data_path = 'data/test_encode.json'
# 测试数据的时间信息
test_date_path = 'data/test_span.json'


#################################
## 模型初始化配置
#################################
# 此处设置程序使用哪些显卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载编码文件
with open(encoder_json, 'r', encoding='utf8')as fp:
    encoder = json.load(fp)
# 生成解码文件
decoder = {v:k for k,v in encoder.items()}
# 加载GPT-2模型
model_config = config.GPT2Config.from_json_file(model_config)
time_model_config = config.GPT2Config.from_json_file(time_submodel_config)
model = GPT2LMHeadModel(config=model_config, config_time=time_model_config)
model.load_state_dict(torch.load(model_path))
if hasattr(model, 'tie_weights'):
    model.tie_weights()
model.eval()
model.to(device)
n_ctx = model.config.n_ctx


#################################
## 加载测试数据集
#################################
def Read_Data_Json(path):
    """
    作用：从JSON格式中读取数据
    参数：
        path -> 要读取的路径（str）
    返回：
        json_data -> 加载的JSON文件（json）
    """
    with open(path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    return json_data
# 当前样本生成序列的长度，默认为2048
if length == -1:
    length = model.config.n_ctx
# 生成样本的保存位置
if save_samples:
    if not os.path.exists(save_samples_path):
        os.makedirs(save_samples_path)
    samples_file = open(save_samples_path+'/inference_data_8.txt', 'w', encoding='utf8')
# 加载测试数据
test_data = Read_Data_Json(raw_data_path)    # 转发行为数据
test_date = Read_Data_Json(test_date_path)   # 转发时间戳数据


#################################
## 模型推理部分
#################################
def _get_time_id(time_span_sum):
    # 按间隔分窗,这里的6表示6小时,也即将时间按照固定窗口分箱,然后获取时间ID
    time_window = 6
    flag = []
    window_flag = 0
    for i in range(len(time_span_sum)):
        window_flag = ((time_span_sum[i]) // time_window)
        flag.append(int(window_flag))
    flag_diff = [int(flag[i + 1] - flag[i]) for i in range(len(flag) - 1)]
    return flag, flag_diff


def _calc_time_span(time_stamp):
    """
    示例：
    time_stamp = [12, 11, 14, 69]
    result = _calc_time_span(time_stamp)
    >>> time_id -> [0, 0, 0, 1, 4]
    >>> id_diff -> [0, 0, 1, 3]
    """
    time_span_sum = [sum(time_stamp[:i]) for i in range(len(time_stamp) + 1)]
    # time_span_sum = time_span_sum[::-1]
    time_id, id_diff = _get_time_id(time_span_sum)
    return time_id, id_diff


# 由于测试，所以这里只用部分数据，全量计算只需要把[:1]去掉即可
index = 0
for test_data_index in tqdm(range(len(test_data))):
    # 转发行为序列
    sub_data = test_data[test_data_index]
    # 相邻转发行为之间的时间跨度序列
    sub_date = test_date[test_data_index]
    generated = 0
    for _ in range(nsamples // batch_size):
        # 这里返回编码后的推理结果
        out = utils.generate(
                n_ctx=n_ctx,
                model=model,
                context=sub_data, date_stamp=_calc_time_span(sub_date),
                length=length,
                is_fast_pattern=fast_pattern, tokenizer=None,
                temperature=temperature, top_k=topk, top_p=topp, repitition_penalty=repetition_penalty, device=device, cluster=index
            )
        for i in range(batch_size):
            generated += 1
            text = [decoder[i] for i in out]
            # info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
            text = ' '.join(text)
            if save_samples:
                # samples_file.write(info)
                samples_file.write(text+' <|ThisIsOneUser|> ')
        if generated == nsamples:
            continue
    index += 1
if save_samples:
    samples_file.close()
