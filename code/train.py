# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020

@author: Zhenyu Wu
"""
#################################
## 导入必要的包
#################################
# 系统设定部分
import os
import json
from datetime import datetime
from tqdm import tqdm
import warnings

# 模型设定部分
from model import GPT2LMHeadModel
import config
import transformers 
import torch                                        
from torch.utils.tensorboard import SummaryWriter      # 损失可视化
from torch.nn import DataParallel                      # 数据并行化

# 查看包的版本号
print('{}的版本号为{}'.format('transformers', transformers.__version__))
print('{}的版本号为{}'.format('torch', torch.__version__))
warnings.filterwarnings("ignore")


#################################
## 参数初始化
#################################
# 设置使用哪些显卡
device = '0, 1'   
# 选择模型参数，这个可以自定义，需要修改vocab_size这个参数适应于自己的数据集
model_config = 'config/model_config.json'
# 时间部分的小模型超参定义
time_submodel_config = 'config/model_config_time.json'
# 原始训练语料，我们这里选择train_encode.json，一个元素是一个样本
raw_data_path = 'data/train_encode.json'
# 训练集样本的时间戳
raw_date_path = 'data/train_span.json'
# 训练循环数目
epochs = 100
# 训练batch size
batch_size = 1
# 学习率
lr = 1.5e-4
# warm up步数，前多少步学习率是逐步增加的，这里根据GPT-2的论文选取
warmup_steps = 2000
# 多少步汇报一次loss，设置为gradient accumulation的整数倍
log_step = 1280
# 训练时取训练数据的窗口步长，这个主要是根据我们句子长度设定的
stride = 768
# 梯度积累
gradient_accumulation = 128
max_grad_norm = 1.0
# 模型输出路径
output_dir = 'model_save/'
# 模型训练起点路径，这里可以设定下次训练的起点位置
pretrained_model = ''
# 当我们需要接着训练时用下面这个代码
# pretrained_model = 'model_save/model_epoch1/pytorch_model.bin'
# Tensorboard路径，用于训练过程的可视化
writer_dir = 'tensorboard_summary/'
# encoder.json，这里我们自己利用用户昵称编码生成encoder.json
encoder_json = 'data/encoder.json'
# 每20轮保存一次训练结果
save_every = 1


#################################
## 模型初始化配置
#################################
# 此处设置程序使用哪些显卡，由于服务器只有两张显卡，所以我们用0号和1号显卡
os.environ["CUDA_VISIBLE_DEVICES"] = device
# 加载大模型的参数
model_config = config.GPT2Config.from_json_file(model_config)
print('config:\n' + model_config.to_json_string())
# 加载时间戳处理部分模型的参数
time_model_config = config.GPT2Config.from_json_file(time_submodel_config)
print('config:\n' + time_model_config.to_json_string())
# 因果掩模的维度（通常与n_positions相同）。
n_ctx = model_config.n_ctx
# 这里选择GPU训练模型
select_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', select_device)
# 设定tensorboard中损失值等信息的存储地址
tb_writer = SummaryWriter(log_dir=writer_dir)
# 断言，log_step必须为gradient accumulation的整数倍
assert log_step % gradient_accumulation == 0
# 模型的存储路径
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


#################################
## 加载编码好的训练数据
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
# 加载编码后的用户转发行为数据
train_data_encode = Read_Data_Json(raw_data_path)
# 样例测试
print('encoder of a sample in train dataset: {}'.format(train_data_encode[0]))
print('number of samples for train: {}'.format(len(train_data_encode)))
# 加载用户转发行为对应的时间戳
train_date = Read_Data_Json(raw_date_path)
# 样例测试
print('time stamp of a sample in train dataset: {}'.format(train_date[0]))


#################################
## 模型训练的配置
#################################
# 我们针对自己的问题选择从头开始训练
if not pretrained_model:
    print("Not Pre-trained")
    model = GPT2LMHeadModel(config=model_config, config_time=time_model_config)
# 当训练了一段时间之后，想接着训练就从check point加载已经训练好的权重
else:
    print("From Pre-trained")
    model = GPT2LMHeadModel(config=model_config, config_time=time_model_config)
    model.load_state_dict(torch.load(pretrained_model))
    if hasattr(model, 'tie_weights'):
        model.tie_weights()
# 模型配置为训练模式，并加载到GPU中进行训练
model.train()
model.to(select_device)
# 打印模型结构
print(model)
# 模型的参量，我们这里用的是132M的模型，也就是GPT-small
num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of parameters: {}'.format(num_parameters))
# 计算训练多少步
multi_gpu = False
full_len = 0
print('calculating total steps')
full_len = sum([len(i) for i in train_data_encode])
total_steps = int(full_len / stride * epochs / batch_size / gradient_accumulation)
print('total steps = {}'.format(total_steps))
# 优化器设定AdamW，并采用学习率退火
optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
# 多卡训练
"""
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model, device_ids=[0, 1], output_device=1)
    model = model.cuda()
    multi_gpu = True
"""


#################################
## 模型的训练过程
#################################
def _get_time_id(time_span_sum):
    # 按间隔分窗,这里的6表示6小时
    time_window = 6
    flag = []
    window_flag = 0
    for i in range(len(time_span_sum)):
        window_flag = ((time_span_sum[i]) // time_window)
        flag.append(int(window_flag))
    # 分箱结果做差分，方便于模型训练
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
    # 我们是要对id_diff进行训练，然后将其还原为time_id
    time_id, id_diff = _get_time_id(time_span_sum)
    return id_diff

print('starting training')
overall_step = 0
running_loss = 0
try:
    for epoch in range(epochs):
        ######################################################
        ## 数据预处理部分
        ######################################################
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        piece_num = 0
        # 将样本一条一条喂入
        for i in range(len(train_data_encode)):
            tokens = train_data_encode[i]
            start_point = 0
            samples = []
            # 将样本切分为等长数据，我们设定参数大于最长行为序列长度，所以整个训练过程是挨个样本送入
            while start_point < len(tokens) - n_ctx:
                samples.append(tokens[start_point: start_point + n_ctx])
                start_point += stride

            if start_point < len(tokens):
                samples.append(tokens[len(tokens) - n_ctx:])
            ######################################################
            ## 模型训练部分
            ######################################################
            for step in range(len(samples) // batch_size):
                #  数据准备
                batch = samples[step * batch_size: (step + 1) * batch_size]
                batch_inputs = []
                for ids in batch:
                    int_ids = [int(x) for x in ids]
                    batch_inputs.append(int_ids)
                # print(batch_inputs)
                # 我们这里都是将整条样本一次送入
                cluster_inputs = [piece_num%2038]*len(_calc_time_span(train_date[i]))
                cluster_inputs = torch.tensor([cluster_inputs]).long().to(select_device)
                batch_inputs = torch.tensor([tokens]).long().to(select_device)

                # 前向过程，我们这里送入的time_stamp是一个list，表示时间ID的一阶差分
                # 这里是整个模型训练的核心过程
                outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs, time_stamp=_calc_time_span(train_date[i]), cluster=cluster_inputs)
                loss, logits = outputs[:2]
                
                #  计算损失
                if multi_gpu:
                    loss = loss.mean()
                if gradient_accumulation > 1:
                    loss = loss / gradient_accumulation

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                #  优化步骤
                if (overall_step + 1) % gradient_accumulation == 0:
                    running_loss += loss.item()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                if (overall_step + 1) % log_step == 0:
                    # 可视化
                    tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)

                    print('now time: {}:{}. Step {} of piece {} of epoch {}, total loss {}'.format(
                        datetime.now().hour,
                        datetime.now().minute,
                        step + 1,
                        piece_num,
                        epoch + 1,
                        running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                    running_loss = 0
                overall_step += 1
            piece_num += 1

        # 模型的保存，如果从检查点加载模型继续训练注意修改epoch + 1后面加的这个常数，应该改为epoch + 1 + 加载检查点的文件加编号
        if epoch % save_every == 0:
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))

# 中断服务并保存模型，可以把损失值也保存到文件名中
except KeyboardInterrupt:
    print('interrupted')
    print('saving model for epoch {}'.format(epoch + 1))
    if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
        os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
print('saving model for epoch {}'.format(epoch + 1))

print('epoch {} finished'.format(epoch + 1))
then = datetime.now()
print('time: {}'.format(then))
print('time for one epoch: {}'.format(then - now))

print('training finished')
if not os.path.exists(output_dir + 'final_model'):
    os.mkdir(output_dir + 'final_model')
model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir + 'final_model')
