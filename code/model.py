# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
Highlight：
    1、定义转发时间ID预测模型
    2、定义转发行为预测模型
    3、注意473~475的屏蔽方式
reference：
    1、https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
@author: Zhenyu Wu
"""
from layers import Block, Conv1D
import torch.nn as nn
import torch
import copy
import logging
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import os
import utils


logger = logging.getLogger(__name__)
WEIGHTS_NAME = "pytorch_model.bin"
            
            
class TimerGPT2(nn.Module):
    # 这个模块的设置来自于model_config_time.json
    def __init__(self, config_time):
        super(TimerGPT2, self).__init__()
        # 6个解码器
        self.n_layer = config_time.n_layer
        self.n_embd = config_time.n_embd
        # 这里的值可以通过set(list())得到
        # >>> date_token = list(set(list([j for i in train_date for j in i])+list([j for i in test_date for j in i])))
        self.n_vocab = config_time.vocab_size
        self.initializer_range = config_time.initializer_range
        self.output_past = config_time.output_past
        
        # 对时间间隔进行编码
        self.time_wte = nn.Embedding(config_time.vocab_size, config_time.n_embd)
        self.time_wpe = nn.Embedding(config_time.n_positions, config_time.n_embd)
        self.wce = nn.Embedding(config_time.n_cluster, config_time.n_embd)
        self.drop = nn.Dropout(config_time.embd_pdrop)
        
        # transformer解码器部分
        block = Block(config_time.n_ctx, config_time, scale=True)
        self.time_h = nn.ModuleList([copy.deepcopy(block) for _ in range(config_time.n_layer)])
        
        # decoder head
        self.time_ln_f = nn.LayerNorm(config_time.n_embd, eps=config_time.layer_norm_epsilon)
        self.apply(self._init_weights)
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
        
    # 参数初始化
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, time_stamp=None, cluster=None):
        """
        作用：定义模型的前向过程
        """
        # 输入维度计算
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if past is None:
            past_length = 0
            past = [None] * len(self.time_h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            # 这里生成结果和arange类似
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            # unsqueeze()这个函数主要是对数据维度进行扩充，给指定位置加上维数为一的维度
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 掩码注意力机制
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 注意力头掩码
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer
        
        # 模型部分
        # id转embedding
        cluster_ids = cluster.view(-1, input_shape[-1])
        inputs_embeds = self.time_wte(input_ids)
        position_embeds = self.time_wpe(position_ids)
        cluster_embeds = self.wce(cluster_ids)
        hidden_states = inputs_embeds + position_embeds + cluster_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.time_h, past)):
            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)
        hidden_states = self.time_ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        return outputs
    
    
class TimerGPT2LMHeadModel(nn.Module):
    def __init__(self, config_time):
        super(TimerGPT2LMHeadModel, self).__init__()
        # 配置文件
        self.initializer_range = config_time.initializer_range
        self.config = config_time
        # 层的结构化定义
        self.transformer = TimerGPT2(config_time)
        self.lm_head = nn.Linear(config_time.n_embd, config_time.vocab_size, bias=False)
        # 参数初始化配置
        self.apply(self._init_weights)
        self.tie_weights()
        
    def _init_weights(self, module):
        """
        作用：层定义中参数的初始化配置
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ 
        Tie or clone module weights depending of weither we are using TorchScript or not
        """
        first_module.weight = second_module.weight
        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

    def tie_weights(self):
        """
        作用：共享输入和输出层的embedding权重
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.time_wte)
        
    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, time_stamp=None, cluster=None):
        """
        作用：定义模型的前向过程
        """
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask, 
                                               time_stamp=time_stamp,
                                               cluster=cluster)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs
        # 通过这一行代码可以获取到模型的输出，也即返回结果的第二个元素
        # print(shift_logits.view(-1, shift_logits.size(-1)).argmax(dim=1))
        # 这一行代码获取最后一层的概率计算结果
        # print(shift_logits.view(-1, shift_logits.size(-1)))
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
    
    
class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.vocab_size
        self.initializer_range = config.initializer_range
        self.output_past = config.output_past
        
        # input embedding stem
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.wse = nn.Embedding(config.n_timestamp, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # transformer
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.apply(self._init_weights)
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))
    
    # 参数初始化
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, time_stamp=None):
        """
        作用：定义模型的前向过程
        """
        # 输入维度计算
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            # 这里生成结果和arange类似
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            # unsqueeze()这个函数主要是对数据维度进行扩充，给指定位置加上维数为一的维度
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # 掩码注意力机制
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # 注意力头掩码
        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.n_layer

        ##################################################
        ## 核心贡献点
        ##################################################
        #
        time_stamp = torch.tensor(time_stamp).long().to('cuda')
        time_stamp_ids = time_stamp.view(-1, input_shape[-1])

        # 模型部分
        # id转embedding
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        timer_embeds = self.wse(time_stamp_ids)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        # 将时间特性引入，这里需要考虑将时间嵌入的矩阵加到哪里更为合适
        hidden_states = inputs_embeds + position_embeds + token_type_embeds + timer_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        return outputs
            

class GPT2LMHeadModel(nn.Module):
    """
    >>> import config
    >>> from model import GPT2LMHeadModel
    
    >>> model_config = 'config/model_config.json'
    >>> model_config = config.GPT2Config.from_json_file(model_config)
    >>> print('config:\n' + model_config.to_json_string())
    
    >>> model = GPT2LMHeadModel(model_config)
    >>> model.train()
    >>> model.to(select_device)
    """
    def __init__(self, config, config_time):
        super(GPT2LMHeadModel, self).__init__()
        # 配置文件
        self.initializer_range = config.initializer_range
        self.config = config
        # 层的结构化定义
        self.transformer = GPT2Model(config)
        self.time_transformer = TimerGPT2LMHeadModel(config_time)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 参数初始化配置
        self.apply(self._init_weights)
        self.tie_weights()
        
    def _init_weights(self, module):
        """
        作用：层定义中参数的初始化配置
        """
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def _tie_or_clone_weights(self, first_module, second_module):
        """ 
        Tie or clone module weights depending of weither we are using TorchScript or not
        """
        first_module.weight = second_module.weight
        if hasattr(first_module, 'bias') and first_module.bias is not None:
            first_module.bias.data = torch.nn.functional.pad(
                first_module.bias.data,
                (0, first_module.weight.shape[0] - first_module.bias.shape[0]),
                'constant',
                0
            )

    def tie_weights(self):
        """
        作用：共享输入和输出层的embedding权重
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.wte)

        
    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, time_stamp=None, state='Train', date_log=None, cluster=None):
        #####################################################
        ## 创新点：在GPT-2的基础上加入对时间跨度的预测
        #####################################################
        time_id = time_stamp
        # 将时间ID的一阶差分转换为张量，并加载到GPU中
        time_stamp = torch.tensor(time_id).long().to('cuda')
        # 我们应该将train和test过程在这里分隔开,默认是训练过程
        if state=='Train':
            # 利用min-GPT训练模型，得到时间ID的一阶差分
            time_transformer_outputs = self.time_transformer(input_ids=time_stamp, labels=time_stamp, cluster=cluster)
            # 这里的时间ID应该采用模型的预测值
            shift_logits = time_transformer_outputs[1][..., :-1, :].contiguous()
            # 通过这一行代码可以获取到模型的输出，也即返回结果的第二个元素
            shift_logits = shift_logits.view(-1, shift_logits.size(-1)).argmax(dim=1)
            shift_logits = [0] + [time_stamp.cpu().numpy().tolist()[0]] + shift_logits.cpu().numpy().tolist()
            # 将时间ID间隔还原为时间ID
            """
            time_id = [0]
            for i in range(len(shift_logits)):
                temp = time_id[i]+shift_logits[i]
                time_id.append(temp)
            """
            time_id = torch.tensor(shift_logits).long().to('cuda')
            # time_transformer_outputs[0]中存放这个模型的loss值
            # print(time_transformer_outputs[0])

        if state=='Test':
            # _表示时间戳ID的概率值计算结果
            # timer_past表示timer_GPT的decoder权重计算结果
            _, timer_past = self.time_transformer(input_ids=time_stamp, cluster=cluster)[:2]
            timer_output = self.time_transformer(time_stamp, past=timer_past, cluster=cluster)
            timer_output, timer_past = timer_output[:2]
            timer_output = timer_output[-1].squeeze(0) / 1.0
            # 这里如果用top-k=1则得到的预测结果均为0，所以我们采用累计概率计算
            filtered_logits = utils.top_k_top_p_filtering(timer_output, top_p=0.9)
            next_time_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
            # 时间部分预测到的下一个时间戳ID
            timer_token = next_time_token.item()
            # 将时间ID间隔还原为时间ID
            """
            final_time_id = date_log[0]+timer_token
            final_time_id = torch.tensor(final_time_id).long().to('cuda')
            time_id = final_time_id
            """
            final_time_id = torch.tensor(timer_token).long().to('cuda')
            time_id = final_time_id
        
        
        """
        作用：定义模型的前向过程
        """
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask, 
                                               time_stamp=time_id)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            # 平衡两个损失函数在量级上的差异性，两个损失值在量级上相差大概6倍
            time_model_loss = time_transformer_outputs[0]*6
            # 损失合并
            all_loss = loss + time_model_loss
            # 这里是为了将每一部分损失值都打印出来
            outputs = (all_loss,) + outputs + (loss,) + (time_model_loss,)
        ################################################
        ## 注意此处的接口
        ################################################
        if state=='Test':
            return outputs + (timer_token,) + (time_id.cpu().numpy().tolist(),)
        if state=='Train':
            # 训练过程
            # return outputs            # (loss), lm_logits, presents, (all hidden_states), (attentions)
            # 测试过程
            return outputs + (time_id.cpu().numpy().tolist(),)
    
    def save_pretrained(self, save_directory):
        """ 
        Save a model and its configuration file to a directory, so that it
        can be re-loaded using the `:func:`~transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        self.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        logger.info("Model weights saved in {}".format(output_model_file))
