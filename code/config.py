# -*- coding: utf-8 -*-
"""
Created on Fri Sep 4 11:25:40 2020
本文件管理GPT2模型所有的配置，并且可以对配置信息进行读取（from_json_file）、打印（to_json_string）、存储（to_json_file）

reference：https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_gpt2.py
@author: Zhenyu Wu
"""
import json
import copy
import os
import logging


logger = logging.getLogger(__name__)
CONFIG_NAME = "config.json"


class GPT2Config():
    """
    作用：
        设定GPT-2模型的超参
    参数：
        vocab_size：int型，可选参数，默认值为50257
            GPT-2模型所使用的编码字典的大小。该参数定义了GPT-2模型的前向过程中使用tokens的多少。
        n_positions：int型，可选参数，默认值为1024
            GPT-2模型单次可以处理的最大序列长度。通常这个值设定为较大的数，例如512、1024、2048等等。
        n_ctx：int型，可选参数，默认值为1024
            因果掩码的维度，通常和n_positions取值相同。
        n_embd：int型，可选参数，默认值为768
            指定词嵌入维度以及隐层神经元个数。
        n_layer：int型，可选参数，默认值为12
            Transformer解码器中解码器的个数。
        n_head：int型，可选参数，默认值为12
            Transformer解码器的注意力层中注意力头的个数。
        n_inner：int型，可选参数，默认值为None
            前馈神经网络层的维度，默认值None表示维度为4*n_embd。
        activation_function：str型，可选参数，默认值为"gelu_new"
            可以从下述列表中选取激活函数：["relu", "swish", "gelu", "tanh", "gelu_new"]。
        resid_pdrop：float型，可选参数，默认值为0.1
            embedding、encoder、pooler层中全连接层的dropout概率。
        embd_pdrop：float型，可选参数，默认值为0.1
            embedding层的dropout概率。
        attn_pdrop：float型，可选参数，默认值为0.1
            注意力层的dropout概率。
        layer_norm_epsilon：float型，可选参数，默认值为1e-5
            在层归一化层中所使用的epsilon参数。
        initializer_range：float型，可选参数，默认值为0.02
            初始化所有权重矩阵的truncated_normal_initializer的标准差。
    示例：
        >>> import config
        >>> model_config = 'config/model_config.json'
        
        >>> # 初始化GPT-2配置
        >>> model_config = config.GPT2Config.from_json_file(model_config)
        
        >>> # 获取当前的配置
        >>> print('config:\n' + model_config.to_json_string())
    """
    model_type = "gpt2"
    def __init__(self, vocab_size=50257, n_positions=1024, n_ctx=1024, n_embd=768, n_layer=12, n_head=12, n_inner=None, activation_function="gelu_new", resid_pdrop=0.1, embd_pdrop=0.1, 
                 attn_pdrop=0.1, layer_norm_epsilon=1e-5, initializer_range=0.02, output_past=True, **kwargs):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.output_past = output_past
        # 用于对其他配置信息的更新
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    # 使用@property装饰器创建只读属性，@property装饰器会将方法转换为相同名称的只读属性，可以与所定义的属性配合使用，这样可以防止属性被修改
    @property
    def max_position_embeddings(self):
        return self.n_positions
    
    @property
    def hidden_size(self):
        return self.n_embd
    
    @property
    def num_attention_heads(self):
        return self.n_head
    
    @property
    def num_hidden_layers(self):
        return self.n_layer
    
    # classmethod修饰符对应的函数不需要实例化，不需要self参数，但第一个参数需要是表示自身类的cls参数，可以来调用类的属性，类的方法，实例化对象等
    # json_file: str这种表示方式会对参数类型进行检查，学名：类型注解(type hints)
    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
    
    @classmethod
    def from_json_file(cls, json_file: str) -> "GPT2Config":
        config_dict = cls._dict_from_json_file(json_file)
        return GPT2Config(**config_dict)
        
    def to_dict(self):
        """
        作用：
            将此实例序列转换为一个Python字典
        返回：
            Dict[str, Any]
                当前超参配置中所有属性构成的字典
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output
    
    def to_json_string(self) -> str:
        """
        作用：
            将该实例序列化为JSON字符串
        返回：
            本次模型的所有配置
        示例：
            >>> import config
            >>> model_config = 'config/model_config.json'
            
            >>> model_config = config.GPT2Config.from_json_file(model_config)
            >>> print('config:\n' + model_config.to_json_string())
        """
        config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    def to_json_file(self, json_file_path: str):
        """
        作用：
            将模型配置写入到JSON文件中
        示例：
            >>> import config
            >>> model_config = 'config/model_config.json'
            
            >>> model_config = config.GPT2Config.from_json_file(model_config)
            >>> model_config.to_json_file('test.json')
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
        
    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))