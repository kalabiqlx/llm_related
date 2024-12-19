import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import os
import pandas as pd

from torch.utils.data import IterableDataset, Dataset
import json
import numpy as np
from transformers import  PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PretrainedConfig
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator, DataCollatorForTokenClassification, AutoConfig
from dataset import SFTDataset, LLMDataset


class RMSNorm(nn.Module): # RMSNorm，源自LLaMA，与LayerNorm的区别在于没有减去均值
    def __init__(self, hidden_size, eps=1e-6):
        
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # RMSNorm 的可学习参数，类似于 LayerNorm 中的缩放参数 gamma，初始化为全 1，形状是 [hidden_size]
        self.variance_epsilon = eps # 一个非常小的常数（默认值为 1e-6），用于避免方差为 0 时除以零的数值不稳定问题

    def _norm(self, hidden_states: Tensor) -> Tensor: # hidden_states：形状为 [batch_size, seq_length, hidden_size] 的张量（如 Transformer 中的输入向量）
        variance = hidden_states.pow(2).mean(-1, keepdim=True) # 输入张量 hidden_states 的最后一个维度（即 hidden_size）进行平方平均
        return hidden_states * torch.rsqrt(variance + self.variance_epsilon) # torch.rsqrt() 计算倒数平方根
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.weight * self._norm(hidden_states.float()).type_as(hidden_states) 
        # 先强制转换为float类型，再把最终结果的类型转换回输入张量的原始类型
    
def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1) #  # 将张量 x 沿最后一个维度 (dim=-1) 平均分成两部分
    return torch.cat((-x2, x1), dim=-1) # # 交换两部分并对第二部分取反符号，最终拼接

def apply_rotate_pos_emb(q, k, cos, sin, unsqueeze_dim=2): # 旋转位置编码Rope
    """
    q: 查询向量（query），通常是注意力机制中的输入。
    k: 键向量（key），也是注意力机制中的输入。
    cos: 位置编码的余弦部分，维度应与 q 和 k 的最后一维一致。
    sin: 位置编码的正弦部分，维度同上。
    unsqueeze_dim=2: 指定在哪一维度上扩展 cos 和 sin 的维度（默认扩展第 2 维）
    """
    cos = cos.unsqueeze(unsqueeze_dim) # 通过 unsqueeze 方法，cos 和 sin 的维度被扩展了一维。
    sin = sin.unsqueeze(unsqueeze_dim) # 扩展是为了将 cos 和 sin 与 q 和 k 对应的最后一维进行广播（broadcasting），从而实现点乘运算

    # 对 q 和 k 分别应用旋转位置编码公式：
公式的核心是将原始向量的一部分通过正弦函数旋转，另一部分通过余弦函数旋转，达到在特征维度上对位置进行编码的目的
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    
    return q_embed, k_embed # 返回经过旋转位置编码处理的查询向量 q_embed 和键向量 k_embed

class RotaryEmbedding(nn.Module): # 旋转位置编码嵌入
    def __init__(self, dim, max_seq_len=1024): # dim 和 max_seq_len 被存储为类属性。
        super(RotaryEmbedding, self).__init__()
        self.dim = dim # dim: 特征维度（即嵌入向量的维度）
        self.max_seq_len = max_seq_len # max_seq_len: 最大序列长度（默认值为 1024）。这是位置编码支持的最大长度。
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float().unsqueeze(1)
        freqs = t @ inv_freq.unsqueeze(0)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", freqs.cos())
        self.register_buffer("sin_cached", freqs.sin())
        
    def forward(self, q, k):
        cos = self.cos_cached[:q.shape[1], :].unsqueeze(0)
        sin = self.sin_cached[:q.shape[1], :].unsqueeze(0)
        return apply_rotate_pos_emb(q, k, cos, sin)
    
def repeat_kv(hidden_states, n_rep):
    
    batch, slen, num_key_value_heads, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, slen, num_key_value_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, slen, num_key_value_heads * n_rep, head_dim)

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.k_cache, self.v_cache = None, None
        self.is_causal = True
        self.flash_attn = self.config.flash_attn

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.residual_dropout = nn.Dropout(self.dropout)
        self.attention_dropout = nn.Dropout(self.dropout)
        self.rotary_emb = RotaryEmbedding(self.head_dim)
        
    def forward(self, hidden_states, use_kv_cache=False):
        b, s = hidden_states.shape[:2]
        if use_kv_cache and self.eval():
            if self.k_cache is None or self.k_cache.shape[1] != s-1:
                q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            else:
                token = hidden_states[:, -1:, :]
                q = torch.cat((torch.zeros_like(hidden_states[:, :-1, :]), self.q_proj(token)), dim=1)
                k = torch.cat((self.k_cache, self.k_proj(token)), dim=1)
                v = torch.cat((self.v_cache, self.v_proj(token)), dim=1)
            self.k_cache, self.v_cache = k, v
            
        else:
            q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
            
        q = q.view(b, s, self.num_heads, self.head_dim)
        k = k.view(b, s, self.num_key_value_heads, self.head_dim)
        v = v.view(b, s, self.num_key_value_heads, self.head_dim)
        
        q, k = self.rotary_emb(q, k)
        
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        
        q = q.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        k = k.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        v = v.transpose(1, 2) # b, self.num_heads, s, self.head_dim
        
        if self.flash_attn:
        
            # q*k转置，（b, self.num_heads, s, self.head_dim）* (b, self.num_heads, self.head_dim，s) = （b, self.num_heads, s, s）
            # q*k/sqrt(self.head_dim)*v  （b, self.num_heads, s, s）* (b, self.num_heads, s, self.head_dim) = b, self.num_heads, s, self.head_dim
            output = F.scaled_dot_product_attention(q, k, v, attn_mask=None, 
                                                    dropout_p=self.dropout if self.training else 0.0, 
                                                    is_causal=self.is_causal) 
        else:
            mask = torch.full((1, 1, self.config.max_seq_len, self.config.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            scores = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + self.mask[:, :, :s, :s]  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(q)  # 计算 softmax
            scores = self.attention_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, v)  # 计算输出
        
        output = output.transpose(1, 2).contiguous().view(b, s, -1) # b, s, self.hidden_size
        
        output = self.o_proj(output)
        output = self.residual_dropout(output)
        return output
    
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        
    def forward(self, x):
        
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.layer_idx = layer_idx
    def forward(
        self,
        hidden_states,
        use_kv_cache
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            use_kv_cache=use_kv_cache
        )
        
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        outputs = residual + hidden_states
        return outputs
   
   
# 编写自定义配置时需要记住的三个重要事项如下：
# 1、必须继承自 PretrainedConfig
# 2、PretrainedConfig 的 __init__ 方法必须接受任何 kwargs
# 3、这些 kwargs 需要传递给超类的 __init__ 方法。
class Config(PretrainedConfig):
    model_type = "small_model"
    
    def __init__(self,
                hidden_size = 512,
                num_attention_heads = 16,
                num_key_value_heads = 8,
                flash_attn = True,
                attention_bias = False,
                max_seq_len = 512,
                intermediate_size = 2048,
                mlp_bias = False,
                vocab_size = 6400,
                n_layers = 8,
                dropout = 0.0,
                **kwargs):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.flash_attn = flash_attn
        self.attention_bias = attention_bias
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.mlp_bias = mlp_bias
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        super().__init__(**kwargs)
         

class LLM(PreTrainedModel):
    config_class = Config
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vocab_size = self.config.vocab_size
        self.n_layers = self.config.n_layers

        self.tokon_embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.dropout) 
        self.layers = torch.nn.ModuleList() 
        for layer_idx in range(self.n_layers):
            self.layers.append(DecoderLayer(self.config, layer_idx)) 
        self.norm = RMSNorm(self.config.hidden_size)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False) 
        self.tokon_embeddings.weight = self.output.weight
        self.apply(self._init_weights) 
        self.loss = None 
        
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)) 

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            
        
    def forward(self, input_ids, labels, use_kv_cache=False):
       
        hidden_states = self.tokon_embeddings(input_ids) 
        hidden_states = self.dropout(hidden_states)  
        for idx, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, use_kv_cache=use_kv_cache)  

        hidden_states = self.norm(hidden_states) 

        if labels is not None:
            logits = self.output(hidden_states)  
            self.loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0) 
        else:
            logits = self.output(hidden_states[:, [-1], :])  
            self.loss = None  

        return CausalLMOutputWithPast(self.loss, logits)
    
    @torch.inference_mode
    def generate(self, inputs, eos, max_new_tokens, temperature=0.7, top_k=None, stream=True, repetition_penalty=1.,
                 use_kv_cache=True):
        
        input_ids = inputs['input_ids']
        labels = inputs['labels']
        s = input_ids.shape[1]
        while input_ids.shape[1] < max_new_tokens - 1:  
            inference_res = self(input_ids, labels, use_kv_cache=use_kv_cache)  
            logits = inference_res.logits 
            logits = logits[:, -1, :] 

            for token in set(input_ids.tolist()[0]):  
                logits[:, token] /= repetition_penalty

            if temperature == 0.0: 
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature  
                if top_k is not None:  
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf') 

                probs = F.softmax(logits, dim=-1)  
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)  

            if idx_next == eos:  
                break

            input_ids = torch.cat((input_ids, idx_next), dim=1)  
            if stream:  
                yield input_ids[:, s:]  

        if not stream:  
            yield input_ids[:, s:]  
               
if __name__ == '__main__':   

    config = Config()
    model = LLM(config)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    data_collator = DefaultDataCollator()
    tokenizer = AutoTokenizer.from_pretrained("./tokenizer", use_fast=True)
    args = TrainingArguments(output_dir='./results2048', 
                            num_train_epochs=10, 
                            do_train=True, 
                            per_device_train_batch_size=128,
                            gradient_accumulation_steps=8,
                            # max_steps=15000,
                            logging_steps=100,
                            report_to='tensorboard',
                            save_total_limit=5,
                            bf16=True,
                            learning_rate=2e-4,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True,
                            save_safetensors=False)          
    dataset = LLMDataset('./mobvoi_seq_monkey_general_open_corpus.jsonl', tokenizer=tokenizer, max_seq_len=512)
    trainer = Trainer(model=model, args=args, train_dataset=dataset, tokenizer=tokenizer, data_collator=data_collator)
    # 如果是初次训练resume_from_checkpoint为false，接着checkpoint继续训练，为True
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves/model')
    trainer.save_state()
