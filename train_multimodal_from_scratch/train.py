from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
import zipfile
from PIL import Image
import io
import os
import json
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from typing import List, Dict, Any

class VLMConfig(PretrainedConfig): # 预训练配置
    model_type = "vlm_model" #  模型类型的标识符，序列化到 JSON 文件中，并用于在AutoConfig中重新创建正确的对象。用于之后用transformer库直接调用文件
    def __init__(self,llm_model_path = '/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                 vision_model_path = '/home/user/Downloads/siglip-so400m-patch14-384',
                 freeze_vision_model = True,
                 image_pad_num = 49,
                **kwargs):
        self.vision_model_path = vision_model_path
        self.llm_model_path = llm_model_path
        self.freeze_vision_model = freeze_vision_model
        self.image_pad_num = image_pad_num
        super().__init__(**kwargs) # 调用父类 PretrainedConfig 的 __init__ 方法，将 **kwargs 传递给父类。
        
        
        
class VLM(PreTrainedModel):
    config_class = VLMConfig # config类是前面定义的VLMConfig类
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.vision_model = AutoModel.from_pretrained(self.config.vision_model_path) # 加载视觉模型
        self.processor = AutoProcessor.from_pretrained(self.config.vision_model_path) # 加载视觉模型的图像处理器
        self.llm_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path) # 自动加载自回归语言模型，无需手动指定模型类
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.llm_model_path) # 加载语言模型的分词器

        # 用于视觉与文本token的对齐，这里之所以有个*4是因为omni-vision中采用了对图像token的压缩（reshape），即将图像token序列的长度减小了4倍，同时增大了四倍的维度。
        self.linear1 = nn.Linear(self.vision_model.config.vision_config.hidden_size*4, self.llm_model.config.hidden_size)
        self.linear2 = nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size)
        
        # 预训练只训练两个线性层(与LLaVA一致)
        if self.config.freeze_vision_model:
            for param in self.vision_model.parameters():
                param.requires_grad = False
        for param in self.llm_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids, labels, pixel_values, attention_mask=None): # 前向传播过程
        text_embeds = self.llm_model.get_input_embeddings()(input_ids) # 返回模型的输入文本嵌入
        
        image_embeds = self.vision_model.vision_model(pixel_values).last_hidden_state # 返回模型的输入图像嵌入
        b, s, d = image_embeds.shape # (b, 14 * 14, d)
        image_embeds = image_embeds.view(b, -1, d*4)  # (b, 196, d) --> (b, 49, d*4) 压缩图片tokens
        image_features = self.linear2(F.silu(self.linear1(image_embeds))) # 图像token经过两层线性层之后得到的用于对齐的图像token序列
        
        text_embeds = text_embeds.to(image_features.dtype) # 将文本数据类型变为图像的数据类型
        
        inputs_embeds = self.merge_input_ids_with_image_features(image_features, text_embeds, input_ids) 
        outputs = self.llm_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = outputs[0]
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)), labels.view(-1).to(logits.device)
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits)
        
    def merge_input_ids_with_image_features(self, image_features, inputs_embeds, input_ids):
        
        num_images, num_image_patches, embed_dim = image_features.shape # (b, 49, d)
        batch_indices, image_indices = torch.where(input_ids == self.tokenizer('<|image_pad|>')['input_ids'][0])
        # 用分词器将文本中表示图像的特殊占位符<|image_pad|>分词
        # 对整个 input_ids 张量进行比较，生成一个布尔矩阵，标记出等于 <|image_pad|> 的位置
        # torch.where：找出布尔矩阵中为 True 的位置，返回两个索引数组：
        # batch_indices：标识每个匹配位置所属的批次索引。
        # image_indices：标识每个匹配位置在序列中的索引。
        
        inputs_embeds[batch_indices, image_indices] = image_features.view(-1, embed_dim) # (b*49, d)
        # 使用 batch_indices 和 image_indices 找到 inputs_embeds 中占位符<|image_pad|>对应位置，将这些位置的嵌入替换为image_features
        
        return inputs_embeds
    
class MyDataset(Dataset): # 处理数据集
    def __init__(self, images_path, data_path, tokenizer, processor, config):
        super().__init__()
        self.data_path = data_path
        self.images_path = images_path
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        with open(self.data_path, 'r', encoding='utf-8') as f:
            # 只读模式打开文件，指定文件编码为 UTF-8（确保可以正确读取包含中文、特殊字符等的文件）
            # 使用 with 语句确保文件在操作完成后会自动关闭，避免资源泄露
            self.datas = json.load(f)   # 将文件对象 f 中的 JSON 数据读取并解析为 Python 对象（通常是字典或列表）
        
            
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index): 
        # 这是 Python 数据加载类常用的方法（如继承自 torch.utils.data.Dataset）
        # index：表示要访问的数据样本的索引
        
        sample = self.datas[index]
        try:
            image_name = sample['image']
            conversations = sample['conversations'] # 一组对话，包含用户和助手的内容。其中conversations[0]：用户输入（问题）。conversations[1]：助手回复（答案）

            # 问题文本
            # 使用 tokenizer 的 聊天模板 来格式化对话数据
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":conversations[0]['value']}], \
                tokenize=False, \ # 不立即进行分词，保持文本格式
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num) 
                # 添加一个生成文本的提示符，方便模型生成回复
                # 将数据集中的图像占位符<image>替换为qwen2.5中图像占位符的表示<|image_pad|>

            # 答案文本
            # 将助手的回答（conversations[1]['value']）与结束符 eos_token 拼接在一起，标识回复的结束
            a_text = conversations[1]['value'] + self.tokenizer.eos_token
            
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids # 问题部分的 token 用 pad_token_id 填充，因为这些位置不需要计算损失。回答部分使用真实的 a_input_ids 作为标签

            # 为了实现 自回归语言模型 的训练方式，即：模型输入是前面的 token，目标是预测下一个 token
            input_ids = input_ids[:-1] # 移除最后一个 token
            labels = labels[1:] # 移除第一个 token
        
            image = Image.open(os.path.join(self.images_path, image_name)).convert("RGB") # 使用 PIL（Python Imaging Library）打开指定路径下的图像文件并转换为RGB格式
            pixel_values = self.processor(text=None, images=image)['pixel_values'] # 预处理图像数据
            
        except: # 如果图像加载失败（例如文件不存在或读取错误），代码捕获异常
            default_image = Image.new('RGB', (224, 224), color='white') # 创建一个默认的白色图像，尺寸为 224x224 像素
            pixel_values = self.processor(text=None, images=default_image)['pixel_values']
            q_text = self.tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":"图片内容是什么\n<image>"}], \
                tokenize=False, \
                add_generation_prompt=True).replace('<image>', '<|image_pad|>'*self.config.image_pad_num)
            a_text = '图片内容为空' + self.tokenizer.eos_token
            q_input_ids = self.tokenizer(q_text)['input_ids']
            a_input_ids = self.tokenizer(a_text)['input_ids']
            input_ids = q_input_ids + a_input_ids
            labels = [tokenizer.pad_token_id] * len(q_input_ids) + a_input_ids
            input_ids = input_ids[:-1]
            labels = labels[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'pixel_values': pixel_values
        } 
     

class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    # __call__ 方法使得类实例可以像函数一样调用
    # 输入是一个列表，每个元素是一个字典，代表一个样本。
    # 每个样本包含以下键值对：
    # input_ids：文本的 token ID 列表。
    # labels：训练时的目标 token ID 列表。
    # pixel_values：图像数据，通常是经过预处理的张量
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(len(feature['input_ids']) for feature in features) # 遍历所有样本获取input_ids 的最大长度
        input_ids = []
        labels = []
        pixel_values = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids']))) # 填充
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            pixel_values.append(feature['pixel_values'])
            
        return {'input_ids': torch.tensor(input_ids, dtype=torch.long), # 将填充后的 input_ids 列表转换为 PyTorch 张量，并表示为长整型
                'labels': torch.tensor(labels, dtype=torch.long),
                'pixel_values': torch.cat(pixel_values, dim=0)} # 将批次中的所有图像张量拼接（concatenate）在一起, (batch_size, ...)
            
        
        
if __name__ == '__main__':
    config = VLMConfig(vision_model_path='/home/user/wyf/siglip-base-patch16-224', image_pad_num=49)
    model = VLM(config).cuda()
    print(model)
    print(f'模型参数量为：{sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    images_path = './dataset/LLaVA-CC3M-Pretrain-595K/images'
    data_path = './dataset/Chinese-LLaVA-Vision-Instructions/LLaVA-CC3M-Pretrain-595K/chat-translated.json'
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    processor = AutoProcessor.from_pretrained(config.vision_model_path)
    output_dir = 'save/pretrain' 
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=8,
        logging_steps=100,
        report_to='tensorboard',
        dataloader_pin_memory=True,
        dataloader_num_workers=1
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=MyDataset(images_path, data_path, tokenizer, processor, config),
        data_collator=MyDataCollator(tokenizer)  
    )
    
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('save/pretrain')
    trainer.save_state()
    
    

    
    
