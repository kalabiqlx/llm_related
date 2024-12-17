import gradio as gr
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoConfig
from PIL import Image
from train import VLMConfig, VLM
import torch
from torch.nn import functional as F
device = "cuda:1"
processor = AutoProcessor.from_pretrained("/home/user/wyf/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained('/home/user/Downloads/Qwen2.5-0.5B-Instruct')
AutoConfig.register("vlm_model", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, VLM)

pretrain_model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_multimodal_from_scratch/save/pretrain')
pretrain_model.to(device)

sft_model = AutoModelForCausalLM.from_pretrained('/home/user/wyf/train_multimodal_from_scratch/save/sft')
sft_model.to(device)

pretrain_model.eval()
sft_model.eval()
def generate(mode, image_input, text_input, max_new_tokens = 100, temperature = 0.0, top_k = None):
    q_text = tokenizer.apply_chat_template([{"role":"system", "content":'You are a helpful assistant.'}, {"role":"user", "content":f'{text_input}\n<image>'}], \
            tokenize=False, \
            add_generation_prompt=True).replace('<image>', '<|image_pad|>'*49)
    input_ids = tokenizer(q_text, return_tensors='pt')['input_ids'] # tokenizer在输出'输入ID列表'时，还在其顶部添加了一个维度：这是为了处理多个序列而新增的一个维度，故input_ids.shape=torch.Size([序列数，长度])
    input_ids = input_ids.to(device)
    # image = Image.open(image_input).convert("RGB")
    pixel_values = processor(text=None, images=image_input).pixel_values
    pixel_values = pixel_values.to(device)
    eos = tokenizer.eos_token_id
    
    s = input_ids.shape[1] # input_ids的长度
    while input_ids.shape[1] < s + max_new_tokens - 1: # 已生成的 token 数量未达到上限时，继续生成新的 token
        if mode == 'pretrain':
            model = pretrain_model
        else:
            model = sft_model
        inference_res = model(input_ids, None, pixel_values)  # 之所以有三个参数是对应我们训练的sft和预训练模型中前向过程的input_ids, label, pixel_values。因为这里是推理，不需要label，所以是None
        logits = inference_res.logits # 形状为(batch_size, sequence_length, vocab_size)
        logits = logits[:, -1, :]  # 取最后一个时间步的预测结果，因为自回归模型每次只需要预测下一个 token

        for token in set(input_ids.tolist()[0]): # 遍历当前 input_ids 序列中所有已出现的 token
            logits[:, token] /= 1.0 # 通过除法（这里除以 1.0，实际未改变权重）对重复 token 的概率进行惩罚，可能是实现去重的基础

        if temperature == 0.0: # 当 temperature == 0.0 时，模型输出的概率分布完全确定，取最大概率的 token（贪心采样）
            _, idx_next = torch.topk(logits, k=1, dim=-1) # 用于返回输入张量中指定维度上最大的 k 个元素 及其对应的 索引
        else: # 当 temperature > 0.0 时，对 logits 进行缩放，改变概率分布的平滑程度：
            logits = logits / temperature  
            if top_k is not None: # 保留前 top_k 个概率最高的 token，将其他 token 的概率设置为 -Inf，确保它们不会被采样。
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) # 不指认dim，默认为最后一维
                logits[logits < v[:, [-1]]] = -float('Inf') 

            probs = F.softmax(logits, dim=-1) # 将logits转换为概率分布。
            idx_next = torch.multinomial(probs, num_samples=1, generator=None) # 根据概率分布随机采样一个 token,返回其索引
            """
            torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
                input：输入的概率分布张量（Tensor），要求所有元素非负且可以归一化为概率分布。
                num_samples：要采样的元素个数。
                replacement（默认 False）：
                False：不放回采样（每次采样时，选中过的元素不会再次被选中）。
                True：有放回采样（每次采样都从原始分布重新抽取）。
                generator（可选）：用于控制随机数生成的生成器。
                out（可选）：输出结果的张量
            """

        if idx_next == eos: # 如果是结束符，停止
            break

        input_ids = torch.cat((input_ids, idx_next), dim=1)  # 将新生成的 token idx_next 添加到 input_ids 中，作为下一步模型的输入
    return tokenizer.decode(input_ids[:, s:][0]) # 取输出部分

with gr.Blocks() as demo:
    with gr.Row():
        # 上传图片
        with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="选择图片")
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrain", "sft"], label="选择模型") # 两种模型选择
            text_input = gr.Textbox(label="输入文本")
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成")
            generate_button.click(generate, inputs=[mode, image_input, text_input], outputs=text_output)
            

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7891)
    
