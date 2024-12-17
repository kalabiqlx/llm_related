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
    # gr.Blocks 是 Gradio 中的一个高级布局容器，允许灵活地组合多个组件（比如文本框、按钮等）
    # 将当前 Blocks 实例命名为 demo，之后可以通过 demo.launch() 来启动
    
    with gr.Row(): # 将内部组件横向排列
        # 上传图片
        with gr.Column(scale=1): # 将组件纵向排列，形成列布局。scale=1：表示两列所占的空间比相等
                image_input = gr.Image(type="pil", label="选择图片")
                """
                gr.Image：提供上传图片的功能。
                type="pil"：将上传的图片转换为 PIL.Image 格式，方便后续处理。
                label="选择图片"：在 UI 界面上显示的标签名称。
                """
        with gr.Column(scale=1):
            mode = gr.Radio(["pretrain", "sft"], label="选择模型") 
            # 两种模型选择，gr.Radio：提供单选按钮。label="选择模型"：UI 上显示的描述标签
            text_input = gr.Textbox(label="输入文本") # gr.Textbox：文本输入框。label="输入文本"：输入框上方显示的标签。
            text_output = gr.Textbox(label="输出文本")
            generate_button = gr.Button("生成") # gr.Button：按钮组件。"生成"：按钮上显示的文本
            generate_button.click(generate, inputs=[mode, image_input, text_input], outputs=text_output)
            """
            generate_button.click：绑定按钮的点击事件。
            generate：回调函数，当按钮被点击时执行这个函数。
            inputs：传入回调函数的输入数据，这里包括：
            mode：模型选择（"pretrain" 或 "sft"）。
            image_input：用户上传的图片（PIL 格式）。
            text_input：用户输入的文本。
            outputs：回调函数的输出，这里将结果显示在 text_output 文本框中。
            """
            

if __name__ == "__main__": # 确保只有当脚本被直接运行时，下面的代码才会执行。如果该文件被作为模块导入到其他脚本中，if 语句中的代码将不会执行。
    demo.launch(share=False, server_name="0.0.0.0", server_port=7891)
    """
    launch() 方法用于启动 Gradio Web 服务器，打开应用程序。
    share 参数控制是否生成一个可以公开访问的链接。
    server_name 参数指定服务器绑定的网络地址。
        "0.0.0.0"：表示将服务器绑定到所有可用的网络接口。
        这使得该应用可以被局域网中的其他设备访问。
        如果设置为 "127.0.0.1"，则只能在本地访问。
    server_port 指定 Web 服务器监听的端口号。
        7891：自定义端口，访问时可以使用 http://<ip地址>:7891。
        默认端口 是 7860，通过设置此参数可以避免与其他服务端口冲突。
    """
