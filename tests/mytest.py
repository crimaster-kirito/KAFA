# import torch
# print(torch.cuda.is_available())  # True
# print(torch.cuda.get_device_name(0))  # NVIDIA GeForce RTX 4090
#
# print(torch.__version__)
# print(torch.version.cuda)
# print(torch.cuda.is_available())  #输出为True，则安装无误

from transformers import AutoModelForCausalLM, AutoTokenizer

# 替换为你的本地模型绝对路径
local_model_path = "/home/tyx/ywp/FedBiOT/models/llama-2-7b-hf"

try:
    # 仅加载本地文件，禁用网络下载
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True)
    print("本地模型+Tokenizer 加载成功！")
except Exception as e:
    print(f"加载失败：{e}")  # 若报错，说明缺少 Tokenizer 文件或路径错误