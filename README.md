## 代码调用

```python
>>> from peft import PeftModel
>>> from transformers import AutoTokenizer, AutoModel
>>> model_path = "THUDM/chatglm2-6b"
>>> model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
>>> tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
>>> #  给模型加上LoRA参数 
>>> model = PeftModel.from_pretrained(model, "output/sft").half()
>>> model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=history)
>>> print(response)
```

## ChatGLM2-6B模型下载

[Hugging Face Hub](https://huggingface.co/THUDM/chatglm2-6b)下载ChatGLM2-6B的模型文件

## LoRA 微调

LoRA 微调脚本：

`dataset`, 分词后的数据集，即在 data/ 地址下的文件夹名称

`lora_rank`, 设置 LoRA 的秩，推荐为4或8，显存够的话使用8

`per_device_train_batch_size`, 每块 GPU 上的 batch size,显存不大尽量1-2

`gradient_accumulation_steps`, 梯度累加，可以在不提升显存占用的情况下增大 batch size

`save_steps`, 多少步保存一次

`save_total_limit`, 保存多少个checkpoint

`learning_rate`, 学习率

`output_dir`, 模型文件保存地址


```shell
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --model_name_or_path THUDM/chatglm2-6b \
    --stage sft \
    --use_v2 \
    --do_train \
    --dataset dataset \
    --finetuning_type lora \
    --lora_rank 8 \
    --output_dir ./output \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 4.0 \
    --fp16
```

修改 model_name_or_path 为本地已下载的 ChatGLM2-6B 模型即可。

BaiChuan-7B 微调脚本：

在 train_baichuan.py 中修改模型路径、数据集路径等参数
```shell
CUDA_VISIBLE_DEVICES=0 python src/train_baichuan.py \
    --lora_rank 8 \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --max_steps 600 \
    --save_steps 60 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 10 \
    --output_dir output/baichuan-sft
```

## 推理结果
```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel

# 加载原始LLM
model_path = "THUDM/chatglm2-6b"
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.chat(tokenizer, "你好",history=[])
```
```python
# 加载LoRA微调后的LLM
model = PeftModel.from_pretrained(model, "output").half()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```