#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：predict.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/14 17:20 
"""
import torch

from config import *
from model import TranslationModel
from tokenizer import ZhTokenizer, EnTokenizer

# 预测核心逻辑：传入一批数据，自回归生成，得到目标语言id列表
def predict_batch(model, inputs, tgt_tokenizer, device):
    model.eval()
    # 前向传播
    with torch.no_grad():
        inputs = inputs.to(device)
        # 1. 编码
        src_padding_mask = (inputs == model.src_padding_idx)
        memory = model.encode(inputs, src_padding_mask)

        # 2. 解码（自回归生成）
        # 2.1 定义初始输入：N个<sos>，形状(N, T=1)
        N = inputs.shape[0]
        decoder_inputs = torch.full((N, 1), tgt_tokenizer.sos_id).to(device)

        # 2.2 循环迭代，设置最大长度
        generated = []
        is_end = torch.full((N, ), False).to(device)
        for i in range(MAX_SEQ_LEN):
            # 2.2.1 准备掩码
            tgt_padding_mask = (decoder_inputs == model.tgt_padding_idx)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]).bool().to(device)

            # 2.2.2 前向传播（解码），输出形状 (N, T, tgt_vocab_size)
            decoder_outputs = model.decode(decoder_inputs, memory, tgt_mask, tgt_padding_mask, src_padding_mask)

            # 2.2.3 提取最后一个位置的输出，贪心解码，得到形状 (N, 1)
            next_token_ids = torch.argmax( decoder_outputs[:, -1], dim=-1 ).unsqueeze(-1)

            # 2.2.4 拼接得到的id序列，更新解码器输入，得到形状 (N, T+1)
            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)

            # 保存到列表，元素都是(N, 1)的Tensor
            generated.append(next_token_ids)

            # 2.2.5 判断是否应该结束生成
            is_end |= ( next_token_ids.squeeze(1) == tgt_tokenizer.eos_id )
            if is_end.all():
                break

        # 自回归生成结束，拼接生成结果
        generated_list = torch.cat(generated, dim=-1).tolist()

        # 截掉<eos>以及之后的token
        for i, id_list in enumerate(generated_list):
            eos_pos = id_list.index( tgt_tokenizer.eos_id )
            generated_list[i] = id_list[:eos_pos]

        return generated_list

# 预测主流程：传入用户文本（中文），利用模型推理，得到译文（英文）返回
def predict(text, model, src_tokenizer, tgt_tokenizer, device):
    # 1. 处理输入数据
    ids = src_tokenizer.encode(text)
    input = torch.tensor([ids])
    # 2. 调用核心预测逻辑，得到预测输出
    result = predict_batch(model, input, tgt_tokenizer, device)
    # 3. 解码得到英文句子
    return tgt_tokenizer.decode(result[0])

# 应用程序
def run_predict_app():
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 获取分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 3. 定义并加载模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_id,
        en_tokenizer.pad_id,
    ).to(device)
    model.load_state_dict( torch.load(MODELS_DIR/BEST_MODEL) )
    print("模型加载成功！")

    # 4. 程序主流程
    print("欢迎使用中文翻译小程序！输入 q 或者 quit 退出...")
    while True:
        # 等待用户输入
        user_input = input("中文：")

        if user_input in ["q", "quit"]:
            print("欢迎下次再来！")
            break
        if user_input.strip() == "":
            print("请输入有效内容...")
            continue

        # 调用预测逻辑
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)

if __name__ == '__main__':
    run_predict_app()