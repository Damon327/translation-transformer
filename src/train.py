#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/14 15:38 
"""
import time

import torch
from torch import nn, optim

from config import *
from dataset import get_dataloader
from model import TranslationModel
from tokenizer import ZhTokenizer, EnTokenizer

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 底层方法：训练一个轮次，返回平均损失
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    # 按批次迭代
    for input, target in tqdm(train_loader, desc='Train'):
        input = input.to(device)
        target = target.to(device)

        # 0. 准备工作
        decoder_input = target[:, :-1]
        decoder_target = target[:, 1:]
        # 填充掩码
        src_padding_mask = (input == model.src_padding_idx)
        tgt_padding_mask = (decoder_input == model.tgt_padding_idx)
        # 因果性掩码
        tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).bool().to(device)

        # 1. 前向传播，得到输出 (N, T, tgt_vocab_size)
        decoder_output = model(input, decoder_input, src_padding_mask, tgt_mask, tgt_padding_mask)
        # 2. 计算损失
        loss = loss_fn(decoder_output.mT, decoder_target)
        # 3. 反向传播
        loss.backward()
        # 4. 更新参数
        optimizer.step()
        # 5. 梯度清零
        optimizer.zero_grad()

        # 累加损失
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 整体训练流程
def train():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 获取训练集加载器
    train_loader = get_dataloader()

    # 3. 获取分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 4. 定义模型
    model = TranslationModel(
        zh_tokenizer.vocab_size,
        en_tokenizer.vocab_size,
        zh_tokenizer.pad_id,
        en_tokenizer.pad_id,
    ).to(device)

    # 5. 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 引入可视化工具
    writer = SummaryWriter(log_dir=LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 6. 开始训练
    min_loss = float('inf')
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}')
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f'Train Loss: {train_loss}')
        writer.add_scalar('Train Loss', train_loss, epoch+1)
        # 保存最优模型
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), MODELS_DIR/BEST_MODEL)
            print("Best model saved")
    writer.close()

if __name__ == '__main__':
    train()
