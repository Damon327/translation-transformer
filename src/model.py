#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/13 20:43 
"""
import math

import torch
from torch import nn
from torch.nn import Transformer

from config import  *

#自定义模型类
class TranslationModel(nn.Module):

    #初始化
    def __init__(self,src_vocab_size,tgt_vocab_size,src_padding_idx=0,tgt_padding_idx=0):
        super().__init__()
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        #定义所有层结构
        #词嵌入层
        self.src_embedding = nn.Embedding(
            num_embeddings=src_vocab_size,
            embedding_dim=DIM_MODEL,
            padding_idx=src_padding_idx
        )

        self.tgt_embedding = nn.Embedding(
            num_embeddings=tgt_vocab_size,
            embedding_dim=DIM_MODEL,
            padding_idx=tgt_padding_idx
        )

        #定义位置编码层
        self.pos_encoding = PositionEncoding(MAX_SEQ_LEN,DIM_MODEL)

        #transformer层
        self.transformer = Transformer(
            d_model=DIM_MODEL,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            batch_first=True
        )

        #线性层
        self.linear = nn.Linear(in_features=DIM_MODEL,out_features=tgt_vocab_size)

    #前向传播
    def forward(self,src_seq,tgt_seq,src_padding_mask,tgt_mask,tgt_padding_mask):
        #编码
        memory = self.encode(src_seq,src_padding_mask)
        #解码
        output = self.decode(tgt_seq,memory,tgt_mask,tgt_padding_mask,src_padding_mask)
        return output

    #编码
    def encode(self,src_seq,src_padding_mask):
        #1词嵌入
        embed = self.src_embedding(src_seq)
        #2叠加位置编码向量
        src = self.pos_encoding(embed)
        #3编码器前向传播
        memory = self.transformer.encoder(src=src,src_key_padding_mask=src_padding_mask)
        return memory

    #解码
    def decode(self,tgt_seq,memory,tgt_mask,tgt_padding_mask,memory_padding_mask):
        #1词嵌入
        embed = self.tgt_embedding(tgt_seq)
        #2叠加位置编码向量
        tgt = self.pos_encoding(embed)
        #3解码器前向传播
        output = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        #线性层整合特征，映射到词表
        output = self.linear(output)
        return output


#自定义位置编码层
class PositionEncoding(nn.Module):

    def __init__(self, max_len, d_model):
        super().__init__()
        # 定义一个位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 预计算：遍历每一行（每个位置），进行编码计算
        for pos in range(max_len):
            for _2i in range(0, d_model, 2):
                # 代入公式，分别计算奇偶数位置的值
                pe[pos, _2i] = math.sin( pos / (10000 ** (_2i / d_model)) )
                pe[pos, _2i+1] = math.cos( pos / (10000 ** (_2i / d_model)) )

        self.register_buffer('pe', pe)

    # 前向传播，传入一批数据的词向量 (N, L, d_model)
    def forward(self, x):
        # 在pe矩阵中，截取对应长度L的向量
        part_pe = self.pe[0 : x.shape[1]]
        # 广播叠加返回
        return x + part_pe


if __name__ == '__main__':
    # 定义词表大小
    src_vocab_size = 1000
    tgt_vocab_size = 1500
    # 定义模型
    model = TranslationModel(src_vocab_size, tgt_vocab_size)

    # 定义数据
    src_seq = torch.randint(src_vocab_size, (BATCH_SIZE, 20))
    tgt_seq = torch.randint(tgt_vocab_size, (BATCH_SIZE, 17))

    # 填充掩码
    padding_idx = 0
    src_padding_mask = (src_seq == padding_idx)
    tgt_padding_mask = (tgt_seq == padding_idx)
    # 因果性掩码
    tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_seq.shape[1]).bool()

    # 前向传播
    output = model(src_seq, tgt_seq, src_padding_mask, tgt_mask, tgt_padding_mask)

    print(output.shape)




