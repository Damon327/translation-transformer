#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：dataset.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/13 18:04 
"""
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import  Dataset,DataLoader
from config import *
import torch.nn.utils.rnn as rnn_utils

#自定义数据集类
class TranslationDataset(Dataset):
    #初始化
    def __init__(self,data_path):
        #从文件中读书数据，得到字典的列表
        df = pd.read_json(data_path,lines=True)
        self.data = df.to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = self.data[idx]
        input = torch.tensor(data_dict['zh'])
        target = torch.tensor(data_dict['en'])

        #以Tensor元祖形式返回
        return input, target


#定义一个整理函数，把一个批次的样本的整合成一个大的Tensor，张量对其
def collate_fn(batch):
    input_list = [item[0] for item in batch]
    target_list = [item[1] for item in batch]

    #分别填充，构成张量
    #每个句子长度不同，把他们补齐到同一批这次最长句子的长度
    input_tensor = pad_sequence(input_list, batch_first=True, padding_value=0)
    target_list = pad_sequence(target_list, batch_first=True, padding_value=0)

    return input_tensor, target_list

def get_dataloader(train=True):
    data_path = PROCESSED_DATA_DIR/ (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    #创建数据集
    dataset = TranslationDataset(data_path)
    #创建数据集加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train,collate_fn=collate_fn)
    return dataloader


if __name__ == '__main__':
    train_dataset = TranslationDataset(PROCESSED_DATA_DIR/TRAIN_DATA_FILE)

    print(len(train_dataset))
    print(train_dataset[0])

    # 简单测试一下
    train_loader = get_dataloader(train=True)
    for inputs, targets in train_loader:
        print("输入批次形状:", inputs.shape)
        print("目标批次形状:", targets.shape)
        break










