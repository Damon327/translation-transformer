#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：process.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/12 09:12 
"""
import pandas as pd
from PIL.ImagePalette import sepia
from sklearn.model_selection import train_test_split
from config import  *
from tokenizer import ZhTokenizer,EnTokenizer

def processed():
    print('Preprocessing... data ...')

    # 读取原始文件
    df = pd.read_csv(
        RAW_DATA_DIR/RAW_DATA_FILE,
        sep='\t',
        header=None,
        usecols = [0,1],
        names = ['en','zh'],
        encoding='utf-8'
    )
    df.dropna(inplace=True)
    #print(df.head())

    #划分数据集
    train_df,test_df = train_test_split(df, test_size=0.2)

    #创建词表
    ZhTokenizer.build_vocab(df['zh'].tolist(),MODELS_DIR/ZH_VOCAB_FILE)
    EnTokenizer.build_vocab(df['en'].tolist(),MODELS_DIR/EN_VOCAB_FILE)

    #创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    #编码:将原始语料，分词后转成ID列表
    zh_encode = lambda text: zh_tokenizer.encode(text)
    en_encode = lambda text: en_tokenizer.encode(text,target=True)
    train_df['zh'] = train_df['zh'].apply(zh_encode)
    train_df['en'] = train_df['en'].apply(en_encode)
    test_df['zh'] = test_df['zh'].apply(zh_encode)
    test_df['en'] = test_df['en'].apply(en_encode)

    #保存成josnl文件
    train_df.to_json(PROCESSED_DATA_DIR/TRAIN_DATA_FILE,orient='records',lines=True)
    test_df.to_json(PROCESSED_DATA_DIR/TEST_DATA_FILE,orient='records',lines=True)

    print('Preprocessing... done')

if __name__ == '__main__':
    processed()














