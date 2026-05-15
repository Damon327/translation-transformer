#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/12 13:46 
"""
from pathlib import Path

# 1. 路径配置 (最优先)
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# 2.特殊符号
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'


BEST_MODEL = 'best_model.pt'
ZH_VOCAB_FILE = 'zh_vocab.txt'
EN_VOCAB_FILE = 'en_vocab.txt'

#文件
RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train'
TEST_DATA_FILE = 'test'


# 超参数
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 128

DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2


