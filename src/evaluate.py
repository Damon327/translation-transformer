#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：evaluate.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/14 17:37 
"""
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from config import *
from model import TranslationModel
from tokenizer import ZhTokenizer, EnTokenizer
from predict import predict_batch
from dataset import get_dataloader

# 评估逻辑
def evaluate(model, test_loader, tgt_tokenizer, device):
    model.eval()
    # 定义列表，记录所有的预测结果和参考译文
    predictions = []
    references = []
    with torch.no_grad():
        # 循环迭代，遍历测试集每一批数据
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.tolist()
            # 前向传播
            results = predict_batch(model, inputs, tgt_tokenizer, device)
            # 将本批次的预测结果和参考译文加入列表
            predictions.extend(results)
            references.extend( [ [target[1:target.index(tgt_tokenizer.eos_id)]] for target in targets ] )

    # 得到bleu评分
    return corpus_bleu(references, predictions)

# 评估主流程
def run_evaluate():
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
    model.load_state_dict(torch.load(MODELS_DIR / BEST_MODEL))
    print("模型加载成功！")

    # 4. 获取测试集加载器
    test_loader = get_dataloader(train=False)

    # 5. 评估
    bleu = evaluate(model, test_loader, en_tokenizer, device)

    print("评估结果-BLEU:", bleu)

if __name__ == '__main__':
    run_evaluate()