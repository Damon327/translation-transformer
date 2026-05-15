#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：translation-transformer 
@File    ：tokenizer.py
@IDE     ：PyCharm 
@Author  ：Clark
@Date    ：2026/5/12 09:13 
"""
from nltk import TreebankWordDetokenizer
from nltk.tokenize import TreebankWordTokenizer
from src.config import *


class BaseTokenizer():

    def __init__(self,vocab_list):
        self.vocab_size = len(vocab_list)
        self.id2word = vocab_list
        self.word2id = { words:id for id,words in enumerate(vocab_list) }
        #记录特殊token的id
        self.unk_id = self.word2id[UNK_TOKEN]
        self.pad_id = self.word2id[PAD_TOKEN]
        self.sos_id = self.word2id[SOS_TOKEN]
        self.eos_id = self.word2id[EOS_TOKEN]

    #分词
    @classmethod
    def tokenize(cls, text)  -> list[str]:
        pass

    #构建词表：传入原始语料句子，及保存词表文件路径
    @classmethod
    def build_vocab(cls, sentences, vocab_path):
        vocab_set = set()
        #遍历语料，分词并去重
        for sentence in sentences:
            words = cls.tokenize(sentence)
            vocab_set.update(words)

        #转换为词表，增加特殊token
        vocab_list = [PAD_TOKEN ,UNK_TOKEN,SOS_TOKEN,EOS_TOKEN] + list(vocab_set)
        print(f"Vocab size: {len(vocab_list)} ")

        #保存成文件
        with open(vocab_path, 'w', encoding= 'utf-8') as f:
            f.write('\n'.join(vocab_list))

    #工厂方法：读取词表文件，创建分词器实例对象
    @classmethod
    def from_vocab(cls, vocab_path):
        with open(vocab_path, 'r', encoding = 'utf-8') as f:
            vocab_list = [ line.strip() for line in  f.readlines()]
        return cls(vocab_list)

    #编码
    def encode(self, text, target = False):
        tokens  = self.tokenize(text)
        if target:
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        ids = [self.word2id.get(token,self.unk_id) for token in tokens]
        return ids


class ZhTokenizer(BaseTokenizer):

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)


class EnTokenizer(BaseTokenizer):

    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    #解码方法
    def decode(self, ids):
        tokens = [ self.id2word[id] for id in ids ]
        return self.detokenizer.detokenize(tokens)



if __name__ == '__main__':
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR/ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR/EN_VOCAB_FILE)

    print("中文词表大小",zh_tokenizer.vocab_size)
    print("英文词表大小",en_tokenizer.vocab_size)
    print(zh_tokenizer.unk_id)
    print(zh_tokenizer.sos_id)
    print(zh_tokenizer.eos_id)

    text = "我爱刘亦菲"
    print(zh_tokenizer.encode(text))

    text = 'hello world!'
    print(en_tokenizer.encode(text,target=True))




