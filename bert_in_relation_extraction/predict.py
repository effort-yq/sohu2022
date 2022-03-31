import numpy as np
import time
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
import time
import argparse
import json
import os
from transformers import BertTokenizer
from model import BERT_Classifier
from tqdm import tqdm
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

warnings.filterwarnings("ignore")
setup_seed(44)

from loader import map_id_rel

rel2id, id2rel = map_id_rel()
max_length = 128

USE_CUDA = torch.cuda.is_available()
tokenizer = BertTokenizer.from_pretrained(r'/home/huangyongqing/ali_pytorch/WenTianSearch-main/WenTianSearch-main/chinese-roberta-wwm-ext')

test_data = {}
test_data['mask'] = []
test_data['text'] = []

with open("../nlp_data/test_hand.json", 'r', encoding='utf-8') as load_f:
    data = json.load(load_f)
    for dic in tqdm(data, total=len(data), desc='processing data......'):
        spo_list = dic['spo_list']
        for item in spo_list:
            sent = item[0] + item[-1] + dic['content']
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            test_data['text'].append(indexed_tokens)
            test_data['mask'].append(att_mask)

test_text = test_data['text']
test_mask = test_data['mask']
test_text = [t.numpy() for t in test_text]
test_mask = [t.numpy() for t in test_mask]


test_text = torch.tensor(test_text)
test_mask = torch.tensor(test_mask)
test_dataset = torch.utils.data.TensorDataset(test_text, test_mask)
test_iter = torch.utils.data.DataLoader(test_dataset, 128)


def test(net_path):
    net = torch.load(net_path)
    # net = BERT_Classifier(5)
    net.eval()
    result = []
    if USE_CUDA:
        net = net.cuda()
    with torch.no_grad():
        for text, mask in tqdm(test_iter, total=len(test_iter), desc='Predicting......'):
            text = text.reshape(128, -1).cuda()
            mask = mask.reshape(128, -1).cuda()

            outputs = net(text, attention_mask=mask)
            if len(outputs) == 1:
                logits = outputs[0]  # 保证和旧模型参数的一致性
            else:
                logits = outputs[1]
            _, predicted = torch.max(logits.data, 1)
            result.extend(predicted.cpu().numpy().tolist())

        return result


result = test('0.9468315972222222.pth')

with open('../nlp_data/test.txt', 'r', encoding='utf-8') as f:

    res = []
    for line in f:
        dic = {}
        line = json.loads(line)
        id = line['id']
        entity_list = line['entity']
        for item in entity_list:
            dic[item] = int(result.pop(0)-2)
        res.append([str(id) + '\t' + str(dic)])

with open('section1.txt', 'w', encoding='utf-8') as g:
    g.writelines(['id', '\t', 'result'])
    g.write('\n')
    for line in res:
        for i in line:
            g.write(i)
            g.write('\n')
