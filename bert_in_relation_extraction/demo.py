import numpy as np
from sklearn.preprocessing import MinMaxScaler
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
import json
# from transformers import BertPreTrainedModel
import random
from model import BERT_Classifier

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


warnings.filterwarnings("ignore")
setup_seed(44)

from transformers import BertModel

from loader import map_id_rel

rel2id, id2rel = map_id_rel()

print(len(rel2id))
print(id2rel)

USE_CUDA = torch.cuda.is_available()


def test(net_path, text_list, ent1_list, ent2_list, result, show_result=False):
    max_length = 128
    net = BERT_Classifier(5)


    net.eval()
    if USE_CUDA:
        net = net.cuda()
    pred_ent2_list = []
    correct = 0
    total = 0
    with torch.no_grad():
        for text, ent1, ent2, label in zip(text_list, ent1_list, ent2_list, result):
            sent = ent1 + ent2 + text
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            indexed_tokens = tokenizer.encode(sent, add_special_tokens=True)
            avai_len = len(indexed_tokens)
            while len(indexed_tokens) < max_length:
                indexed_tokens.append(0)  # 0 is id for [PAD]
            indexed_tokens = indexed_tokens[: max_length]
            indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)

            # Attention mask
            att_mask = torch.zeros(indexed_tokens.size()).long()  # (1, L)
            att_mask[0, :avai_len] = 1
            if USE_CUDA:
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()

            if USE_CUDA:
                indexed_tokens = indexed_tokens.cuda()
                att_mask = att_mask.cuda()
            outputs = net(indexed_tokens, attention_mask=att_mask)
            # print(y)
            if len(outputs) == 1:
                logits = outputs[0]  # 保证和旧模型参数的一致性
            else:
                logits = outputs[1]
            _, predicted = torch.max(logits.data, 1)
            result = predicted.cpu().numpy().tolist()[0]
            if show_result:
                print("Source Text: ", text)
                print("Entity1: ", ent1, " Predict Entity2: ", id2rel[str(result-2)], " True Relation: ",
                      label)
            # if id2rel[result] == label:
            #     correct += 1
            total += 1
            # print('\n')
            pred_ent2_list.append(id2rel[str(result-2)])
    print(correct, " ", total, " ", correct / total)
    return pred_ent2_list




def demo_output():
    text_list = []
    ent1 = []
    ent2 = []   # 要预测
    result = []
    total_num = 5
    with open("../nlp_data/test_hand.json", 'r', encoding='utf-8') as load_f:
        lines = json.load(load_f)
        for line in lines:
            spo_list = line['spo_list']
            for item in spo_list:
                text_list.append(line['content'])
                ent1.append(item[0])
                result.append(item[1])
                ent2.append('')
    test('./bert-base-chinese/test.pth', text_list, ent1, ent2, result, True)





demo_output()

