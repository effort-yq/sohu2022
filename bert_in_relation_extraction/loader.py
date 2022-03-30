import json
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(44)


def prepare_data():
    print("---Regenerate Data---")
    with open("train_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {}
                single_data['rel'] = j["predicate"]
                single_data['ent1'] = j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text'] = dic['text']
                info.append(single_data)
        sub_train = info
    with open("train.json", "w", encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")

    with open("dev_data.json", 'r', encoding='utf-8') as load_f:
        info = []
        import random
        for line in load_f.readlines():
            dic = json.loads(line)
            for j in dic['spo_list']:
                single_data = {}
                single_data['rel'] = j["predicate"]
                single_data['ent1'] = j["object"]
                single_data['ent2'] = j["subject"]
                single_data['text'] = dic['text']
                info.append(single_data)

        sub_train = info
    with open("dev.json", "w", encoding='utf-8') as dump_f:
        for i in sub_train:
            a = json.dumps(i, ensure_ascii=False)
            dump_f.write(a)
            dump_f.write("\n")


# prepare_data()


# def map_id_rel():
#     rel = ["UNK"]
#     with open("train.json", 'r', encoding='utf-8') as load_f:
#         for line in load_f.readlines():
#             dic = json.loads(line)
#             if dic['rel'] not in rel:
#                 rel.append(dic['rel'])
#     id2rel={}
#     rel2id={}
#     for i in range(len(rel)):
#         id2rel[i]=rel[i]
#         rel2id[rel[i]]=i
#     return rel2id,id2rel

def map_id_rel():
    id2rel = {'-2': '极负向', '-1': '负向', '0': '中立', '1': '正向', '2': '极正向'}
    rel2id = {}
    for i in id2rel:
        rel2id[id2rel[i]] = i
    return rel2id, id2rel

pretrained_path = r'D:\tianchi_match\bert_chinese'
def load_train():
    rel2id, id2rel = map_id_rel()
    max_length = 128
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("../nlp_data/train_hand.json", 'r', encoding='utf-8') as load_f:
        temp = json.load(load_f)
        temp = temp[:200]
        for dic in temp:
            # dic = json.loads(line)
            spo_list = dic['spo_list']
            for item in spo_list:
                if item[-1] not in rel2id:
                    # todo
                    train_data['label'].append(00)
                else:
                    train_data['label'].append(int(rel2id[item[-1]])+2)
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
                train_data['text'].append(indexed_tokens)
                train_data['mask'].append(att_mask)

    return train_data


def load_dev():
    rel2id, id2rel = map_id_rel()
    max_length = 128
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    train_data = {}
    train_data['label'] = []
    train_data['mask'] = []
    train_data['text'] = []

    with open("../nlp_data/dev_hand.json", 'r', encoding='utf-8') as load_f:
        data = json.load(load_f)
        for dic in data:
            # dic = json.loads(line)
            spo_list = dic['spo_list']
            for item in spo_list:
                if item[-1] not in rel2id:
                    # todo
                    train_data['label'].append(00)
                else:
                    train_data['label'].append(int(rel2id[item[-1]])+2)
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
                train_data['text'].append(indexed_tokens)
                train_data['mask'].append(att_mask)

    return train_data