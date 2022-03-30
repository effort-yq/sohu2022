from transformers import BertModel
import torch.nn as nn
pretrained_path = r'D:\tianchi_match\bert_chinese'
class BERT_Classifier(nn.Module):
    def __init__(self, label_num):
        super().__init__()
        self.encoder = BertModel.from_pretrained(pretrained_path)
        self.dropout = nn.Dropout(0.1,inplace=False)
        self.fc = nn.Linear(768, label_num)
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, attention_mask ,label=None):
        x = self.encoder(x, attention_mask=attention_mask)[0]
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.fc(x)
        if label == None:
            return None, x
        else:
            return self.criterion(x, label), x