import os
import sys
sys.path.append('../../')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from codem.utils.tools import get_data

from codem.configm.config import args

def generate_data(train_df,random_state=42):

    X = np.array(train_df.index)

    y = list(train_df['label'])
    train_list = []
    dev_list = []


    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    for train_index, dev_index in skf.split(X, y):


        tmp_train_df = train_df.iloc[train_index]

        tmp_dev_df = train_df.iloc[dev_index]


        train_list.append(tmp_train_df)
        dev_list.append(tmp_dev_df)
    return train_list, dev_list



if __name__=='__main__':


    train_path1 = r'/home/huangyongqing/ali_pytorch/WenTianSearch-main/WenTianSearch-main/Sohu2022_data/Sohu2022_data/nlp_data/train_hand.json'


    train_data = get_data(train_path1, 'train')


    train_data = train_data.drop_duplicates(['content', 'entity'])
    train_data.index = range(len(train_data))
    train_list, dev_list = generate_data(train_data, random_state=args.seed)



    def kf3(traindt,seed=42):
        train_list, dev_list=generate_data(traindt, random_state=seed)
        for idx, (q, p) in enumerate(zip(train_list, dev_list)):
            if not os.path.exists(f'./kfold/data_KFold_{seed}/data{idx}/'):
                os.makedirs(f'./kfold/data_KFold_{seed}/data{idx}/')
            q.to_csv(f'./kfold/data_KFold_{seed}/data{idx}/train.csv', index=False)
            p.to_csv(f'./kfold/data_KFold_{seed}/data{idx}/dev.csv', index=False)


    kf3(train_data, 42)
    kf3(train_data, 24)
    kf3(train_data, 33)
    print('...............kf finish...........')