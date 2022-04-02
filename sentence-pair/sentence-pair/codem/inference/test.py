import sys
import os
sys.path.append('../../')

import numpy as np

import  codem.utils  as ut
import pandas as pd
import transformers
transformers.logging.set_verbosity_error()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import json


def output_data(test_data):
    pp_dict = {0: '极负向', 1: '负向', 2: '中立', 3: '正向', 4: '极正向'}
    # test_data['label'] = test_data['label'].apply(lambda x: pp_dict[x])
    result = []
    for idx, sub_data in test_data.groupby(by='text_id', sort=False):
        sub_dic = {}
        sub_dic['text_id'] = list(sub_data['text_id'])[0]
        sub_dic['content'] = list(sub_data['content'])[0]
        sub_dic['label'] = list(sub_data['label'])[0]
        result.extend(list(sub_data['label']))

    with open(r'/home/huangyongqing/ali_pytorch/WenTianSearch-main/WenTianSearch-main/Sohu2022_data/Sohu2022_data/nlp_data/test.txt', 'r', encoding='utf-8') as f:
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



def get_model(test_name,test_data,weighted,sd):

    opr = ut.OptimizedRounder()

    pre_label = []
    pp_dict = {'极负向': 0, '负向': 1, '中立': 2, '正向': 3, '极正向': 4}

    for i in range(1):
        if 'bert' in test_name:
            modeltype = 'bert'
        else:
            modeltype = 'nezha'

        pre = ut.simmodels(test_name + f"/{i}",
                           './temp',
                     model_type=modeltype, ).predict(test_data)



        dev_data = pd.read_csv(f'../utils/kfold/data_KFold_{sd}/data{i}/dev.csv')
        dev_data['label'] = dev_data['label'].apply(lambda x: pp_dict[x])

        pre_dev = ut.simmodels(test_name + f"/{i}",
                     './temp',
                     model_type=modeltype).predict(dev_data)


        opr.fit(X=pre_dev, y=list(dev_data['label']))

        pre_label.append(weighted*pre)




    coef = opr.coef_['x']
    sub = opr.predict(np.array(pre_label), coef)

    return sub

def main():

    test_path='/home/huangyongqing/ali_pytorch/WenTianSearch-main/WenTianSearch-main/Sohu2022_data/Sohu2022_data/nlp_data/test_hand.json'
    test_data = ut.get_data(test_path, 'test')


    tmp = []
    for i, weighted, ss in test_model_list:

        tmp.extend(get_model(i, test_data, weighted, ss))


    sub_all=np.argmax(np.mean(tmp,axis=0),axis=-1)




    test_data['label']=sub_all

    output_data(test_data)


if __name__=='__main__':


    test_model_name1 = '/home/huangyongqing/ali_pytorch/WenTianSearch-main/WenTianSearch-main/Sohu2022_data/Sohu2022_data/ccks2021-track3-top1-main/ccks2021-track3-top1-main/codem/train/model_param/saved_model'
    # test_model_name2 = 'user_data/model_param/saved_model/nezha_wwm/'
    # test_model_name3 = 'user_data/model_param/saved_model/mac_bert/'

    test_model_list = [(test_model_name1, 1, 42)]

    # test_model_list = [(test_model_name1,0.45,42),
    #                    (test_model_name2,0.35,24),
    #                    (test_model_name3,0.2,33),
    #                   ]





    main()


