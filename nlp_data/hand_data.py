import json
import random
# file = './train.txt'
# with open(file, 'r', encoding='utf-8') as f:
#     for line in f:
#         # print(line)
#         line = json.loads(line)
#         tt = line['entity']
#         for i in tt:
#             print(i)
#         break
#     # data = json.loads(data)

# 89195条训练集，可拆成训练和测试集
id2text = {'-2': '极负向', '-1': '负向', '0': '中立', '1': '正向', '2': '极正向'}

def handle_data(path, type='train'):
    if type == 'train':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dic = {}
                line = json.loads(line)
                dic['id'] = line['id']
                dic['content'] = line['content']
                dic['spo_list'] = []
                for item in line['entity']:
                    dic['spo_list'].append([item, '情感', id2text.get(str(line['entity'][item]))])
                data.append(dic)
        random.shuffle(data)
        with open('./train_hand.json', 'w', encoding='utf-8') as f:
            json.dump(data[:-2000], f, indent=4, ensure_ascii=False)
        with open('./dev_hand.json', 'w', encoding='utf-8') as f:
            json.dump(data[-2000:], f, indent=4, ensure_ascii=False)
    elif type == 'test':
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dic = {}
                line = json.loads(line)
                dic['id'] = line['id']
                dic['content'] = line['content']
                dic['spo_list'] = []
                for item in line['entity']:
                    dic['spo_list'].append([item, '情感', ''])
                data.append(dic)
        with open('./test_hand.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

handle_data('./train.txt', 'train')
handle_data('./test.txt', 'test')