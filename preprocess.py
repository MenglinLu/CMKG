# -*- coding: utf-8 -*-

from langconv import *
import re
import pandas as pd
import numpy as np
from config import *

def TransStr(sentence):
    change_sentence=""
    for word in sentence:
        inside_code=ord(word)
        if inside_code==12288: 
            inside_code=32
        elif inside_code>=65281 and inside_code<=65374: 
            inside_code-=65248
        change_sentence+=chr(inside_code)
    change_sentence = change_sentence.lower().strip()
    change_sentence = Converter('zh-hans').convert(change_sentence)
    string_i = re.sub(r'[^\u4e00-\u9fa5^\u0041-\u005a^\u0030-\u0039^\u3002^\uFF01^\uFF01^\uFF0C^\u002C]', '', change_sentence)
    string_i = re.sub(r'[0-9a-zA-Z./\u007E\u002A]+', ' 1 ', string_i)
    
    return ''.join(string_i.split())

def BuildDataset(config):
    pat_symp_train = pd.read_csv(config.train_path)
    pat_symp_val = pd.read_csv(config.val_path)
    pat_symp_test = pd.read_csv(config.test_path)
    pat_symp_train['content'] = [TransStr(i) for i in pat_symp_train.content]
    pat_symp_val['content'] = [TransStr(i) for i in pat_symp_val.content]
    pat_symp_test['content'] = [TransStr(i) for i in pat_symp_test.content]

    label_dict = np.load(config.label_path, allow_pickle=True).item()
    label_dict_reverse = {label_dict[i]:int(i) for i in label_dict.keys()}

    pat_symp_train['label_map'] = pat_symp_train['new_label'].map(label_dict_reverse)
    pat_symp_val['label_map'] = pat_symp_val['new_label'].map(label_dict_reverse)
    pat_symp_test['label_map'] = pat_symp_test['new_label'].map(label_dict_reverse)

    pat_symp_train['label_map_onehot'] = np.zeros([len(pat_symp_train), config.n_classes],dtype=np.int64).tolist()
    for i in range(len(pat_symp_train)):
        pat_symp_train['label_map_onehot'].iloc[i][int(pat_symp_train['label_map'].iloc[i])] = 1
    pat_symp_train['label_map_onehot'] = [np.array(i) for i in pat_symp_train['label_map_onehot']]

    pat_symp_val['label_map_onehot'] = np.zeros([len(pat_symp_val), config.n_classes],dtype=np.int64).tolist()
    for i in range(len(pat_symp_val)):
        pat_symp_val['label_map_onehot'].iloc[i][int(pat_symp_val['label_map'].iloc[i])] = 1
    pat_symp_val['label_map_onehot'] = [np.array(i) for i in pat_symp_val['label_map_onehot']]
    
    pat_symp_test['label_map_onehot'] = np.zeros([len(pat_symp_test), config.n_classes],dtype=np.int64).tolist()
    for i in range(len(pat_symp_test)):
        pat_symp_test['label_map_onehot'].iloc[i][int(pat_symp_test['label_map'].iloc[i])] = 1
    pat_symp_test['label_map_onehot'] = [np.array(i) for i in pat_symp_test['label_map_onehot']]

    kg_symp = pd.read_excel(config.kg_path_v2)
    kg_symp['disease_name'] = [i.strip() for i in kg_symp['disease_name']]
    kg_symp['disease_map'] = kg_symp['disease_name'].map(label_dict_reverse)
    kg_symp['knowledge_text'] = [TransStr(i) for i in kg_symp['knowledge_describe']]
    kg_symp = kg_symp.sort_values(by="disease_map" , inplace=False, ascending=True).reset_index().iloc[:,1:]
    kg_symp['kg_map_onehot'] = np.zeros([len(kg_symp), config.n_classes],dtype=np.int64).tolist()
    for i in range(len(kg_symp)):
        kg_symp['kg_map_onehot'].iloc[i][int(kg_symp['disease_map'].iloc[i])] = 1
    kg_symp['kg_map_onehot'] = [np.array(i) for i in kg_symp['kg_map_onehot']]

    label_data = pd.DataFrame(list(label_dict.items()), columns=['label_id', 'label_name'])

    assert list(kg_symp['disease_name']) == list(label_data['label_name'])

    return pat_symp_train, pat_symp_val, pat_symp_test, kg_symp, label_data
