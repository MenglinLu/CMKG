# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import numpy as np

class Config(object):
    """配置参数"""
    def __init__(self, dir_path):
        self.model_name = 'ourmodel'
        self.train_path = dir_path + '/data/train_data_v3.csv'                      
        self.val_path = dir_path + '/data/val_data_v3.csv'   
        self.test_path = dir_path + '/data/test_data_v3.csv'  
        self.label_path = dir_path + '/data/labeldict_v4.npy'  
        self.kg_path = dir_path + '/data/diagnosis_unique_completed_v4.xlsx' 
        self.kg_path_v2 = dir_path + '/data/diagnosis_unique_completed_v4.xlsx'
        self.n_classes = 133  
        self.max_len_pt = 512         
        self.max_len_kg = 512  
        self.max_len_label = 20
        self.learning_rate = 1e-5  
        self.bert_path = 'bert-base-chinese'
        self.bert_hidden_size = 768
        self.batch_size = 8
        self.epoches = 500
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.label_dict = np.load(self.label_path, allow_pickle=True).item()
        self.class_name = [self.label_dict[x] for x in self.label_dict.keys()]
        self.dropout = 0.2
        self.patience = 15
        self.classification_loss_weight = 1
        self.contrasive_loss_weight = 1
        self.fc_hidden1 = 768
        self.batch_size2 = 128
        label_dict = np.load(self.label_path, allow_pickle=True).tolist()
        self.target_names = []
        for key, val in label_dict.items():
            self.target_names.append(val)

config = Config('..')