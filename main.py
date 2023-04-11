# -*- coding: utf-8 -*-

from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW
import torch
import torch.nn as nn
from torch.utils.data import dataset, DataLoader
from loguru import logger
import warnings
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import os
import random
from collections import OrderedDict
#import gensim
import json
#import pkuseg
import re
import math
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
#from gensim.models import KeyedVectors
from torch.nn import init
from pytorchtools import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
from langconv import *
import pkuseg
from pathlib import Path
from config import *
from preprocess import *
from model import *
from evaluation import *

warnings.filterwarnings('ignore')
device = torch.device("cuda:3" if torch.cuda.is_available() > 0 else "cpu")

def SeedTorch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:2"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

SeedTorch()

class InfoNCE(nn.Module):
    
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='paired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]

class PatsympDataset(dataset.Dataset):
    def __init__(self, patient_symp, label_map, max_len_pt, tokenizer):
        self.patient_symp = patient_symp
        self.label_map = label_map
        self.max_len_pt = max_len_pt
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.patient_symp)
    
    def __getitem__(self, item):
        patient_symp = self.patient_symp[item].strip()
        label_map = self.label_map[item]
        
        patsymp_encoding = self.tokenizer(
        patient_symp,          
        padding="max_length",     
        max_length=self.max_len_pt,  
        add_special_tokens=True,  
        return_token_type_ids=False, 
        return_attention_mask=True,
        return_tensors='pt',   
        truncation=True) 
        
        return {
            'patsymp_text': patient_symp,
            'patsymp_input_ids': patsymp_encoding['input_ids'].flatten(),
            'patsymp_attention_mask': patsymp_encoding['attention_mask'].flatten(),
            'label_map': torch.tensor(label_map, dtype=torch.long)
    }
    
def CreateDataLoader(df, max_len_pt, tokenizer, batch_size):
    ds = PatsympDataset(
        patient_symp = df.content.to_numpy(),
        label_map = df.label_map.to_numpy(),
        max_len_pt = max_len_pt,
        tokenizer = tokenizer
        )
    
    return DataLoader(
    ds,
    batch_size=batch_size
  )

pat_symp_train, pat_symp_val, pat_symp_test, kg_symp, label_data = BuildDataset(config)
train_data_loader = CreateDataLoader(pat_symp_train, config.max_len_pt, config.tokenizer, config.batch_size)
val_data_loader = CreateDataLoader(pat_symp_val, config.max_len_pt, config.tokenizer, config.batch_size)
test_data_loader = CreateDataLoader(pat_symp_test, config.max_len_pt, config.tokenizer, config.batch_size)

assert list(kg_symp.disease_name) == list(label_data.label_name)
knowledge_text = kg_symp.knowledge_text

knowledge_text_input_ids = [config.tokenizer(
        knowledge_text[i], 
        padding="max_length", 
        max_length=config.max_len_kg, 
        add_special_tokens=True, 
        return_token_type_ids=False, 
        return_attention_mask=True, 
        return_tensors='pt', 
        truncation=True 
    )['input_ids'].flatten().numpy() for i in range(len(knowledge_text))]
knowledge_text_attention_mask = [config.tokenizer(
        knowledge_text[i], 
        padding="max_length", 
        max_length=config.max_len_kg, 
        add_special_tokens=True, 
        return_token_type_ids=False, 
        return_attention_mask=True, 
        return_tensors='pt', 
        truncation=True 
    )['attention_mask'].flatten().numpy() for i in range(len(knowledge_text))]

knowledge_text_input_ids = torch.tensor(knowledge_text_input_ids)
knowledge_text_attention_mask = torch.tensor(knowledge_text_attention_mask)

###label preprocess
label_name = label_data.label_name
label_name_input_ids = [config.tokenizer(
        label_name[i], 
        padding="max_length", 
        max_length=config.max_len_label, 
        add_special_tokens=True, 
        return_token_type_ids=False, 
        return_attention_mask=True, 
        return_tensors='pt', 
        truncation=True 
    )['input_ids'].flatten().numpy() for i in range(len(label_name))]
label_name_attention_mask = [config.tokenizer(
        label_name[i], 
        padding="max_length", 
        max_length=config.max_len_label, 
        add_special_tokens=True, 
        return_token_type_ids=False, 
        return_attention_mask=True, 
        return_tensors='pt', 
        truncation=True 
    )['attention_mask'].flatten().numpy() for i in range(len(label_name))]
label_name_input_ids = torch.tensor(label_name_input_ids)
label_name_attention_mask = torch.tensor(label_name_attention_mask)

class DatasetKgLabel(dataset.Dataset):
    def __init__(self, knowledge_text, knowledge_text_input_ids, knowledge_text_attention_mask, label_name_input_ids, label_name_attention_mask, kg_label_map):
        self.knowledge_text = knowledge_text
        self.knowledge_text_input_ids = knowledge_text_input_ids
        self.knowledge_text_attention_mask = knowledge_text_attention_mask
        self.label_name_input_ids = label_name_input_ids
        self.label_name_attention_mask = label_name_attention_mask
        self.label_map_kg = kg_label_map
                        
    def __len__(self):
        return len(self.knowledge_text_input_ids)
    
    def __getitem__(self, item):
        knowledge_text_input_ids_item = torch.tensor(self.knowledge_text_input_ids[item])
        knowledge_text_attention_mask_item = torch.tensor(self.knowledge_text_attention_mask[item])
        label_name_input_ids_item = torch.tensor(self.label_name_input_ids[item])
        label_name_attention_mask_item = torch.tensor(self.label_name_attention_mask[item])
        kg_label_map_item = torch.tensor(self.label_map_kg[item])
        knowledge_text_item = self.knowledge_text[item].strip()

        return {
            'knowledge_text_input_ids': knowledge_text_input_ids_item,
            'knowledge_text_attention_mask': knowledge_text_attention_mask_item,
            'label_name_input_ids': label_name_input_ids_item,
            'label_name_attention_mask': label_name_attention_mask_item,
            'kg_label_map': kg_label_map_item,
            'knowledge_text': knowledge_text_item
    }

def CreateKgLabelLoader(knowledge_text, knowledge_text_input_ids, knowledge_text_attention_mask, label_name_input_ids, label_name_attention_mask, kg_label_map, batch_size):
    ds = DatasetKgLabel(
        knowledge_text = knowledge_text, 
        knowledge_text_input_ids = knowledge_text_input_ids,
        knowledge_text_attention_mask = knowledge_text_attention_mask,
        label_name_input_ids = label_name_input_ids,
        label_name_attention_mask = label_name_attention_mask,
        kg_label_map = kg_label_map
        )
    return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False
  )
knowledge_text_label_name_dataloader = CreateKgLabelLoader(knowledge_text, knowledge_text_input_ids, knowledge_text_attention_mask, label_name_input_ids, label_name_attention_mask, kg_symp.disease_map, config.batch_size)

model_0 = ModelBert(config).to(device)
model_0.load_state_dict(torch.load('./checkpoint.pt'))	

kg_bert_emb = torch.tensor([]).to(device)
label_kg = torch.tensor([]).to(device)
with torch.no_grad():
    for d in tqdm(knowledge_text_label_name_dataloader):
        knowledge_text_input_ids_d = d['knowledge_text_input_ids'].to(device)
        knowledge_text_attention_mask_d = d['knowledge_text_attention_mask'].to(device)
        _, kg_bert_emb_d_hidden_states = model_0(knowledge_text_input_ids_d, knowledge_text_attention_mask_d)
        kg_bert_emb = torch.concat([kg_bert_emb, kg_bert_emb_d_hidden_states], 0)
        label_kg = torch.concat([label_kg, d['kg_label_map'].to(device)], 0)
assert list(label_kg.cpu().detach().numpy()) == list(range(config.n_classes))

train_bert_emb = torch.tensor([]).to(device)
val_bert_emb = torch.tensor([]).to(device)
test_bert_emb =  torch.tensor([]).to(device)
train_label_map = torch.tensor([]).to(device)
val_label_map = torch.tensor([]).to(device)
test_label_map = torch.tensor([]).to(device)
with torch.no_grad():
    for d_train in tqdm(train_data_loader):
        pat_input_ids_pt_train = d_train['patsymp_input_ids'].to(device)
        pat_attention_mask_pt_train = d_train['patsymp_attention_mask'].to(device)
        label_map_train = d_train['label_map'].to(device)
        _, pat_bert_emb_d = model_0(pat_input_ids_pt_train, pat_attention_mask_pt_train)
        train_bert_emb = torch.concat([train_bert_emb, pat_bert_emb_d], 0)
        train_label_map = torch.concat([train_label_map, label_map_train], 0)
    for d_val in tqdm(val_data_loader):
        pat_input_ids_pt_val = d_val['patsymp_input_ids'].to(device)
        pat_attention_mask_pt_val = d_val['patsymp_attention_mask'].to(device)
        label_map_val = d_val['label_map'].to(device)
        _, pat_bert_emb_d = model_0(pat_input_ids_pt_val, pat_attention_mask_pt_val)
        val_bert_emb = torch.concat([val_bert_emb, pat_bert_emb_d], 0)
        val_label_map = torch.concat([val_label_map, label_map_val], 0)
    for d_test in tqdm(test_data_loader):
        pat_input_ids_pt_test = d_test['patsymp_input_ids'].to(device)
        pat_attention_mask_pt_test = d_test['patsymp_attention_mask'].to(device)
        label_map_test = d_test['label_map'].to(device)
        _, pat_bert_emb_d = model_0(pat_input_ids_pt_test, pat_attention_mask_pt_test)
        test_bert_emb = torch.concat([test_bert_emb, pat_bert_emb_d], 0)
        test_label_map = torch.concat([test_label_map, label_map_test], 0)
        
class PatsympDataset2(dataset.Dataset):
    def __init__(self, pat_symp_bert, label_map):
        self.pat_symp_bert = pat_symp_bert
        self.label_map = label_map
        
    def __len__(self):
        return len(self.pat_symp_bert)
    
    def __getitem__(self, item):
        pat_symp_bert_i = self.pat_symp_bert[item]
        label_map = self.label_map[item] 
        
        return {
            'pat_symp_bert': pat_symp_bert_i,
            'label_map': torch.tensor(label_map, dtype=torch.long)
    }
    
def CreateDataLoader2(pat_symp_bert, label_map, batch_size):
    ds = PatsympDataset2(
        pat_symp_bert = pat_symp_bert,
        label_map = label_map
        )
    
    return DataLoader(
    ds,
    batch_size=batch_size
  )
train_data_loader2 = CreateDataLoader2(train_bert_emb, train_label_map, config.batch_size2)
val_data_loader2 = CreateDataLoader2(val_bert_emb, val_label_map, config.batch_size2)
test_data_loader2 = CreateDataLoader2(test_bert_emb, test_label_map, config.batch_size2)

predict_loss = nn.CrossEntropyLoss().to(device)
contrasive_loss = InfoNCE().to(device)
model = ModelV30(config).to(device)
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, correct_bias=False)

total_steps = len(train_data_loader) * config.epoches

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

early_stopping = EarlyStopping(config.patience, verbose=True, path='./checkpoint.pt')

def TrainEpoch(model, train_data_loader2, kg_bert_emb_all, optimizer, device, scheduler):
    model.train()
    losses_classification = []    
    losses_contrasive = []    
    losses = []
    pred_epochi = torch.tensor([]).to(device)
    target_epochi = torch.tensor([], dtype=torch.int64).to(device)
    for d in tqdm(train_data_loader2):
        #print(d)
        #break
        optimizer.zero_grad()

        pat_bert_emb_d = d['pat_symp_bert'].to(device)
        label_map = d['label_map'].to(device)
        kg_bert_emb_d = kg_bert_emb_all.to(device)

        diag_pred, pat_bert_emb, kg_bert_emb = model(pat_bert_emb_d, kg_bert_emb_d)

        negative_label_map = torch.tensor([list(range(config.n_classes))[:i] + list(range(config.n_classes))[i+1:] for i in label_map])
        positive_key = kg_bert_emb[label_map,:]
        negative_keys = kg_bert_emb[negative_label_map,:]

        pred = diag_pred
        loss_classification = predict_loss(diag_pred, label_map)
        loss_contrasive = contrasive_loss(pat_bert_emb, positive_key, negative_keys)
        loss = loss_classification * config.classification_loss_weight + loss_contrasive * config.contrasive_loss_weight

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses_classification.append(loss_classification.item())
        losses_contrasive.append(loss_contrasive.item())
        losses.append(loss.item())

        pred_epochi = torch.cat((pred_epochi, pred), 0)
        target_epochi = torch.cat((target_epochi, label_map), 0)
        
    train_metric = Metrics(pred_epochi, target_epochi)
    return train_metric, np.mean(losses), np.mean(losses_classification), np.mean(losses_contrasive)

def ValEpoch(model, val_data_loader2, kg_bert_emb_all, device):
    val_loss_classification = []
    val_loss_contrasive = []
    val_loss = []
    model.eval()

    pred_epochi = torch.tensor([]).to(device)
    target_epochi = torch.tensor([], dtype=torch.int64).to(device)

    with torch.no_grad():
        for d_val in val_data_loader2:
            pat_bert_emb_d = d_val['pat_symp_bert'].to(device)
            label_map = d_val['label_map'].to(device)
            kg_bert_emb_d =  kg_bert_emb_all.to(device)

            diag_pred, pat_bert_emb, kg_bert_emb = model(pat_bert_emb_d, kg_bert_emb_d)
            
            negative_label_map = torch.tensor([list(range(config.n_classes))[:i] + list(range(config.n_classes))[i+1:] for i in label_map])
            positive_key = kg_bert_emb[label_map,:]
            negative_keys = kg_bert_emb[negative_label_map,:]

            loss_classification = predict_loss(diag_pred, label_map)
            loss_contrasive = contrasive_loss(pat_bert_emb, positive_key, negative_keys)
            loss = loss_classification * config.classification_loss_weight + loss_contrasive * config.contrasive_loss_weight

            val_loss_classification.append(loss_classification.item())
            val_loss_contrasive.append(loss_contrasive.item())
            val_loss.append(loss.item())
                
            pred_epochi = torch.cat((pred_epochi, diag_pred), 0)
            target_epochi = torch.cat((target_epochi, label_map), 0)
    
    val_metric = Metrics(pred_epochi, target_epochi)
    return val_metric, np.mean(val_loss), np.mean(val_loss_classification), np.mean(val_loss_contrasive)

def TestEpoch(model, test_data_loader2, kg_bert_emb_all, device):
    test_loss_classification = []
    test_loss_contrasive = []
    test_loss = []
    model.eval()

    pred_epochi = torch.tensor([]).to(device)
    target_epochi = torch.tensor([], dtype=torch.int64).to(device)

    with torch.no_grad():
        for d_test in test_data_loader2:
            pat_bert_emb_d = d_test['pat_symp_bert'].to(device)
            label_map = d_test['label_map'].to(device)
            kg_bert_emb_d =  kg_bert_emb_all.to(device)

            diag_pred, pat_bert_emb, kg_bert_emb = model(pat_bert_emb_d, kg_bert_emb_d)
            
            negative_label_map = torch.tensor([list(range(config.n_classes))[:i] + list(range(config.n_classes))[i+1:] for i in label_map])
            positive_key = kg_bert_emb[label_map,:]
            negative_keys = kg_bert_emb[negative_label_map,:]

            loss_classification = predict_loss(diag_pred, label_map)
            loss_contrasive = contrasive_loss(pat_bert_emb, positive_key, negative_keys)
            loss = loss_classification * config.classification_loss_weight + loss_contrasive * config.contrasive_loss_weight

            test_loss_classification.append(loss_classification.item())
            test_loss_contrasive.append(loss_contrasive.item())
            test_loss.append(loss.item())
                
            pred_epochi = torch.cat((pred_epochi, diag_pred), 0)
            target_epochi = torch.cat((target_epochi, label_map), 0)
    
    test_metric = Metrics(pred_epochi, target_epochi)
    return test_metric, np.mean(test_loss), np.mean(test_loss_classification), np.mean(test_loss_contrasive)

#Train Phase
history = defaultdict(list)
best_val_f1 = 0.1
train_res_li = []
val_res_li = []
test_res_li = []
for epoch in range(config.epoches):
    print(f'Epoch {epoch + 1}/{config.epoches}')
    print('-' * 10)
    metrics_train, train_loss, train_loss_classification, train_loss_contrasive = TrainEpoch(
        model, train_data_loader2, kg_bert_emb, optimizer, device, scheduler
        )
    print(f'Train loss {train_loss} contrasive loss {train_loss_contrasive} classification loss {train_loss_classification} metrics-acc {metrics_train["acc_top1"]} precision {metrics_train["precision_macro_top1"]} recall {metrics_train["recall_macro_top1"]} F1 {metrics_train["f1_macro_top1"]}')
    
    metrics_val, val_loss, val_loss_classification, val_loss_contrasive = ValEpoch(model, val_data_loader2, 
        kg_bert_emb, device)
    print(f'Val loss {val_loss} contrasive loss {val_loss_contrasive} classification loss {val_loss_classification} metrics-acc {metrics_val["acc_top1"]} precision {metrics_val["precision_macro_top1"]} recall {metrics_val["recall_macro_top1"]} F1 {metrics_val["f1_macro_top1"]}')

    early_stopping(val_loss, model)
    if(early_stopping.early_stop):
        print("Early stopping")
        break

###Test Phase
model = ModelV30(config).to(device)
model.load_state_dict(torch.load('./checkpoint.pt'))	
model.eval()

test_metric, test_loss, test_loss_classification, test_loss_contrasive = TestEpoch(model, test_data_loader2, kg_bert_emb, device)




