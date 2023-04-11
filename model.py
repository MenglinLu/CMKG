# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init

class ModelBert(nn.Module):
    def __init__(self, config):
        super(ModelBert, self).__init__()
        self.bert_pat = AutoModel.from_pretrained(config.bert_path)
                
        self.diag_predict = nn.Sequential(
            nn.Linear(self.bert_pat.config.hidden_size, config.n_classes)
            )
           
    def forward(self, input_ids_pt, attention_mask_pt):
        output_pt = self.bert_pat(
        input_ids = input_ids_pt,
        attention_mask = attention_mask_pt, output_hidden_states=True)
        
        hidden_states = output_pt.hidden_states
        hidden_states = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
        
        diag_pred = self.diag_predict(hidden_states)
        
        return diag_pred, hidden_states

class ModelV30(nn.Module):
    def __init__(self, config):
        super(ModelV30, self).__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.bert_hidden_size, config.fc_hidden1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(config.fc_hidden1, config.n_classes)            
        
    def forward(self, pat_bert_emb, kg_bert_emb):
        hidden_state_pat = self.dropout(pat_bert_emb)
        hidden_state_pat = self.fc1(hidden_state_pat)
        hidden_state_pat = self.relu(hidden_state_pat)
        pred = self.fc2(hidden_state_pat)

        hidden_state_kg = self.dropout(kg_bert_emb)
        hidden_state_kg = self.fc1(hidden_state_kg)
        hidden_state_kg = self.relu(hidden_state_kg)

        return pred, hidden_state_pat, hidden_state_kg