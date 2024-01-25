from re import S
import re
import torch
from torch.functional import tensordot
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Normal, kl_divergence

class ScaledAttention(nn.Module):
    def __init__(self, temperature, dropout_rate):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v):
        score = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        output = torch.matmul(score, v)
        return score, output


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, embedding_dim, hid_dim, dropout_rate):
        super().__init__()
        self.num_head = num_head
        self.size_per_head = hid_dim // num_head
        self.hid_dim = hid_dim

        self.q_linear = nn.Linear(embedding_dim, hid_dim)
        self.k_linear = nn.Linear(embedding_dim, hid_dim)
        self.v_linear = nn.Linear(embedding_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = ScaledAttention(temperature = self.size_per_head ** 0.5, dropout_rate = dropout_rate)
    
    def forward(self, q, k, v):
        sample_size = q.size()[0]
        batch_size = q.size()[1]
        q_len, k_len, v_len = q.size()[2], k.size()[2], v.size()[2]

        q = self.q_linear(q).view(sample_size, batch_size, q_len, self.num_head, self.size_per_head)
        k = self.k_linear(k).view(sample_size, batch_size, k_len, self.num_head, self.size_per_head)
        v = self.v_linear(v).view(sample_size, batch_size, v_len, self.num_head, self.size_per_head)

        q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
        score, output = self.attention(q, k, v)

        output = output.transpose(2, 3).contiguous().view(sample_size, batch_size, v_len, self.hid_dim)
        output = self.fc(output)
        output = self.dropout(output)
        return output


class News_Encoder(nn.Module):
    def __init__(self, num_head, hid_dim, word_dim, word_matrix, dropout_rate):
        super().__init__()
        self.word_embedding = nn.Embedding.from_pretrained(word_matrix, freeze = False)
        self.word_attention = MultiHeadAttention(num_head, word_dim, hid_dim, dropout_rate)
        self.hid_dim = hid_dim
        
        self.W1 = nn.Parameter(torch.Tensor(hid_dim, 200))
        self.proj1 = nn.Parameter(torch.Tensor(200, 1))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        nn.init.xavier_uniform_(self.proj1.data, gain=1.414)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, text):
        title, abstract = text[0], text[1]
        title_size = title.size() # [30, 5, 30]
        text = torch.cat([title, abstract], dim = -1)
        text = self.word_embedding(text)
        text = self.word_attention(text, text, text)
        text_att = torch.tanh(torch.matmul(text, self.W1))   # [30, 5, 30, 400]
        text_att = torch.matmul(text_att, self.proj1)
        text_att = F.softmax(text_att, dim = 2)
        text = torch.matmul(text_att.transpose(-2, -1), text).squeeze(dim = 2)    # [30, 5, 400]            
        return text.reshape(title_size[0], title_size[1], -1)



class Basic_User_Encoder(nn.Module):
    def __init__(self, num_head, hid_dim, dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
        self.W = nn.Parameter(torch.Tensor(hid_dim, 200))
        self.proj = nn.Parameter(torch.Tensor(200, 1))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.proj.data, gain=1.414)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, history_rep):    # [30, 50, 400]
        history_rep = self.attention(history_rep.unsqueeze(dim = 0), history_rep.unsqueeze(dim = 0), history_rep.unsqueeze(dim = 0))
        history_rep = history_rep.squeeze(dim = 0)
        att = torch.tanh(torch.matmul(history_rep, self.W))    # [30, 50, 1]
        att = torch.matmul(att, self.proj)
        att = F.softmax(att, dim = 1)

        user_rep = self.dropout(torch.matmul(att.transpose(-2, -1), history_rep).squeeze(dim = 1))
        return user_rep
    

class Predictor(nn.Module):
    def __init__(self, num_prototype, num_head, hid_dim, word_dim, word_matrix, num_layer, dropout_rate):
        super().__init__()
        self.prototype = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
        self.news_encoder = News_Encoder(num_head, hid_dim, word_dim, word_matrix, dropout_rate)
        self.user_encoder = Basic_User_Encoder(num_head, hid_dim, dropout_rate)
        self.hid_dim = hid_dim
        self.W = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        # self.W = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
        self.proj = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.proj.data, gain=1.414)
        nn.init.xavier_uniform_(self.prototype.data, gain=1.414)
        # nn.init.normal_(self.prototype.data, std = 0.02)
        # self.layer_norm1 = nn.LayerNorm(hid_dim)
        # self.layer_norm2 = nn.LayerNorm(hid_dim)
        self.w1 = nn.Parameter(torch.Tensor(hid_dim, 1))
        self.w2 = nn.Parameter(torch.Tensor(hid_dim, 1))
        self.w3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim
        nn.init.xavier_uniform_(self.w1.data, gain=1.414)
        nn.init.xavier_uniform_(self.w2.data, gain=1.414)
        nn.init.xavier_uniform_(self.w3.data, gain=1.414)


    def Discri(self, pos_rep, user_rep):
        return torch.matmul(pos_rep.unsqueeze(dim = 1), user_rep.unsqueeze(dim = 2)).reshape(user_rep.size(0), 1)
        # return torch.matmul(torch.matmul(pos_rep.unsqueeze(dim = 1), self.DW), user_rep.unsqueeze(dim = 2)).reshape(user_rep.size(0), 1)

    def forward(self, pos_candidate, candidate, ori_his, gen_his, training = True):
        device = candidate[0].device

        if training == True:
            pos_candidate_rep = self.news_encoder(pos_candidate).squeeze(dim = 1)
            # pos_candidate_rep = self.layer_norm2(torch.matmul(pos_candidate_rep, self.W))

        candidate_rep = self.news_encoder(candidate)  #[30, 5] -> [30, 5, 400]
        # candidate_rep = self.layer_norm2(torch.matmul(candidate_rep, self.W))

        gen_batch, gen_length = gen_his[0].size(0), gen_his[0].size(1)
        gen_rep = self.news_encoder(gen_his)  # [30, 5] -> [30, 5, 400]
        # gen_rep = self.layer_norm2(torch.matmul(gen_rep, self.W))
        gen_rep = gen_rep.reshape(gen_batch * gen_length, self.hid_dim)

        att = torch.matmul(self.prototype.unsqueeze(dim = 1).repeat(1, gen_rep.size(0), 1), self.w1) + torch.matmul(gen_rep.unsqueeze(dim = 0).repeat(self.num_prototype, 1, 1), self.w2)
        att = F.softmax(att, dim = 1)
        self.prototype.data = F.relu(self.prototype + torch.matmul(torch.matmul(att.transpose(-1, -2), gen_rep).squeeze(dim = 1), self.w3))
        
        weight = torch.matmul(self.prototype, gen_rep.transpose(-1, -2))
        values, indices = torch.max(weight, dim = 0)
        gen_rep = self.prototype[indices].reshape(gen_batch, gen_length, self.hid_dim)

        # 先gate, 后cat
        ori_rep = self.news_encoder(ori_his)
        # ori_rep = self.layer_norm2(torch.matmul(ori_rep, self.W))
        gate_h = torch.sigmoid(torch.matmul(ori_rep, self.proj))
        his_relevant = torch.mul(gate_h, ori_rep)
        # his_global = self.layer_norm1(ori_rep)
        his_global = torch.mul(1 - gate_h, ori_rep)
        
        his_relevant = torch.cat([his_relevant, gen_rep], dim = 1)
        his_global = torch.cat([his_global, gen_rep], dim = 1)
        user_rep = self.user_encoder(his_relevant)
        irre_rep = self.user_encoder(his_global)

        logits = torch.matmul(candidate_rep, user_rep.unsqueeze(dim = 2)).squeeze(dim = 2)    # [30, 5, 1]
        
        # InfoNCE (BCEwithlogits)
        if training == True:
            pos_val = self.Discri(pos_candidate_rep, user_rep)
            neg_val = self.Discri(pos_candidate_rep, irre_rep)
            info_logits = torch.cat([pos_val, neg_val], dim = 1)
        else:
            info_logits = None
        return logits, info_logits