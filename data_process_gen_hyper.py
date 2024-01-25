from re import S
from torch._C import dtype
from nltk.tokenize import word_tokenize
import numpy as np
import random
import torch
import json
import pickle
from scipy.sparse import coo_matrix, find, save_npz, load_npz, csr_matrix

#import nltk
#nltk.download('punkt')

class DataProcess():
    def __init__(self, file1, file2, file3, file4, file5):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.file5 = file5

        self.news_id = {'NULL': 0}  # {'NULL': 0, 'N46466': 1}
        self.gen_news_id = {'NULL': 0}  # {'NULL': 0, 'N46466': 1}
        self.title_content = {}  # {'N46466': ['the', 'brands', 'queen', 'elizabeth', ',', 'prince', 'charles', ',', 'and', 'prince', 'philip', 'swear', 'by']}
        self.abstract_content = {}
        self.word_dict = {'PADDING': 0}
        self.news_title_dict = {'0': [0] * 20, }  # {'1': [1, 2, ……, 30]}
        self.news_abstract_dict = {'0': [0] * 40, }
        self.newsid_topic = {0: 'NULL'}
        self.embedding_dict = {}

        self.user_id = {'NULL': 0, }
        self.id_user = {0: 'NULL', }
        self.npratio1 = 4
        self.npratio2 = 50
        self.user_his = {0: [0] * 50, }
        self.gen_user_his = {0: [0] * 5, }
        self.his_pad = {0: [1] * 50, }
        self.user_his_downsample = {0: [0] * 20, }
        self.user_his_complete = {0: set([0]), }
        
        self.user_news_index = set()
        
        self.warm_user = set()
        self.cold_user = set()
        self.warm_complete = {}
        self.warm_downsample = {}
        self.cold_complete = {}
        
        self.num_neighbor = 10
        self.user_2hop_news = {}
        self.user_correlate_neighbor = {0: [0] * self.num_neighbor, }
        self.user_random_neighbor = {0: [0] * self.num_neighbor, }
        self.user_popular_neighbor = {}

        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.train_pos_candidate = []

        self.val_index = []
        self.val_candidate = []
        self.val_pos_candidate = []
        self.val_label = []
        self.val_user_his = []
        self.val_user = []
        
        self.news_degree = {0: 0, }
    
    def newsample(self, array, npratio):
        if npratio > len(array):
            return random.sample(array*(npratio // len(array) + 1), npratio)
        else:
            return random.sample(array, npratio)


    # 处理新闻数据

    def process_news(self, file):
        f = open(file, 'r', encoding = 'utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[0] not in self.news_id:
                self.news_id[line[0]] = len(self.news_id)


    def process_train_val_news(self):
        print ('process news start')
        self.process_news(self.file1)
        self.process_news(self.file2)

        # f = open('prompt_v1/small_news_gen.tsv', 'r')
        # lines = f.readlines()
        # for line in lines:
        #     line = line.strip().split('\t')
        #     if line[0] not in self.news_id:
        #         self.news_id[line[0]] = len(self.news_id)
        print ('process news finished')
    

    def generate_user_his(self):    
        f3 = open(self.file3, 'r', encoding = 'utf-8')
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue
            if line[1] not in self.user_id:
                self.user_id[line[1]] = len(self.user_id)
                self.id_user[self.user_id[line[1]]] = line[1]

                his_complete = set()
                for index in line[3].split():
                    his_complete.add(self.news_id[index])
                self.user_his[self.user_id[line[1]]] = his_complete
        f3.close()

        f4 = open(self.file4, 'r', encoding = 'utf-8')
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue
            if line[1] not in self.user_id:
                self.user_id[line[1]] = len(self.user_id)
                self.id_user[self.user_id[line[1]]] = line[1]

                his_complete = set()
                for index in line[3].split():
                    his_complete.add(self.news_id[index])
                self.user_his[self.user_id[line[1]]] = his_complete
        f4.close()

        f = open('../Dataset/prompt_v2/small_user_news_gen.txt', 'r')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            for index in line[1].split():
                if index not in self.gen_news_id:
                    self.gen_news_id[index] = len(self.gen_news_id)
                if self.user_id[line[0]] not in self.gen_user_his:
                    self.gen_user_his[self.user_id[line[0]]] = set()
                self.gen_user_his[self.user_id[line[0]]].add(self.gen_news_id[index])

        for u, his in self.user_his.items():
            his_pad = list(his)[:45]
            self.his_pad[u] = [1] * len(his_pad) + [0] * (45 - len(his_pad))
            his_pad = his_pad + [0] * (45 - len(his_pad))
            self.user_his[u] = his_pad

            if u not in self.gen_user_his:
                self.gen_user_his[u] = [0] * 5
            else:
                self.gen_user_his[u] = list(self.gen_user_his[u])[:5] + [0] * (5 - len(self.gen_user_his[u]))
        
        for k,v in self.gen_user_his.items():
            if len(v) != 5:
                print (k, v)


    # 处理训练集数据
    def pre_train_behaviors(self):
        print ('reset train variables')
        self.train_pos_candidate = []
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.train_user = []
        self.train_mask = []
        
        print ('process train behaviors start')

        f3 = open(self.file3, 'r', encoding = 'utf-8')
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue
                
            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            for doc in p_doc:
                neg_doc = self.newsample(n_doc, self.npratio1)
                neg_doc.append(doc)
                candidate_label = [0] * self.npratio1 + [1]
                candidate_order = list(range(self.npratio1 + 1))
                random.shuffle(candidate_order)
                candidate_shuffle = []
                candidate_label_shuffle = []
                for i in candidate_order:
                    candidate_shuffle.append(neg_doc[i])
                    candidate_label_shuffle.append(candidate_label[i])
                self.train_pos_candidate.append(doc)
                self.train_candidate.append(candidate_shuffle)
                self.train_label.append(candidate_label_shuffle)
                self.train_user.append(self.user_id[line[1]])
                if self.user_id[line[1]] in self.warm_user:
                    self.train_mask.append(1)
                else:
                    self.train_mask.append(0)
        self.train_pos_candidate = torch.LongTensor(np.array(self.train_pos_candidate, dtype = 'int32'))
        self.train_candidate = torch.LongTensor(np.array(self.train_candidate, dtype='int32'))
        self.train_label = torch.FloatTensor(np.array(self.train_label, dtype='int32'))
        self.train_user = torch.LongTensor(np.array(self.train_user, dtype = 'int32'))
        self.train_mask = torch.LongTensor(np.array(self.train_mask, dtype = 'int32'))

        print ('train_pos_candidate.size: ', self.train_pos_candidate.size())
        print ('train_candidate.size: ', self.train_candidate.size())
        print ('train_label.size:', self.train_label.size())
        print ('train_user.size:', self.train_user.size())
        print ('train_mask.size:', self.train_mask.size())
        print ('process train behaviors finished')
        return [self.train_pos_candidate, self.train_candidate, self.train_user, self.train_mask, self.train_label]


    # 处理验证集数据
    def pre_val_behaviors(self, file):
        print('process val behaviors start')
        self.val_index = []
        self.val_candidate = []
        self.val_pos_candidate = []
        self.val_label = []
        self.val_user_his = []
        self.val_user = []
        self.val_mask = []

        f = open(file, 'r', encoding = 'utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            sess_index = []
            sess_index.append(len(self.val_candidate))
            for i in p_doc:
                self.val_candidate.append(i)
                self.val_label.append(1)
                self.val_user.append(self.user_id[line[1]])
                if self.user_id[line[1]] in self.warm_user:
                    self.val_mask.append(1)
                else:
                    self.val_mask.append(0)
            
            for i in n_doc:
                self.val_candidate.append(i)
                self.val_label.append(0)
                self.val_user.append(self.user_id[line[1]])
                if self.user_id[line[1]] in self.warm_user:
                    self.val_mask.append(1)
                else:
                    self.val_mask.append(0)

            sess_index.append(len(self.val_candidate))
            self.val_index.append(sess_index)

        self.val_candidate = np.array(self.val_candidate, dtype='int32')
        self.val_label = torch.FloatTensor(np.array(self.val_label, dtype='int32'))
        self.val_user = torch.LongTensor(np.array(self.val_user, dtype = 'int32'))
        self.val_mask = torch.LongTensor(np.array(self.val_mask, dtype = 'int32'))

        print('val_candidate.shape: ', self.val_candidate.shape)
        print('val_label.size: ', self.val_label.size())
        print ('val_user.size:', self.val_user.size())
        print('len(val_index): ', len(self.val_index))

        print('process val behaviors finished')
        return [self.val_candidate, self.val_user, self.val_mask, self.val_label, self.val_index]