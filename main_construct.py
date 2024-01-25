from data_process import DataProcess
from model_construct import Predictor
from evaluate import scoring

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import time
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import argparse
import os
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--num_dataset', type=int, default=1)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--num_prototype', type=int, default=1000)
    parser.add_argument('--num_hyperedge', type=int, default=1000)
    parser.add_argument('--num_aspect', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--hid_dim', type=int, default=400)
    parser.add_argument('--num_head', type=int, default=20)
    parser.add_argument('--alpha', type=int, default=1)
    parser.add_argument('--num_negative_sample', type=int, default=3)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--preserve_dir', type=str)
    parser.add_argument('--pretrain_method', type=str, default='glove')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    args = parser.parse_args()

    num_epoch = args.num_epoch
    num_dataset = args.num_dataset
    num_layer = args.num_layer
    num_hyperedge = args.num_hyperedge
    num_aspect = args.num_aspect
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    hid_dim = args.hid_dim
    num_head = args.num_head
    word_dim = args.word_dim
    num_negative_sample = args.num_negative_sample
    preserve_dir = args.preserve_dir
    pretrain_method = args.pretrain_method
    num_prototype = args.num_prototype
    alpha = args.alpha
    dropout_rate = args.dropout_rate

    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)

    file1 = '../Dataset/MINDsmall_train/news.tsv'
    file2 = '../Dataset/MINDsmall_dev/news.tsv'
    file3 = '../Dataset/MINDsmall_train/behaviors.tsv'
    file4 = '../Dataset/MINDsmall_dev/behaviors.tsv'

    # file1 = '../Dataset/MINDlarge_train/news.tsv'
    # file2 = '../Dataset/MINDlarge_dev/news.tsv'
    # file3 = '../Dataset/MINDlarge_train/behaviors.tsv'
    # file4 = '../Dataset/MINDlarge_dev/behaviors.tsv'

    file5 = '../Dataset/glove.840B.300d.txt'

    data = DataProcess(file1, file2, file3, file4, file5)
    news_title, news_abs = data.process_train_val_news()
    news_title, news_abs = torch.LongTensor(news_title), torch.LongTensor(news_abs)

    word_matrix = None
    if pretrain_method == 'glove':
        word_matrix = data.load_glove()

    ori_his, gen_his = data.generate_user_his()
    ori_his = torch.LongTensor(np.array(list(ori_his.values()), dtype = 'int32'))
    gen_his = torch.LongTensor(np.array(list(gen_his.values()), dtype = 'int32'))
    print (ori_his.size(), gen_his.size())

    model = Predictor(num_prototype, num_head, hid_dim, word_dim, word_matrix, num_layer, dropout_rate)
    device_ids = [0]
    model = nn.DataParallel(model, device_ids = device_ids)
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.001)

    best_epoch = 0
    min_loss = float('inf')
    
    for n_d in range(num_dataset):
        [train_pos_candidate, train_candidate, train_user, train_mask, train_label] = data.pre_train_behaviors()
        train_dataset = Data.TensorDataset(train_pos_candidate, train_candidate, train_user, train_label)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

        for n_ep in range(num_epoch):
            acc, all = 0, 0
            t0 = time.time()
            loss_per_epoch = []
            

            for step, (pos, candidate, user, label) in enumerate(train_loader):
                t1 = time.time()
                pos_title, pos_abs = Variable(news_title[pos].unsqueeze(dim = 1).cuda()), Variable(news_abs[pos].unsqueeze(dim = 1).cuda())
                candidate_title, candidate_abs = Variable(news_title[candidate].cuda()), Variable(news_abs[candidate].cuda())
                ori_his_title, ori_his_abs = Variable(news_title[ori_his[user]].cuda()), Variable(news_abs[ori_his[user]].cuda())
                gen_his_title, gen_his_abs = Variable(news_title[gen_his[user]].cuda()), Variable(news_abs[gen_his[user]].cuda())
                label = Variable(label.cuda())
                pos, candidate, ori, gen = (pos_title, pos_abs), (candidate_title, candidate_abs), (ori_his_title, ori_his_abs), (gen_his_title, gen_his_abs)

                model.train()
                optimizer.zero_grad()

                # InfoNCE
                logits, aug_logits = model(pos, candidate, ori, gen, training = True)
                aug_label = torch.FloatTensor(np.array([1, 0], dtype = 'int32')).repeat(candidate_title.size(0), 1).cuda()
                aug_label = aug_label.to(logits.device)
                aug_loss = criterion(aug_logits, aug_label)

                bce_loss = criterion(logits, label)
                loss = bce_loss + 0.1 * aug_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1, norm_type = 2)

                optimizer.step()

                loss_per_epoch.append(loss.data.item())
                print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'step: {:04d}'.format(step + 1), 'aug_loss: {:.4f}'.format(aug_loss.data.item()), 'bce_loss: {:.4f}'.format(bce_loss.data.item()), 'loss: {:.4f}'.format(np.mean(loss_per_epoch)), 'time: {:.4f}'.format(time.time() - t1))

            torch.save(model.state_dict(), preserve_dir + '/model_{}.pkl'.format(n_d * num_epoch + n_ep + 1))
            print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'time: {:.4f}'.format(time.time() - t0))
        del train_candidate, train_user, train_label
    
    [val_candidate, val_user, val_mask, val_label, val_index] = data.pre_val_behaviors(file4)
    val_candidate = torch.LongTensor(val_candidate)

    f = open(preserve_dir + '/val_label.pkl', 'wb')
    pickle.dump(val_label, f)
    f.close()
    f = open(preserve_dir + '/val_index.pkl', 'wb')
    pickle.dump(val_index, f)
    f.close()   
    
    truth_file = open(preserve_dir + '/truth.txt', 'w')
    for i in val_index:
        i_label = val_label[i[0]: i[1]].data.numpy().tolist()
        truth_file.write(str(val_index.index(i)) + ' ' + '[')
        for item in i_label[:-1]:
            truth_file.write(str(item) + ',')
        truth_file.write(str(i_label[-1]) + ']' + '\n')
    truth_file.flush()
    truth_file.close()

    val_dataset = Data.TensorDataset(val_candidate, val_user, val_label)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size = batch_size * 3, shuffle=False, num_workers=2)
    
    for n_d in range(num_dataset * num_epoch):
        model.load_state_dict(torch.load(preserve_dir + '/model_{}.pkl'.format(n_d + 1)))
        model = model.cuda()
        
        model.eval()
        val_score = []
        t = time.time()        

        with torch.no_grad():
            for step, (candidate, user, label) in enumerate(val_loader):
                t1 = time.time()
                print ('index_of_batch_valdataset: ', step)
                candidate_title, candidate_abs = Variable(news_title[candidate].unsqueeze(dim = 1).cuda()), Variable(news_abs[candidate].unsqueeze(dim = 1).cuda())
                ori_his_title, ori_his_abs = Variable(news_title[ori_his[user]].cuda()), Variable(news_abs[ori_his[user]].cuda())
                gen_his_title, gen_his_abs = Variable(news_title[gen_his[user]].cuda()), Variable(news_abs[gen_his[user]].cuda())
                label = Variable(label.cuda())
                pos, candidate, ori, gen = (None, None), (candidate_title, candidate_abs), (ori_his_title, ori_his_abs), (gen_his_title, gen_his_abs)

                predict_logits, _ = model(pos, candidate, ori, gen, training = False)
                score = torch.sigmoid(predict_logits).cpu().data.numpy()
            
                val_score = val_score + score.tolist()
            print('val_time: {:.4f}'.format(time.time() - t), 'val_score.length: ', len(val_score))

        f = open(preserve_dir + '/val_score_{}.pkl'.format(n_d + 1), 'wb')
        pickle.dump(val_score, f)
        f.close()

        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(n_d + 1), 'w')
        print ('process predict_file_{} start'.format(n_d + 1))

        for i in val_index:
            i_score = [item for item in val_score[i[0]: i[1]]]
            i_score_sort = sorted(i_score, reverse=True)

            rank = []
            for item in i_score:
                rank.append(i_score_sort.index(item) + 1)

            predict_file.write(str(val_index.index(i)) + ' ' + '[')
            for item in rank[:-1]:
                predict_file.write(str(item) + ',')
            predict_file.write(str(rank[-1]) + ']' + '\n')

        predict_file.flush()
        predict_file.close()
        print ('process predict_file_{} finished'.format(n_d + 1))

        print ('calculate {}_th auc/mrr/ndcg start'.format(n_d + 1))
        output_filename = preserve_dir + '/scores_{}.txt'.format(n_d + 1)
        output_file = open(output_filename, 'w')

        truth_file = open(preserve_dir + '/truth.txt', 'r')
        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(n_d + 1), 'r')

        auc, mrr, ndcg, ndcg10 = scoring(truth_file, predict_file)

        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
        output_file.close()
        print ('calculate {}_th auc/mrr/ndcg finished'.format(n_d + 1))
    del val_candidate, val_user, val_label, val_index
    