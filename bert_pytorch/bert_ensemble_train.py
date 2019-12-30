from model.bert import BERT
import bert_pytorch
import torch
import torch.nn as nn
from dataset.dataset import BERTDataset
from dataset.vocab import WordVocab
from torch.utils.data import DataLoader
import csv
import random
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from ENSEMBLE_BERT import DeepSEA_BERT
import os
import argparse
base_dict = {'A':0, 'G':1, 'C':2, 'T':3}
def seq2idxseq(seq, base_dict):
    idx_list = [np.zeros(len(seq)) for i in range(len(base_dict))]
    for i in range(len(seq)):
        idx_list[base_dict[seq[i]]][i] = 1
    return idx_list
def tensor_generate(data_set):
    x = torch.FloatTensor(np.array([seq2idxseq(x[0], base_dict) for x in data_set])).cuda()
    y = torch.FloatTensor(np.array([x[1] for x in data_set])).cuda()
    return x,y


def seq_split(seq):
    head = list(seq)
    return ' '.join(head)
def sentence_processing(sentence1):
    out1 = [vocab.sos_index] + vocab.to_seq(sentence1) + [vocab.eos_index] 
    seg_labels = [1 for x in out1]
    return (out1, seg_labels)
def data_generate(data_set):
    seq_list= []
    label_list = []
    target_list = []
    for row in data_set:
        curr_seq = seq_split(row[0])
        bert_seq,bert_label = sentence_processing(curr_seq)
        seq_list.append(bert_seq)
        label_list.append(bert_label)
        target_list.append(row[1])
    return torch.LongTensor(seq_list).cuda(),torch.LongTensor(label_list).cuda(),torch.FloatTensor(target_list).cuda()



def training_loop(model, loss, optimizer, epochs,batch_size, saved_path):
    train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    for j in range(epochs):
        loss_sum = 0
        for i,data in enumerate(train_loader):
            if i % 2000 == 0:
                print(i)
            model.train()
            bert_seq,bert_label,sea, target = data
            if target.size(0) <= 1:
                continue
            model.zero_grad()
            output = model(bert_seq,bert_label, sea)
            lossy = loss(output, target)
            lossy.backward()
            optimizer.step()
            

            loss_sum = loss_sum + lossy.detach().cpu().numpy()
        torch.save(model, os.path.join(saved_path, 'bert.ensemble.ep'+ str(j+1)))
        print( "Epochs %i; Loss %f, Validation Accuracy %f" %(j+1, loss_sum, test_model(valid_dataset, model, batch_size, th=0.5)))
        

def test_model(xy_set, model, batch_size, th=0.5):
    model.eval()
    data_loader = DataLoader(dataset=xy_set,batch_size=batch_size,shuffle=True)
    hit = 0
    total = 0
    for i,data in enumerate(data_loader):
        bert_seq,bert_label,sea,target = data
        out = model(bert_seq,bert_label, sea)
        out = out.detach().cpu().numpy()
        label = target.cpu().numpy()
        for j in range(len(label)):
            curr = 0
            if out[j] > th:
                curr = 1
            if label[j] == curr:
                hit = hit + 1
            total = total + 1
    return hit / total




parser = argparse.ArgumentParser()
parser.add_argument("--train_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input train corpus.")

parser.add_argument("--validation_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input validation corpus.")

parser.add_argument("--learning_rate",
                        default=0.0001,
                        type=float,
                        required=False,
                        help="model learning rate.")

parser.add_argument("--epochs",
                        default=10,
                        type=int,
                        required=False,
                        help="model training epochs.")

parser.add_argument("--dropout_prob",
                        default=0.1,
                        type=float,
                        required=False,
                        help="model dropout prob.")

parser.add_argument("--deepsea_out_size",
                        default=256,
                        type=int,
                        required=False,
                        help="deepsea out size.")

parser.add_argument("--batch_size",
                        default=32,
                        type=int,
                        required=False,
                        help="batch size")

parser.add_argument("--vocab",
                        default='vocab.small',
                        type=str,
                        required=False,
                        help="vocab path.")

parser.add_argument("--bert_path",
                        default=None,
                        type=str,
                        required=True,
                        help="bert path.")

parser.add_argument("--saved_path",
                        default=None,
                        type=str,
                        required=True,
                        help="save path.")
args = parser.parse_args()

train_set = []
with open(args.train_corpus, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        train_set.append([row[1], int(row[0])])

valid_set = []
with open(args.validation_corpus, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        valid_set.append([row[1], int(row[0])])

vocab = WordVocab.load_vocab(args.vocab)

sea_train,_ = tensor_generate(train_set)
sea_valid,_ = tensor_generate(valid_set)

seq_list,label_list,target_list = data_generate(train_set)
train_dataset = TensorDataset(seq_list,label_list,sea_train,target_list)
seq_list,label_list,target_list = data_generate(valid_set)
valid_dataset = TensorDataset(seq_list,label_list,sea_valid,target_list)


bert_model = torch.load(args.bert_path).cuda()
classifiy_model = DeepSEA_BERT(bert_model, args.deepsea_out_size, args.dropout_prob).cuda()
#classifiy_model = torch.load('output/bert.fine_tune.ep9').cuda()


print('Epochs %i; Batch_size %i, Learning Rate %f,dropout_prob %f' %(args.epochs, args.batch_size, args.learning_rate, args.dropout_prob))
loss = nn.BCELoss()
optimizer = torch.optim.Adam(classifiy_model.parameters(), lr=args.learning_rate)
training_loop(classifiy_model, loss, optimizer, args.epochs, args.batch_size, args.saved_path)
