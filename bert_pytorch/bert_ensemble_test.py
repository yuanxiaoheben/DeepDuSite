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
from sklearn.metrics import recall_score,precision_score,f1_score,accuracy_score
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
       

def test_model_out(xy_set, model, batch_size, th=0.5):
    model.eval()
    data_loader = DataLoader(dataset=xy_set,batch_size=batch_size,shuffle=True)
    true_out = []
    model_out = []
    for i,data in enumerate(data_loader):
        bert_seq,bert_label,sea,target = data
        out = model(bert_seq,bert_label,sea)
        out = out.detach().cpu().numpy()
        target = target.cpu().numpy()
        for j in range(len(target)):
            curr = 0
            if out[j] > th:
                curr = 1
            true_out.append(target[j])
            model_out.append(curr)
    return model_out, true_out




parser = argparse.ArgumentParser()
parser.add_argument("--test_corpus",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test corpus.")

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

parser.add_argument("--model_path",
                        default=None,
                        type=str,
                        required=True,
                        help="model path.")

args = parser.parse_args()

test_set = []
with open(args.test_corpus, 'r') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in csv_reader:
        test_set.append([row[1], int(row[0])])


vocab = WordVocab.load_vocab(args.vocab)

sea_test,_ = tensor_generate(test_set)

seq_list,label_list,target_list = data_generate(test_set)
test_dataset = TensorDataset(seq_list,label_list,sea_test,target_list)


classifiy_model = torch.load(args.model_path).cuda()
model_out, true_out = test_model_out(test_dataset, classifiy_model, args.batch_size)
print('Recall:' + str(recall_score(true_out, model_out)))
print('Precision:' + str(precision_score(true_out, model_out)))
print('F1:' + str(f1_score(true_out, model_out)))
print('Accuracy:' + str(accuracy_score(true_out, model_out)))
