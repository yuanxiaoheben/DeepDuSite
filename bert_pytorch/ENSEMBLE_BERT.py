import torch
import torch.nn as nn
class DeepSEA_BERT(nn.Module):

    def __init__(self, bert_model, sea_out_size, dropout_prob,sequence_length=400, bert_hidden_size=256):
        super(DeepSEA_BERT, self).__init__()
        conv_kernel_size = 8
        pool_kernel_size = 4
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(320),

            nn.Conv1d(320, 480, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=pool_kernel_size, stride=pool_kernel_size),
            nn.BatchNorm1d(480),
            nn.Dropout(p=0.2),

            nn.Conv1d(480, 960, kernel_size=conv_kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(960),
            nn.Dropout(p=0.2))
        
        pool_kernel_size = float(pool_kernel_size)
        self._n_channels = 15
        self.deepsea_encoder = nn.Sequential(
            nn.Linear(960 * self._n_channels, sea_out_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(sea_out_size),
            nn.Linear(sea_out_size, sea_out_size))
        self.classifier = nn.Sequential(
            nn.Linear( bert_hidden_size+sea_out_size, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1),
            nn.Linear(1, 1),
            nn.Sigmoid())

    def forward(self, sequence, segment_label, deepsea_seq):
        """
        Forward propagation of a batch.
        """
        x = self.bert(sequence, segment_label)
        pooled_output = x[:,0]
        pooled_output = self.dropout(pooled_output)

        out = self.conv_net(deepsea_seq)
        self._n_channels = out.size(2)
        reshape_out = out.view(out.size(0), 960 * self._n_channels)
        #deepsea_out = self.dropout2(self.deepsea_encoder(reshape_out))
        deepsea_out = self.deepsea_encoder(reshape_out)
        cat_out = torch.cat((deepsea_out, pooled_output),1)
        cat_out = self.classifier(cat_out)
        return cat_out