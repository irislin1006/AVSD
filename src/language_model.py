import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size, drop=0.0):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.wemb = nn.Embedding(vocab_size, emb_size)
        self.dropout = nn.Dropout(drop)

    def init_wemb(self, pretrained_embedding):
        pretrained_embedding = torch.from_numpy(pretrained_embedding)
        assert pretrained_embedding.shape == (self.vocab_size, self.emb_size)
        self.wemb.weight.data.copy_(pretrained_embedding)

    def forward(self, x):
        """
        :param x: [B, seq_len]
        :return: [10, B, seq_len, emb_dim]
        """
        wemb = self.wemb(x)
        wemb = self.dropout(wemb)
        # if x.size(0) !=32:
        #     print('::WE::')
        #     print('w_emb', wemb.size(),'\n')
        return wemb

class SequenceEmbedding(nn.Module):
    def __init__(self, emb_size, hidden_size, n_layers=1, drop=0.0):
        super(SequenceEmbedding, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=True, dropout=drop)

    def forward(self, embedd_question):
        """
        :param embedd_question: language after wordembedding [B, 10, seq_len, emb_dim]
        :return: a bidirectional lstm 2*hidden [B, 10, hidden_size*2]
        """
        B, dnum, seq_len, emb_dim = embedd_question.size()
        h, c = self.init_h_c(B)
        for i, embedd_qa in enumerate(embedd_question.permute(1, 0, 2, 3)):  # embedd_qa (B, seq_len, emb_dim)
            lstm_in = embedd_qa.permute(1, 0, 2)
            lstm_out, (h, c) = self.lstm(lstm_in, (h,c))
            # lstm_hidden_state.append(c)     #(2, B, hidden)

        # print(c.size())
        lstm_hidden_state = c   #(n_layers * num_dirs, batch, hidden_size) == (2, B*10, hidden)
        lstm_hidden_state = lstm_hidden_state.permute(1, 0, 2)
        lstm_hidden_state = lstm_hidden_state.contiguous().view(-1, self.hidden_size*2)
        # if embedd_question.size(0) !=32:
        #     print('::SE::')
        #     print('w_emb', lstm_hidden_state.size(),'\n')
        return lstm_hidden_state#.long()

    def init_h_c(self, B):
        return torch.zeros(self.n_layers*2, B, self.hidden_size).cuda(), \
               torch.zeros(self.n_layers*2, B, self.hidden_size).cuda()


