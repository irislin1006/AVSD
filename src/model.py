from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from attention import Attention, FC
from language_model import WordEmbedding, SequenceEmbedding

class BaselineEncoder(nn.Module):
    def __init__(self, v_dim, a_dim, l_dim, hidden_size, w_emb, l_emb):
        """Encode language prior with different modality features"""
        super(BaselineEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.w_emb = w_emb                                  # WordEmbedding
        self.l_emb = l_emb                                  # SequenceEmbedding
        self.v_att = Attention(v_dim, l_dim, hidden_size)   # Attention(vis_hid_dim, dialoge_dim, hidden_size)
        self.a_att = Attention(a_dim, l_dim, hidden_size)   # Attention(aud_hid_dim, dialoge_dim, hidden_size)
        self.c2d_v = FC([v_dim, hidden_size])               # On paper
        self.c2d_a = FC([a_dim, hidden_size])
        # self.tanh = nn.Tanh()

    def forward(self, v, a, d):
        """
        Using dialoge as language guide
        :param v: [B, V_F, v_dim]
        :param a: [B, A_F, a_dim]
        :param d: [B, 10, seq_len]
        :return: c_v: visual context        [B, v_dim]
                 c_a: audio context         [B, a_dim]
                 d_v: on paper?             [B, H]
                 d_a:                       [B, H]
                 l_emb: dialoge embedded    [B, H*2]
        """

        # getting my language prior
        w_emb = self.w_emb(d)
        l_emb = self.l_emb(w_emb)                       # [B, H*2]

        attn_weight_v = self.v_att(v, l_emb)            # [B, V_F]
        attn_weight_v = attn_weight_v.unsqueeze(1)
        attn_weight_a = self.a_att(a, l_emb)            # [B, A_F]
        attn_weight_a = attn_weight_a.unsqueeze(1)

        c_v = torch.bmm(attn_weight_v, v)               # [B, 1, v_dim]
        c_a = torch.bmm(attn_weight_a, a)               # [B, 1, a_dim]

        # assert c_v.size(3) == self.hidden_size and c_a.size(3) == self.hidden_size
        d_v = self.c2d_v(c_v)                           # [B, 1, H]
        d_a = self.c2d_a(c_a)                           # [B, 1, H]
        return c_v.squeeze(), c_a.squeeze(), d_v.squeeze(), d_a.squeeze(), w_emb, l_emb

class MultiAttnFusion(nn.Module):
    def __init__(self, x_dim, l_dim, hidden_size):
        super(MultiAttnFusion, self).__init__()
        # self.l_emb = l_emb
        self.m_att = Attention(x_dim, l_dim, hidden_size)    # Attention(?dim, dialoge_dim, hidden_size)
        self.tanh = nn.Tanh()

    def _gav(self, l_emb, c_v, c_a, d_v, d_a):
        """Using language as guide to choose modality"""
        c = self._cat(c_v, c_a)     # [[B, v_dim],[B, a_dim]] -> [B, modality_num=2, a_dim]
        d = self._cat(d_v, d_a)     # [B, 2, H]
        # print('\nself.m_att(c, l_emb)')
        beta = self.m_att(c, l_emb) # [B, 2]
        beta = beta.unsqueeze(1)    # [B, 1, 2]
        gav = torch.bmm(beta, d)    # [B, 1, H]
        gav = self.tanh(gav)
        return gav

    def _cat(self, v, a):
        """Return concat at dim=1"""
        v = v.unsqueeze(1)
        a = a.unsqueeze(1)
        return torch.cat((v, a), dim=1)

    def forward(self, c_v, c_a, d_v, d_a, l_emb):
        """
        Modality Fusion
        c_v, c_a, d_v, d_a, l_emb: outputs of BaselineEncoder
        :return: modality features [B, H*3]
        """
        gq = l_emb.unsqueeze(1)                          # [B, 1, H*2]
        gav = self._gav(l_emb, c_v, c_a, d_v, d_a)       # [B, 1, H]
        g = torch.cat((gav,gq), dim=2)                   # [B, 1, H*3]
        return g.squeeze()

class Decoder(nn.Module):
    def __init__(self, emb_size, hidden_size, voc_size, w_emb, n_layers=1, drop=0.0, max_len=20):
        super(Decoder, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = w_emb # self.embed = nn.Embedding(voc_size, emb_size)
        # self.l_emb = l_emb
        self.dropout = nn.Dropout(drop, inplace=True)
        self.gru = nn.GRU(emb_size,
                          hidden_size,
                          n_layers,
                          dropout=drop,
                          batch_first=True)
        self.linear = FC([hidden_size * 3, emb_size])
        # self.out = nn.Linear(hidden_size, voc_size)
        self.out = FC([hidden_size, voc_size])
        self.tohid = FC([hidden_size * 3, hidden_size])

    def forward(self, caption, prev_hid, features, lengths):
        """
        Train/Valid
        :param captions: [B, padded_len]
        :param prev_hid: [n_layers(1), B, H]
        :param features: [B, H*3]
        :return: output [B, voc_size(5716)], hidden [B, n_layers, H]
        """
        # print(caption.size())
        embeddings = self.embed(caption)                   # [B, padded_len, emb_size]
        embeddings = self.dropout(embeddings)
        features = self.linear(features).unsqueeze(1)#.repeat(1, self.n_layers, 1)      # [B, 1, H]
        embeddings = torch.cat((features, embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        output, hidden = self.gru(packed)      # output , hidden [n_layers, B, H]
        # print(':::::::::::: out', len(output))
        outputs = self.out(output[0])
        return outputs, output[1]

    def sample(self, start_token, features):
        """
        Greedy
        :param features: [B, H*3]
        :param prev_hid:
        :return: [B, maxlen]
        """
        # print('start_token',start_token,start_token.size(),'\n') #start_token tensor(1, device='cuda:0') torch.Size([])
        sample_ids = []
        batch = features.size()
        # print('features.size()', features.size()) # torch.Size([1536])

        embeddings = self.embed(start_token) # [emb_size] print('embeddings.size()', embeddings.size())
        embeddings = embeddings.unsqueeze(0).unsqueeze(0) #[1, 1, emb_size]

        prev_hid = self.tohid(features.unsqueeze(0)).unsqueeze(0)  # [1, B(1),  hidden_size]

        for i in range(self.max_len):
            output, hidden = self.gru(embeddings, prev_hid)  # output [B, 1, H], hidden [B, n_layers, H]
            prev_hid = hidden
            output = self.out(output.squeeze(1))
            _, predicted = output.max(1)                 # predicted: [B]
            sample_ids.append(predicted)
            embeddings = self.embed(predicted).unsqueeze(1)

        sample_ids = torch.stack(sample_ids, 1)
        # print(sample_ids)
        return sample_ids

class BaseModel(nn.Module):
    def __init__(self, args, pretrained_embedding, v_dim, a_dim, l_dim):
        super(BaseModel, self).__init__()
        self.args = args
        # init pretrained word embedding
        vocab_size = len(pretrained_embedding)
        w_emb = WordEmbedding(vocab_size=vocab_size, emb_size=args.embed_size, drop=args.dropout)
        w_emb.init_wemb(pretrained_embedding)
        # language/dialoge embedding
        l_emb = SequenceEmbedding(emb_size=args.embed_size,
                                  hidden_size=args.hidden_size,
                                  n_layers=args.num_layers,
                                  drop=args.dropout)
        # Models
        self.encoder = BaselineEncoder(v_dim=v_dim,
                                  a_dim=a_dim,
                                  l_dim=l_dim,
                                  hidden_size=args.hidden_size,
                                  w_emb=w_emb,
                                  l_emb=l_emb)

        self.fuse_features = MultiAttnFusion(x_dim=v_dim, l_dim=l_dim, hidden_size=args.hidden_size)

        self.decoder = Decoder(emb_size=args.embed_size,
                          hidden_size=args.hidden_size,
                          voc_size=vocab_size,
                          w_emb=w_emb,
                          n_layers=args.num_layers,
                          drop=args.dropout,
                          max_len=args.maxlen)

    def forward(self, images, audios, dialoges, captions, lengths):
        batch = images.size(0)
        # Forward, backward and optimize
        c_v, c_a, d_v, d_a, w_emb, l_emb = self.encoder(images, audios, dialoges)
        g = self.fuse_features(c_v, c_a, d_v, d_a, l_emb)
        #bug

        hidden = l_emb.view(-1, batch, self.args.hidden_size)  # [n_layers*2, B, H] -> use linear
        hidden = hidden[:self.decoder.n_layers]  # only choose one direction [n_layers*1, B, H]
        outputs, bs = self.decoder(captions, hidden, g, lengths)
        # self.features = g
        # print('//////////// ', outputs.size())
        return outputs, bs, g

    def _sample(self, start_token, features):
        return self.decoder.sample(start_token, features)


