import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class FC(nn.Module):
    def __init__(self, dims):
        super(FC, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            layers.append(weight_norm(nn.Linear(dims[i], dims[i+1]), dim=None))
            layers.append(nn.ReLU())
        len_dims = len(dims)
        layers.append(weight_norm(nn.Linear(dims[len_dims-2], dims[len_dims-1]), dim=None))
        layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class Attention(nn.Module):
    def __init__(self, x_dim, q_dim, hidden_size, drop=0.0):
        super(Attention, self).__init__()
        self.x_dim = x_dim
        self.dropout = nn.Dropout(drop)
        self.x_with_q_att = weight_norm(nn.Linear((x_dim+q_dim), hidden_size), dim=None)
        self.fc = FC([x_dim + q_dim, hidden_size])
        self.x_proj = FC([x_dim, hidden_size])  # x_dim -> H
        self.q_proj = FC([q_dim, hidden_size])  # q_dim(H*2) -> H
        self.linear = weight_norm(nn.Linear(hidden_size, 1), dim=None)

    def scores_cat(self, x, q):
        n = x.size(1)
        q = q.unsqueeze(1).repeat(1, n, 1)          # q:[B, H*2] -> q:[B, F, H]
        x_with_q = torch.cat((x, q), -1)            # x_with_q: [B, F, x_dim+q_dim]
        x_with_q = self.fc(x_with_q)                # x_with_q: [B, F, H]
        scores = self.linear(x_with_q)              # [B, F, 1]

        return scores

    def scores_dot(self, x, q):
        n = x.size(1)                                      # n == F(feature nums)
        # print(':::::: 1',x.size(),q.size())
        q = self.q_proj(q).unsqueeze(1).repeat(1, n, 1)    # q:[B, H*2] -> q:[B, F, H]
        # print(':::::: 2', q.size(), 'x_dim', self.x_dim)
        x = self.x_proj(x)
        # print(':::::: 3', x.size())
        joint_repr = x * q                    # x:[B, F, x_dim] -> x_q:[B, F, H]
        # print(':::::: 4', joint_repr.size())
        joint_repr = self.dropout(joint_repr)
        scores = self.linear(joint_repr)                   # [B, F, 1]
        # print(':::::: 5', scores.size())
        return scores

    def forward(self, x, q, concat=False):
        """
                x: modality
                F == number of features (Either audio or image)
                :param x shape: [B, F, x_dim]
                :param q shape: [B, H*2(hidden_size*2)]
                :return : AttentionWeight w softmax [B, F]
        """
        if concat:
            scores = self.scores_cat(x, q)
        else:
            scores = self.scores_dot(x, q)
        return F.softmax(scores, dim=1).squeeze()      # [B, F, 1] -> # [B, F]










