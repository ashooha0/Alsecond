import copy
import math

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
torch.nn.TransformerDecoderLayer

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(3).expand_as(scores)
        # print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SelfAttention(torch.nn.Module):
    def __init__(self, c_args, d_model, dropout=0.5, dim_feedforward=100, layer_norm_eps=1e-5):
        super(SelfAttention,self).__init__()
        self.c_args = c_args
        self.linear_q = torch.nn.Linear(c_args["lstm_dim"] * 2, c_args["lstm_dim"] * 2)
        # self.linear_k = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.linear_v = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.w_query = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        # self.w_value = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        self.dropout = Dropout(dropout)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.relu = F.relu

        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.w_query = torch.nn.Linear(c_args["cnn_dim"], 50)
        self.w_value = torch.nn.Linear(c_args["cnn_dim"], 50)
        self.v = torch.nn.Linear(50, 1, bias=False)

    def forward(self, query, value, mask):
        # attention_states = self.linear_q(query)
        # attention_states_T = self.linear_k(values)
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])
        # print(mask.shape)
        # print(mask)


        weights=torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))    #   mask掉每行后面的列
        attention = F.softmax(weights,dim=2)

        # value=self.linear_v(states)
        merged=torch.bmm(attention, value)
        merged = self.dropout1(merged) + value
        # merged = self.norm1(merged)
        # merged2 = self.linear2(self.dropout(self.relu(self.linear1(merged))))
        # merged = merged + self.dropout2(merged2)
        merged = self.norm2(merged)
        merged=merged * mask.unsqueeze(2).float().expand_as(merged)

        return merged


class PairAwareSelfAttention(torch.nn.Module):  # 配对情况下的自注意力
    def __init__(self,  c_args, d_model, dropout=0.5, dim_feedforward=1024, layer_norm_eps=1e-5):
        super(PairAwareSelfAttention, self).__init__()
        self.c_args = c_args
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(dim_feedforward, d_model*2)
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = LayerNorm(d_model*2, eps=layer_norm_eps)
        self.relu = F.relu
        # self.linear_q = torch.nn.Linear(c_args["lstm_dim"] * 2, c_args["lstm_dim"] * 2)
        # self.linear_k = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.linear_v = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.w_query = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        # self.w_value = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        # self.w_query = torch.nn.Linear(c_args["cnn_dim"], 50)
        # self.w_value = torch.nn.Linear(c_args["cnn_dim"], 50)
        # self.v = torch.nn.Linear(50, 1, bias=False)

    def forward(self, query, value, mask):
        '''
        input: query, value
        output: 2D for a-o pair
        '''
        # query = query + value
        # attention_states = self.linear_q(query)
        # attention_states_T = self.linear_k(values)
        max_seq_len = query.shape[1]
        mmss = torch.eye(max_seq_len, device=self.args.device).unsqueeze(1).expand([max_seq_len, max_seq_len, max_seq_len]).reshape(
            max_seq_len * max_seq_len, max_seq_len)
        q_expd = torch.matmul(mmss, query).reshape(query.shape[0], query.shape[1],
                                                   query.shape[1], query.shape[2])
        v_expd = value.unsqueeze(1)

        final_queryer = (q_expd + v_expd) / 2.
        # final_queryer = self.linear1(final_queryer)
        v_expd, q_v_expd = v_expd, query.unsqueeze(1)
        v_expd_T = v_expd.transpose(2, 3)
        q_v_expd_T = q_v_expd.transpose(2, 3)
        weight = torch.matmul(final_queryer, v_expd_T)
        weight_q = torch.matmul(final_queryer, q_v_expd_T)
        weight = weight.masked_fill(mask.unsqueeze(1).unsqueeze(1).expand_as(weight) == 0, float("-inf"))
        weight_q = weight_q.masked_fill(mask.unsqueeze(1).unsqueeze(1).expand_as(weight_q) == 0, float("-inf"))
        # print(weight)
        attention = F.softmax(weight, dim=3)
        attention_q = F.softmax(weight_q, dim=3)
        # print(attention)
        merged = torch.matmul(attention, v_expd)
        # merged = self.norm1(merged)
        merged_q = torch.matmul(attention_q, q_v_expd)
        # merged_q = self.norm2(merged_q)

        merged = torch.cat((merged, merged_q), dim=-1)
        # merged2 = self.linear2(self.dropout1(self.relu(self.linear1(merged))))
        # merged = merged + self.dropout2(merged2)
        merged = self.norm3(merged)

        merged = merged * mask.unsqueeze(2).unsqueeze(3).float().expand_as(merged)
        # merged = self.relu(merged)
        return merged
        # print(merged)

        # attention_states = query
        # attention_states_T = value
        # attention_states_T = attention_states_T.permute([0, 2, 1])
        #
        # weights = torch.bmm(attention_states, attention_states_T)
        # weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights) == 0, float("-inf"))  # mask掉每行后面的列
        # attention = F.softmax(weights, dim=2)
        #
        # # value=self.linear_v(states)
        # merged = torch.bmm(attention, value)
        # merged = merged * mask.unsqueeze(2).float().expand_as(merged)
        #
        # return merged
