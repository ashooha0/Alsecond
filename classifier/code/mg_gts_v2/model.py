import torch
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
import copy
from classifier.code.mg_gts_v2.attention_module import MultiHeadedAttention, SelfAttention, PairAwareSelfAttention
# torch.nn.TransformerDecoderLayer

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
class CNN_2D_Group(torch.nn.Module):
    def __init__(self,  c_args, dropout, num_layers=2):
        super(CNN_2D_Group, self).__init__()
        self.dropout = dropout
        self.c_args = c_args
        in_channel = c_args["hidden_dim"] * 2
        out_channel = c_args["hidden_dim"]
        self.layers3 = _get_clones(torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                                   kernel_size=(3, 3), padding=1), num_layers - 1)
        self.layers3.append(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=(3, 3), padding=1))

        self.layers5 = _get_clones(torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                                   kernel_size=(5, 5), padding=2), num_layers - 1)
        self.layers5.append(torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                            kernel_size=(5, 5), padding=2))

        self.layers_top = _get_clones(torch.nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel * 2,
                                                      kernel_size=(5, 5), padding=2), num_layers - 1)
        self.layers_top.append(torch.nn.Conv2d(in_channels=out_channel * 2, out_channels=out_channel,
                                               kernel_size=(3, 3), padding=1))
        self.single_1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(1, 1))
        self.single_3 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                        kernel_size=(3, 3), padding=1)
        self.single_5 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                        kernel_size=(5, 5), padding=2)
        self.single_c = _get_clones(torch.nn.Conv2d(in_channels=out_channel*3, out_channels=out_channel,
                                                    kernel_size=(7, 7), padding=3), 1)
        self.single_c.append(torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                             kernel_size=(5, 5), padding=2))
        self.single_c.append(torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                             kernel_size=(3, 3), padding=1))

        #         self.final_conv = torch.nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel,
        #                                           kernel_size=(3, 3), padding=1)

    def forward(self, src):
        inputs = src.permute(0, 3, 1, 2)
        output1 = self.single_1(inputs)
        output3 = self.single_3(inputs)
        output5 = self.single_5(inputs)
        output = torch.cat((output1, output3, output5), dim=1)
        output = self.dropout(torch.nn.functional.relu(output))
        for mod in self.single_c:
            output = torch.nn.functional.relu(mod(output))
            output = self.dropout(output)

        # for mod in self.layers3:
        #     output3 = mod(output3)[:, :, :self.c_args["max_sequence_len"]]
        #
        # for mod in self.layers5:
        #     output5 = mod(output5)[:, :, :self.c_args["max_sequence_len"]]
        #
        # output = torch.cat((output3, output5), dim=1)
        # output = self.dropout(torch.nn.functional.relu(output))
        # for mod in self.layers_top:
        #     output = torch.nn.functional.relu(mod(output))
        #     output = self.dropout(output)

        output = output3.permute(0, 2, 3, 1)
        return output


class CNN_1D_Group(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, c_args, dropout):
        super(CNN_1D_Group, self).__init__()
        self.c_args = c_args
        self.dropout = dropout
        self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 1)
        self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 2, padding=1)
        self.conv3 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 3, padding=1)
        self.conv4 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 4, padding=2)
        self.conv5 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 5, padding=2)
        self.conv6 = torch.nn.Conv1d(5 * c_args["hidden_dim"], 5 * c_args["hidden_dim"], 5, padding=2)
        self.conv7 = torch.nn.Conv1d(5 * c_args["hidden_dim"], 3 * c_args["hidden_dim"], 5, padding=2)
        self.conv8 = torch.nn.Conv1d(3 * c_args["hidden_dim"], 2 * c_args["hidden_dim"], 3, padding=1)

    def forward(self, embedding):
        word_emd = embedding.transpose(1, 2)
        word1_emd = self.conv1(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        word2_emd = self.conv2(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        word3_emd = self.conv3(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        word4_emd = self.conv4(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        word5_emd = self.conv5(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        x_emb = torch.cat((word1_emd, word2_emd), dim=1)
        x_emb = torch.cat((x_emb, word3_emd), dim=1)
        x_emb = torch.cat((x_emb, word4_emd), dim=1)
        x_emb = torch.cat((x_emb, word5_emd), dim=1)

        x_conv = self.dropout(torch.nn.functional.relu(x_emb))

        x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv7(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv8(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = x_conv.transpose(1, 2)
        return x_conv

class MG_GTS(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb, c_args):
        super(MG_GTS, self).__init__()
        self.c_args = c_args
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight.data.copy_(gen_emb)
        self.gen_embedding.weight.requires_grad = False

        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight.data.copy_(domain_emb)
        self.domain_embedding.weight.requires_grad = False
        
        self.dropout = torch.nn.Dropout(0.5)
        
        # self.conv1 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 1)
        # self.conv2 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 2, padding=1)
        # self.conv3 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 3, padding=1)
        # self.conv4 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 4, padding=2)
        # self.conv5 = torch.nn.Conv1d(gen_emb.shape[1] + domain_emb.shape[1], c_args["hidden_dim"], 5, padding=2)
        # self.conv6 = torch.nn.Conv1d(5*c_args["hidden_dim"], 5*c_args["hidden_dim"], 5, padding=2)
        # self.conv7 = torch.nn.Conv1d(5*c_args["hidden_dim"], 3*c_args["hidden_dim"], 5, padding=2)
        # self.conv8 = torch.nn.Conv1d(3*c_args["hidden_dim"], 2*c_args["hidden_dim"], 3, padding=1)
        self.cnn_1d_group1 = CNN_1D_Group(gen_emb, domain_emb, c_args, self.dropout)
        self.cnn_1d_group2 = CNN_1D_Group(gen_emb, domain_emb, c_args, self.dropout)

        self.bilstm = torch.nn.LSTM(2*c_args["hidden_dim"],
                                    c_args["hidden_dim"], num_layers=1, batch_first=True, bidirectional=True)
        self.multi_head_attention_layer_a = MultiHeadedAttention(h=8, d_model=c_args["hidden_dim"]*2)
        # self.multi_head_attention_layer_o = MultiHeadedAttention(h=4, d_model=c_args["hidden_dim"] * 2)
        self.attention_layer = SelfAttention(c_args, c_args["hidden_dim"]*2)
        self.pair_attention_layer = PairAwareSelfAttention(c_args, c_args["hidden_dim"]*2)

        self.cnn_2d_group = CNN_2D_Group(c_args, self.dropout)

        self.cls_linear = torch.nn.Linear(c_args["hidden_dim"]*4, c_args["class_num"])

    def _get_embedding(self, sentence_tokens, mask):
        gen_embed = self.gen_embedding(sentence_tokens)
        domain_embed = self.domain_embedding(sentence_tokens)
        embedding = torch.cat([gen_embed, domain_embed], dim=2)
        embedding = self.dropout(embedding)
        embedding = embedding * mask.unsqueeze(2).float().expand_as(embedding)
        return embedding

    def _lstm_feature(self, embedding, lengths):
        embedding = pack_padded_sequence(embedding, lengths.cpu(), batch_first=True)
        context, _ = self.bilstm(embedding)
        context, _ = pad_packed_sequence(context, batch_first=True)
        return context

    def forward(self, sentence_tokens, lengths, masks):
        embedding = self._get_embedding(sentence_tokens, masks)
        # print(embedding.size())

        # span-chosen representation
        # word_emd = embedding.transpose(1, 2)
        # word1_emd = self.conv1(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        # word2_emd = self.conv2(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        # word3_emd = self.conv3(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        # word4_emd = self.conv4(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        # word5_emd = self.conv5(word_emd)[:, :, :self.c_args["max_sequence_len"]]
        # x_emb = torch.cat((word1_emd, word2_emd), dim=1)
        # x_emb = torch.cat((x_emb, word3_emd), dim=1)
        # x_emb = torch.cat((x_emb, word4_emd), dim=1)
        # x_emb = torch.cat((x_emb, word5_emd), dim=1)

        # x_conv = self.dropout(torch.nn.functional.relu(x_emb))
        #
        # x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        # x_conv = self.dropout(x_conv)
        # x_conv = torch.nn.functional.relu(self.conv7(x_conv))
        # x_conv = self.dropout(x_conv)
        # x_conv = torch.nn.functional.relu(self.conv8(x_conv))
        # x_conv = self.dropout(x_conv)
        x_conv1 = self.cnn_1d_group1(embedding)
        # x_conv2 = self.cnn_1d_group2(embedding)
        # x_conv = x_conv.transpose(1, 2)
        # x_conv = x_conv[:, :lengths[0], :]
        # print(x_conv.size())

        # contextual representation
        feature_a = self._lstm_feature(x_conv1, lengths)
        # feature_o = self._lstm_feature(x_conv2, lengths)

        # feature_attention = self.attention_layer(feature, feature, masks[:, :lengths[0]])
        feature_attention_a = self.multi_head_attention_layer_a(feature_a, feature_a, feature_a, masks[:, :lengths[0]])
        feature_a = feature_a + feature_attention_a
        # feature_attention_o = self.multi_head_attention_layer_o(feature_o, feature_o, feature_o, masks[:, :lengths[0]])
        # feature_o = feature_o + feature_attention_o

        feature_a = feature_a.unsqueeze(2).expand([-1, -1, lengths[0], -1])
        # feature_o = feature_o.unsqueeze(2).expand([-1, -1, lengths[0], -1])
        feature_a_T = feature_a.transpose(1, 2)
        features = torch.cat([feature_a, feature_a_T], dim=3)
        # feature_o = self.attention_layer(feature_o, feature_o, masks[:, :lengths[0]])
        # print(feature.size())
        # features = self.pair_attention_layer(feature_a, feature_o, masks[:, :lengths[0]])

        # features = self.cnn_2d_group(features)

        # feature = feature.unsqueeze(2).expand([-1, -1, lengths[0], -1])
        # feature_T = feature.transpose(1, 2)
        # features = torch.cat([feature, feature_T], dim=3)
        # print(features.size())

        logits = self.cls_linear(features)
        return logits

