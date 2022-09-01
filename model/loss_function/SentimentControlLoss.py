from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn


def constract_batch_T_len_count(batch: List[List[List]], device=None):
    len_batch = len(batch)
    if device is None:
        len_num = torch.ones((len_batch))  # B_
    else:
        len_num = torch.ones((len_batch), device=device)  # B_
    for i, asps in enumerate(batch):
        len_num[i] = sum([len(ap) for ap in asps])
    return len_num

def constract_batch_A_len_count(batch: List[List[List]], device=None):
    len_batch = len(batch)
    if device is None:
        len_num = torch.ones((len_batch))  # B_
    else:
        len_num = torch.ones((len_batch), device=device)  # B_
    for i, asps in enumerate(batch):
        len_num[i] = len(asps) if len(asps) > 0 else 1
    return len_num

# 2020/02/11
# def sentiment_control_loss(arg_config, input_labels, generation_logits_ori,    # B_S、B_S_V
#                            sentiment_atoms: List[List[List]], sub_text_seg=7359, padding_value=0,
#                            start_p=1, end_p=None):  # B_A'_T'
#     generation_logits = generation_logits_ori.clone()
#     assert input_labels.shape[:2] == generation_logits.shape[:2]
#     print(generation_logits)
#     # input_labels = input_labels.clone()  # B_S
#     # generation_logits = generation_logits.clone()   # B_S_V
#
#     second_lens = [len(atom) for atom in sentiment_atoms]
#     data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]
#     data = [[senti[start_p: None] for senti in unit] for unit in sentiment_atoms]  # B_A'_T'
#     batch_T_len_count = constract_batch_T_len_count(data, device=arg_config['data']['device'])  # B_
#     # print(batch_T_len_count)
#     # data = [d for da in data for d in da]
#     data = [torch.tensor(d, device=arg_config['data']['device']) for da in data for d in da]  # BA'_T'
#
#     data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # BA'_T
#     data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]
#     data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # B_A_T
#     A_len = data.shape[1]
#     V_len = generation_logits.shape[-1]  # V
#     S_len = generation_logits.shape[-2]  # S
#     data = F.one_hot(data.long(), num_classes=V_len)  # B_A_T_V
#     data[:, :, :, padding_value] = 0
#     data = data.float()
#     # print(data)
#
#     generation_logits = generation_logits.unsqueeze(dim=1)  # B_1_S_V
#     generation_logits = generation_logits.repeat(1, A_len, 1, 1)  # B_A_S_V
#     print(generation_logits)
#
#     generation_logits = torch.log(generation_logits)
#     A_mask = torch.zeros_like(generation_logits)
#     # A_mask = torch.zeros((generation_logits.shape[:-1]),  # B_A_S
#     #                      device=arg_config['data']['device'])
#     for idx in range(0, input_labels.shape[0]):         # 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
#         cur_A = 0                                       # 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
#         last_seg = 0                                    # 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0
#         for jdx in range(0, input_labels.shape[1]):     # 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0
#             if input_labels[idx, jdx] == sub_text_seg:  # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
#                 if jdx == 0:
#                     continue
#                 A_mask[idx, cur_A, last_seg: jdx] = 1.
#                 last_seg = jdx + 1
#                 cur_A = cur_A + 1
#
#     # print(A_mask)
#     generation_logits = generation_logits * A_mask  # B_A_S_V  改：B_A_S_V → B_A_1_V
#     print(generation_logits)
#     max_2d_pool = nn.MaxPool2d((S_len, 1), stride=1)
#     generation_logits = max_2d_pool(generation_logits)  # B_A_1_V
#     print(generation_logits)
#     generation_logits = generation_logits.transpose(2, 3)  # B_A_V_1
#     # print(generation_logits)
#     print(generation_logits)
#     grade = torch.matmul(data, generation_logits)  # B_A_T_1
#     # print(grade)
#     grade = grade.squeeze(dim=-1)  # B_A_T
#     # print(grade, grade.shape)
#     grade = grade.sum(dim=-1)  # B_A
#     grade = grade.sum(dim=-1)  # B_
#     print(grade, grade.shape)
#     # grade = grade / batch_T_len_count
#     # print(grade, grade.shape)
#     # grade = torch.log(grade)
#     # print(grade, grade.shape)
#
#     loss = -torch.sum(grade)
#     print(loss)
#     # print(grade)
#
#     return loss


# # TODO 按照如此思路修改  20220211
# def sentiment_control_loss(inputs, target):  # B_S, B  → B_A_S_V, B_A_S
#     V_len = inputs.shape[-1]
#     inputs = inputs.double()
#     target = F.one_hot(target.long(), num_classes=V_len)
#     minus_x = inputs * target
# #     print(minus_x)
#     minus_x = -torch.sum(minus_x, dim=-1)
# #     print(minus_x)
#     log_exp = torch.exp(inputs)
# #     print(log_exp)
#     log_exp = torch.sum(log_exp, dim=-1)
# #     print(log_exp)
#     log_exp = torch.log(log_exp)
# #     print(log_exp)
#     ret = minus_x + log_exp
# #     print(ret)
#     ret = torch.mean(ret)
#     return ret.float()

def get_one_hot(tensor_data, class_num, ignore_token):  # tensor_data
    ret = torch.where(tensor_data == ignore_token, class_num, tensor_data)
    # ignore_token = class_num
    ret = F.one_hot(ret.long(), num_classes=class_num+1)
    ret = ret[..., :-1]
    return ret


def sentiment_overall_loss(arg_config, input_labels, generation_logits,  # B_S、B_S_V
                           sentiment_atoms: List[List[List]], cross_entropy, confidences=None,  # B_A'_T'、B_A'
                           padding_value=-1, start_p=1, end_p=None, alpha=0., mask_mean=84):
    # assert input_labels.shape[:2] == generation_logits.shape[:2]
    second_lens = [len(atom) for atom in sentiment_atoms]

    if confidences is not None:
        _confidence_second_lens = [len(conf) for conf in confidences]
        assert _confidence_second_lens == second_lens
        confidences = [torch.tensor(conf, device=arg_config['data']['device']) for conf in confidences]  # B_A'
        confidences = pad_sequence(confidences, batch_first=True, padding_value=0.).unsqueeze(dim=-1)  # B_A_1
    data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]
    data = [[senti[start_p: end_p] for senti in unit] for unit in sentiment_atoms]  # B_A'_T'
    # print(batch_T_len_count)
    # data = [d for da in data for d in da]
    data = [torch.tensor(d, device=arg_config['data']['device']) for da in data for d in da]  # BA'_T'
    data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # BA'_T
    data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]  # B_A'_T
    data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # B_A_T

    A_len = data.shape[1]
    T_len = data.shape[2]
    V_len = generation_logits.shape[-1]  # V
    S_len = generation_logits.shape[-2]  # S

    generation_logits_s_sum = torch.sum(generation_logits, dim=1, keepdim=True)  # B_1_V
    max_2d_pool = nn.MaxPool2d((S_len, 1), stride=1)
    generation_logits = max_2d_pool(generation_logits)  # B_1_V
    generation_logits = generation_logits * 2 - generation_logits_s_sum
    generation_logits = generation_logits.unsqueeze(dim=1)  # B_1_1_V
    generation_logits = generation_logits.repeat(1, A_len, T_len, 1)  # B_A_T_V

    # 20220329
    A_mask = get_one_hot(data, class_num=V_len, ignore_token=padding_value)  # B_A_T_V
    generation_logits = generation_logits * A_mask + ((1. - A_mask) * mask_mean)
    # 20220329
    if confidences is not None:
        confidences = confidences.unsqueeze(dim=-1)  # B_A_1 → B_A_1_1
        confidenced_logits = generation_logits * (confidences ** alpha)
        return cross_entropy(confidenced_logits.view(-1, V_len), data.view(-1))
    else:
        return cross_entropy(generation_logits.view(-1, V_len), data.view(-1))

def sentiment_reinforce_control_loss(lm_logit_first_index, lm_labels_first_index,
                                     input_labels, generation_logits,  # B_S、B_S_V
                                     reinforce_control_masks, cross_entropy, confidences=None,  # B_S、B_A'
                                     padding_value=-1, start_p=1, end_p=None, alpha=0., mask_mean=84):
    compare_input_labels = input_labels[:, lm_labels_first_index:].contiguous()
    compare_generation_logits = generation_logits[:, lm_logit_first_index:, :].contiguous()
    reinforce_control_masks = reinforce_control_masks[:, 1:]
    assert compare_input_labels.shape[1] == compare_generation_logits.shape[1] == reinforce_control_masks.shape[1]
    masked_sequence_labels = compare_input_labels * reinforce_control_masks + ((1 - reinforce_control_masks) * padding_value)
    masked_sequence_labels = masked_sequence_labels.long()
    v_len = compare_generation_logits.shape[-1]
    return cross_entropy(compare_generation_logits.reshape(-1, v_len), masked_sequence_labels.view(-1))


def sentiment_control_loss(arg_config, input_labels, generation_logits_ori,  # B_S、B_S_V
                           sentiment_atoms: List[List[List]], confidences=None,  # B_A'_T'、B_A'
                           sub_text_seg=7359, padding_value=0, start_p=1, end_p=None, alpha=0.):
    generation_logits = generation_logits_ori.clone()
    assert input_labels.shape[:2] == generation_logits.shape[:2]
    # print(generation_logits)
    # input_labels = input_labels.clone()  # B_S
    # generation_logits = generation_logits.clone()   # B_S_V

    second_lens = [len(atom) for atom in sentiment_atoms]
    if confidences is not None:
        _confidence_second_lens = [len(conf) for conf in confidences]
        assert _confidence_second_lens == second_lens
        confidences = [torch.tensor(conf, device=arg_config['data']['device']) for conf in confidences]  # B_A'
        confidences = pad_sequence(confidences, batch_first=True, padding_value=0.).unsqueeze(dim=-1)  # B_A_1
    data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]
    data = [[senti[start_p: end_p] for senti in unit] for unit in sentiment_atoms]  # B_A'_T'
    batch_T_len_count = constract_batch_T_len_count(data, device=arg_config['data']['device'])  # B_
    batch_A_len_count = constract_batch_A_len_count(data, device=arg_config['data']['device'])  # B_
    # print(batch_T_len_count)
    # data = [d for da in data for d in da]
    data = [torch.tensor(d, device=arg_config['data']['device']) for da in data for d in da]  # BA'_T'
    data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # BA'_T
    data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)] # B_A'_T
    data = pad_sequence(data, batch_first=True, padding_value=padding_value)  # B_A_T

    A_len = data.shape[1]
    V_len = generation_logits.shape[-1]  # V
    S_len = generation_logits.shape[-2]  # S
    data = F.one_hot(data.long(), num_classes=V_len)  # B_A_T_V
    data[:, :, :, padding_value] = 0
    data = torch.sum(data, dim=2, keepdim=True)  # B_A_1_V
    T_count = torch.sum(data, dim=-1)  # B_A_1
    T_count = torch.clamp(T_count, min=1)  # 防止除0
    # data = data.float()
    # print(data)

    generation_logits = generation_logits.unsqueeze(dim=1)  # B_1_S_V
    generation_logits = generation_logits.repeat(1, A_len, 1, 1)  # B_A_S_V
    # TODO with torch.no_grad(): ?
    A_mask = torch.zeros_like(generation_logits)
    # A_mask = torch.zeros((generation_logits.shape[:-1]),  # B_A_S
    #                      device=arg_config['data']['device'])
    for idx in range(0, input_labels.shape[0]):           # 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        cur_A = 0                                         # 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0
        last_seg = 0                                      # 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0
        for jdx in range(0, input_labels.shape[1]):       # 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0
            if input_labels[idx, jdx] == sub_text_seg:    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1
                if jdx == 0:
                    continue
                A_mask[idx, cur_A, last_seg: jdx] = 1.
                last_seg = jdx + 1
                cur_A = cur_A + 1
    generation_logits = generation_logits * A_mask + ((1. - A_mask) * -600)
    # print(generation_logits)

    max_2d_pool = nn.MaxPool2d((S_len, 1), stride=1)
    generation_logits = max_2d_pool(generation_logits)  # B_A_1_V
    generation_logits = generation_logits.double()
    # print(generation_logits)
    minus_x = data * generation_logits  # B_A_1_V
    minus_x = -torch.sum(minus_x, dim=-1)  # B_A_1
    # zheli confidence mechanism
    if confidences is not None:
        minus_x = minus_x * (confidences ** alpha)  # B_A_1
    # zheli confidence mechanism
    minus_x = minus_x / T_count  # B_A_1
    log_exp = torch.exp(generation_logits)  # B_A_1_V
    log_exp = torch.sum(log_exp, dim=-1)  # B_A_1
    log_exp = torch.log(log_exp)
    loss = minus_x + log_exp  # B_A_1
    mask = torch.ones_like(loss)  # B_A_1
    for idx in range(0, data.shape[0]):
        for jdx in range(0, data.shape[1]):
            if torch.all(data[idx][jdx] == 0):
                mask[idx, jdx] = 0

    loss = loss * mask
    loss = loss.squeeze(dim=-1)  # B_A
    loss = torch.sum(loss, dim=-1)  # B_
    loss = loss / batch_A_len_count
    loss = torch.mean(loss)  # 1
    return loss.float()


    # generation_logits = torch.log(generation_logits)

    #
    # # print(A_mask)
    # generation_logits = generation_logits * A_mask  # B_A_S_V  改：B_A_S_V → B_A_1_V
    # print(generation_logits)
    #
    # generation_logits = generation_logits.transpose(2, 3)  # B_A_V_1
    # # print(generation_logits)
    # print(generation_logits)
    # grade = torch.matmul(data, generation_logits)  # B_A_T_1
    # # print(grade)
    # grade = grade.squeeze(dim=-1)  # B_A_T
    # # print(grade, grade.shape)
    # grade = grade.sum(dim=-1)  # B_A
    # grade = grade.sum(dim=-1)  # B_
    # print(grade, grade.shape)
    # # grade = grade / batch_T_len_count
    # # print(grade, grade.shape)
    # # grade = torch.log(grade)
    # # print(grade, grade.shape)
    #
    # loss = -torch.sum(grade)
    # print(loss)
    # # print(grade)
    #
    # return loss



class SentimentControlLoss(torch.nn.Module):
    def __init__(self, mask_mean, ignore_token=-1):
        super(SentimentControlLoss, self).__init__()
        self.ignore_token = ignore_token
        self.mask_mean = mask_mean
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_token, size_average=True)

    def forward(self, arg_config, input_labels, generation_logits, sentiment_atoms: List[List[List]],
                start_p=1, end_p=None):
        return sentiment_overall_loss(arg_config, input_labels, generation_logits,
                                      sentiment_atoms, self.loss_fct,
                                      padding_value=self.ignore_token, start_p=start_p, end_p=end_p,
                                      mask_mean=self.mask_mean)

class SentimentReinforceControlLoss(torch.nn.Module):
    def __init__(self, ignore_token=-1):
        super(SentimentReinforceControlLoss, self).__init__()
        self.ignore_token = ignore_token
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_token, size_average=True)

    def forward(self, lm_logit_first_index, lm_labels_first_index, input_labels,
                generation_logits, reinforce_control_masks):
        return sentiment_reinforce_control_loss(lm_logit_first_index, lm_labels_first_index,
                                                input_labels, generation_logits,
                                                reinforce_control_masks, self.loss_fct,
                                                padding_value=self.ignore_token)
    # def forward(self,arg_config, input_labels, generation_logits_ori, sentiment_atoms: List[List[List]],
    #             sub_text_seg=7359, padding_value=0, start_p=1, end_p=None):
    #     return sentiment_control_loss(arg_config=arg_config,
    #                                   input_labels=input_labels, generation_logits_ori=generation_logits_ori,
    #                                   sentiment_atoms=sentiment_atoms, sub_text_seg=sub_text_seg,
    #                                   padding_value=padding_value, start_p=start_p, end_p=end_p)