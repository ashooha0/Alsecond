import os

from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import random
import json
from tqdm import trange
import torch
import torch.nn.functional as F
nlp = StanfordCoreNLP(r'E:\stanford-corenlp\stanford-corenlp-4.4.0')
import pickle

def merge_trees(parse_arrs):
    if len(parse_arrs) == 0:
        return []
    if len(parse_arrs) == 1:
        parse_arrs[0][0][1] = -1
        return parse_arrs[0]
    for idx in range(1, len(parse_arrs)):
        past_len = sum([len(ps) for ps in parse_arrs[:idx]])
        for jdx in range(2, len(parse_arrs[idx])):
            parse_arrs[idx][jdx][1] += (past_len - 1)
        parse_arrs[idx] = parse_arrs[idx][1:]
    ret = None
    for idx, arr in enumerate(parse_arrs):
        if idx == 0:
            ret = arr
        else:
            ret += arr
    ret[0][1] = -1
    return ret


def gen1_short(path):
    del_idxes = []
    for idx in range(len(path) - 1):
        if path[idx] == path[idx + 1]:
            del_idxes.append(idx)
    path = [pa for idx, pa in enumerate(path) if idx not in del_idxes]
    return path


def gen2_short(path):
    similar_syntactic_dict = {'JJR': 'JJ', 'JJS': 'JJ',
                              'NNS': 'NN', 'NNP': 'NN', 'NNPS': 'NN', 'CD': 'NN',
                              'RBR': 'RB', 'RBS': 'RB',
                              'VBD': 'VB', 'VBG': 'VB', 'VBN': 'VB', 'VBP': 'VB', 'VBZ': 'VB', 'VV': 'VB',
                              'SBAR': 'S', 'SBARQ': 'S', 'SINU': 'S', 'SQ': 'S'}
    for idx in range(len(path)):
        if path[idx][:-1] in similar_syntactic_dict:
            path[idx] = similar_syntactic_dict[path[idx][:-1]] + path[idx][-1]
    return path


def parse_to_tree(parse):
    parse_arrs = []
    tokens = []
    for single_parse in parse:
        parse_arr = []
        single_parse = single_parse.split()
        last_parent = 0
        for idx, member in enumerate(single_parse):
            if '(' in member:
                parse_arr.append([member[1:], last_parent, ''])
                last_parent = idx
            elif ')' in member:
                right_len = len(member) - member.index(')')
                parse_arr.append(['', last_parent, member[:member.index(')')]])
                for jdx in range(0, right_len):  # 回溯到上一个未遍历完的分叉节点
                    last_parent = parse_arr[last_parent][1]
        parse_arrs.append(parse_arr)
    parse_tree = merge_trees(parse_arrs)
    for idx, node in enumerate(parse_tree):

        if node[2] != '':
            tokens.append((node[2], idx))

    return parse_tree, tokens


def find_node_path(parse_tree, node_idx):  # node_idx : 0 ~ n-1
    paths = []

    parent = parse_tree[node_idx][1]  # 0 ~ n-1

    while parent != -1:
        paths = [parent] + paths
        parent = parse_tree[parent][1]
    return paths


def node_path(parse_tree, tokens, aspect, opinion):
    asp_path = find_node_path(parse_tree, tokens[aspect][1])
    opn_path = find_node_path(parse_tree, tokens[opinion][1])
    latest_parent = 0
    for idx in range(0, min(len(asp_path), len(opn_path))):
        if asp_path[idx] != opn_path[idx]:
            break
        latest_parent = idx
    a2o_path = []
    for idx in range(len(asp_path) - 1, latest_parent, -1):
        a2o_path.append(parse_tree[asp_path[idx]][0] + "↑")
    for idx in range(latest_parent, len(opn_path)):
        #         if idx < len(opn_path)-1:
        a2o_path.append(parse_tree[opn_path[idx]][0] + "↓")
    #         else:
    #             a2o_path.append(parse_tree[opn_path[idx]][0])
    return a2o_path


def merged_string_to_tokens(tokens):
    assert len(tokens) > 0
    string_idx_to_token = []
    cur_idx = 0
    for tk in tokens:
        for letter in tk[0]:
            string_idx_to_token.append(cur_idx)
        cur_idx += 1
    string = tokens[0][0]
    for index in range(1, len(tokens)):
        string = string + tokens[index][0]
    return string.lower(), string_idx_to_token


def find_pos_for_phrase_in_tokens(sentence_str, string2tokens, phrase):
    phrase = phrase.lower()
    phrase = phrase.replace(' ', '')
    token_pos_from_to = []
    step_index = 0
    while (step_index < len(sentence_str)):
        sub_string = sentence_str[step_index:]
        if not phrase in sub_string:
            break
        pri_pos = step_index + sub_string.index(phrase)
        token_pos_from_to.append((string2tokens[pri_pos], string2tokens[pri_pos + len(phrase) - 1] + 1))
        step_index = pri_pos + len(phrase)
    return token_pos_from_to


def find_phrases_shortest_path(parse_tree, tokens, asp_start, asp_end, opn_start, opn_end):
    shortest_dist = 10000
    asp_idx, opn_jdx = asp_start, opn_start
    shortest_path = None
    for idx in range(asp_start, asp_end):
        for jdx in range(opn_start, opn_end):
            sub_path = node_path(parse_tree, tokens, idx, jdx)  # conver 0 ~ n-1 to 1 ~ n
            sub_path = gen1_short(sub_path)
            sub_path = gen2_short(sub_path)
            if len(sub_path) < shortest_dist:
                shortest_dist = len(sub_path)
                shortest_path = sub_path
                asp_idx = idx
                opn_jdx = jdx
    return shortest_path

def remove_dulp(triples):
    del_list = []
    for idx in range(len(triples)-1):
        if idx in del_list:
            continue
        for jdx in range(idx+1, len(triples)):
            if jdx in del_list:
                continue
            if triples[idx][0] == triples[jdx][0]:
                del_list.append(jdx)
    triples = [trip for idx, trip in enumerate(triples) if idx not in del_list]
    del_list = []
    for idx in range(len(triples)-1):
        if idx in del_list:
            continue
        for jdx in range(idx+1, len(triples)):
            if jdx in del_list:
                continue
            if triples[idx][1] == triples[jdx][1]:
                del_list.append(jdx)
    triples = [trip for idx, trip in enumerate(triples) if idx not in del_list]
    return triples

def offset_triples(triples):
    if len(triples) < 2:
        return None
    elif len(triples) == 2:
        triples[0][1], triples[1][1] = triples[1][1], triples[0][1]
        return triples
    else:
        for idx in range(0, len(triples)-1):
            triples[idx][1] = triples[idx+1][1]
        return triples[:-1]

def offset_n_triples(triples, n_offset):  # TODO: 指定偏移的位数 n(无边界，即尾部的会回到头部)， 若 n % len 则返回None意为跳过
    len_triple = len(triples)
    if n_offset % len_triple == 0:
        return None
    opinion_spans = [trip[1] for trip in triples]
    for idx in range(0, len_triple):
        triples[idx][1] = opinion_spans[(idx+n_offset) % len_triple]
    return triples

def random_alloc_triples(triples):  # TODO: 使aspect和opinion重新随机分配（不会分配为原位置）
    if len(triples) < 2:
        return None
    for idx in range(0, len(triples) - 1):
        rand_swap = random.randint(idx+1, len(triples)-1)
        triples[idx][1], triples[rand_swap][1] = triples[rand_swap][1], triples[idx][1]
    return triples

def random_x_triples(triples, x):  # 所有二元组，有x的几率，opinion错位到随机位置
    if len(triples) < 2:
        return None
    for idx in range(0, len(triples)):
        if random.random() < x:
            rand_target = idx
            while rand_target == idx:
                rand_target = random.randint(0, len(triples)-1)
            triples[idx][1] = triples[rand_target][1]
    return triples

def construct_pos_neg_counters(func='', x_random_thres=0., offset_num=1,
                               fn_path='../data/crop/SemEval/'):
    if func == 'offset':
        cached_features_file = fn_path + func + str(offset_num) + '_cache.pk'
    else:
        cached_features_file = fn_path+func+'_cache.pk'
    if os.path.exists(cached_features_file):
        with open(cached_features_file, "rb") as handle:
            (path_count, total_count) = pickle.load(handle)
        return path_count, total_count
    data = json.load(open(fn_path+'train.json'))
    data += json.load(open(fn_path+'test.json'))
    data += json.load(open(fn_path+'dev.json'))
    sentences = [da["sentence"] for da in data]
    triples = [da["triples"] for da in data]
    distences = []
    path_count = {}
    hddava_count = 0
    for idx in trange(len(sentences)):
        parse = nlp._request('pos,parse', sentences[idx])
        parse = [s['parse'] for s in parse['sentences']]
        parse = [s.replace('\r\n', '') for s in parse]

        parse_tree, tokens = parse_to_tree(parse)
        if not len(tokens) == len(sentences[idx].split()):
            continue
        if func == 'random':
            cur_triples = remove_dulp(triples[idx])
        #     cur_triples = offset_n_triples(cur_triples, offset_num)
            cur_triples = random_alloc_triples(cur_triples)
        elif func == 'offset':
            cur_triples = remove_dulp(triples[idx])
            cur_triples = offset_n_triples(cur_triples, offset_num)
        elif func == 'random_x':
            cur_triples = remove_dulp(triples[idx])
            cur_triples = random_x_triples(cur_triples, x_random_thres)
        else:
            cur_triples = triples[idx]
        if cur_triples is None:
            continue
        for jdx in range(len(cur_triples)):
            path_g = find_phrases_shortest_path(parse_tree, tokens, cur_triples[jdx][0][0], cur_triples[jdx][0][1]+1,
                                                cur_triples[jdx][1][0], cur_triples[jdx][1][1]+1)
            if str(path_g) not in path_count:
                path_count[str(path_g)] = 1
            else:
                path_count[str(path_g)] += 1

    keys = [key for key in path_count]
    values = [path_count[key] for key in path_count]
    total_count = sum(values)
    with open(cached_features_file, 'wb') as handle:
        pickle.dump((path_count, total_count), handle, protocol=pickle.HIGHEST_PROTOCOL)

    return path_count, total_count

def compute_KL_div(path_count, total_count, pred_path_count, pred_total_count):
    ori_keys = set([ke for ke in path_count])
    pred_keys = set([ke for ke in pred_path_count])
    union_jot = ori_keys | pred_keys
    x, y = [], []
    for key in union_jot:
        x.append(pred_path_count.get(key,0) / pred_total_count)
        y.append(path_count.get(key,0) / total_count)
    assert len(x) == len(y)
    x = torch.tensor(x).unsqueeze(dim=0)
    y = torch.tensor(y).unsqueeze(dim=0)
#     print(x.shape)
    kl = F.kl_div(x.softmax(dim=-1).log(), y.softmax(dim=-1), reduction='sum')
    return kl

def beyes_prob(path_count, total_count, random_path_count, random_total_count):
    ori_keys = set([ke for ke in path_count])
    offset_keys = set([ke for ke in random_path_count])
    jot = ori_keys & offset_keys
    un_jot = ori_keys ^ offset_keys
#     print(len(jot))
#     print(len(un_jot))
    jot_dict_minus = {}
    jot_count_ori, jot_count_off = 0, 0
    minin_count = 0
    maxmax_count = 0
    ori_num_count, off_num_count = 0, 0
    for key in jot:
        jot_dict_minus[key] = (path_count[key]/total_count)/((random_path_count[key]/random_total_count)+(path_count[key]/total_count))
    return jot_dict_minus

def beyes_result(target_count, target_total_count, path_count, jot_dict_minus):
    beyes_score = 0.
    for key in target_count:
        if key in jot_dict_minus:
            beyes_score += jot_dict_minus[key] * target_count[key]
        elif key in path_count:
            beyes_score += 1. * target_count[key]
    return beyes_score / target_total_count


def count_pathes(sentences, senti_pairs):
    counter = {}
    for idx in trange(len(sentences)):
        parse = nlp._request('pos,parse', sentences[idx])
        parse = [s['parse'] for s in parse['sentences']]
        parse = [s.replace('\r\n', '') for s in parse]
        parse_tree, tokens = parse_to_tree(parse)
        sentence_string, string2tokens = merged_string_to_tokens(tokens)
        for jdx in range(len(senti_pairs[idx])):
            asp_phrase, opn_phrase = senti_pairs[idx][jdx].split(' is ')
            asp_spans = find_pos_for_phrase_in_tokens(sentence_string, string2tokens, asp_phrase)
            opn_spans = find_pos_for_phrase_in_tokens(sentence_string, string2tokens, opn_phrase)
            if len(asp_spans) == 0 or len(opn_spans) == 0:
                continue
            asp_span = asp_spans[0]
            opn_span = opn_spans[0]
            if len(opn_spans) > 1:
                a2o_dist = min(abs(asp_span[0] - opn_span[1]), abs(asp_span[1] - opn_span[0]))
                for kdx in range(1, len(opn_spans)):
                    cur_dist = min(abs(asp_span[0] - opn_spans[kdx][1]), abs(asp_span[1] - opn_spans[kdx][0]))
                    if cur_dist < a2o_dist:
                        a2o_dist = cur_dist
                        opn_span = opn_spans[kdx]

            path_g = find_phrases_shortest_path(parse_tree, tokens, asp_span[0], asp_span[1], opn_span[0], opn_span[1])
            if str(path_g) not in counter:
                counter[str(path_g)] = 1
            else:
                counter[str(path_g)] += 1
    return counter


def load_generated(fn_path):
    data = []
    with open(fn_path + 'sentricon_output.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    # print(data)
    original_sentences = [da["original_input_text"] for da in data]
    sentiment_pairs = [da["original_sentiment_pair"] for da in data]
    sentricon_outputs = [da["self_sentricon_output"] for da in data]
    sc_gpt2_ar_gens = [da["sc_gpt2_ar_gen"] for da in data]

    original_input_path_count = count_pathes(original_sentences, sentiment_pairs)
    senticon_output_path_count = count_pathes(sentricon_outputs, sentiment_pairs)
    gpt_output_path_count = count_pathes(sc_gpt2_ar_gens, sentiment_pairs)
    # print(path_count)
    original_input_keys = [key for key in original_input_path_count]
    original_input_values = [original_input_path_count[key] for key in original_input_path_count]
    original_input_total_count = sum(original_input_values)
    # print(total_count)
    # print(sorted_kv)
    senticon_output_keys = [key for key in senticon_output_path_count]
    senticon_output_values = [senticon_output_path_count[key] for key in senticon_output_path_count]
    senticon_output_total_count = sum(senticon_output_values)

    gpt_output_keys = [key for key in gpt_output_path_count]
    gpt_output_values = [gpt_output_path_count[key] for key in gpt_output_path_count]
    gpt_output_total_count = sum(gpt_output_values)

    return original_input_path_count, original_input_total_count, senticon_output_path_count, senticon_output_total_count, gpt_output_path_count, gpt_output_total_count


def fast_swap(arr_data, compare_idx, mid_num):
    left, right = 0, len(arr_data)-1
    while left < right:
        while left < right and arr_data[right][compare_idx] >= mid_num:
            right -= 1
        while left < right and arr_data[left][compare_idx] <= mid_num:
            left += 1
        if left < right:
            arr_data[left], arr_data[right] = arr_data[right], arr_data[left]
    assert left == right
    return arr_data

def find_closet_number(arr_data, compare_idx, number):
    min_distance = 100
    min_index = 0
    for idx in range(len(arr_data)):
        if abs(arr_data[idx][compare_idx] - number) < min_distance:
            min_distance = abs(arr_data[idx][compare_idx] - number)
            min_index = idx
    return min_index


def get_matching_score(path_count, total_count, random_path_count,
                       random_total_count, jot_dict_minus,
                       original_input_path_count, original_input_total_count,
                       senticon_output_path_count, senticon_output_total_count,
                       gpt_output_path_count, gpt_output_total_count,
                       fn_path='../save_models/'):
    data = []
    with open(fn_path + 'sentricon_output.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    # print(data)
    original_sentences = [da["original_input_text"] for da in data]
    sentiment_pairs = [da["original_sentiment_pair"] for da in data]
    sentricon_outputs = [da["self_sentricon_output"] for da in data]
    sc_gpt2_ar_gens = [da["sc_gpt2_ar_gen"] for da in data]

    ret = {}
    ori_score = 0.
    for key in path_count:
        if key in jot_dict_minus:
            ori_score += jot_dict_minus[key] * path_count[key]
        else:
            ori_score += 1. * path_count[key]
    ret["所有正例得分"] = ori_score / total_count
    # print("所有正例得分:", ori_score / total_count)

    # compute_offset_scores
    offset_score = 0.
    for key in random_path_count:
        if key in jot_dict_minus:
            offset_score += jot_dict_minus[key] * random_path_count[key]
    ret['所有反例得分'] = offset_score / random_total_count
    # print("所有反例得分:", offset_score / random_total_count)

    original_score = 0.
    for key in original_input_path_count:
        if key in jot_dict_minus:
            original_score += jot_dict_minus[key] * original_input_path_count[key]
        elif key in path_count:
            original_score += 1. * original_input_path_count[key]
    ret['原输入得分'] = original_score / original_input_total_count
    # print("原输入得分:", original_score / original_input_total_count)

    senticon_score = 0.
    for key in senticon_output_path_count:
        if key in jot_dict_minus:
            senticon_score += jot_dict_minus[key] * senticon_output_path_count[key]
        elif key in path_count:
            senticon_score += 1. * senticon_output_path_count[key]
    ret['生成输出得分'] = senticon_score / senticon_output_total_count
    # print("生成输出得分:", senticon_score / senticon_output_total_count)

    gpt_score = 0.
    for key in gpt_output_path_count:
        if key in jot_dict_minus:
            gpt_score += jot_dict_minus[key] * gpt_output_path_count[key]
        elif key in path_count:
            gpt_score += 1. * gpt_output_path_count[key]
    ret['gpt2生成输出得分'] = gpt_score / gpt_output_total_count
    # print("gpt2生成输出得分:", gpt_score / gpt_output_total_count)
    return ret

def get_target_matching_score(path_count, jot_dict_minus,
                              target_input_path_count, target_input_total_count,):
    target_score = 0.
    for key in target_input_path_count:
        if key in jot_dict_minus:
            target_score += jot_dict_minus[key] * target_input_path_count[key]
        elif key in path_count:
            target_score += 1. * target_input_path_count[key]
    return target_score / target_input_total_count


def metric_fit(target_input_path_count, target_input_total_count):
    path_count, total_count = construct_pos_neg_counters()
    random_path_count, random_total_count = construct_pos_neg_counters(func='random')
    off1_path_count, off1_total_count = construct_pos_neg_counters(func='offset', offset_num=1)
    off2_path_count, off2_total_count = construct_pos_neg_counters(func='offset', offset_num=2)
    off3_path_count, off3_total_count = construct_pos_neg_counters(func='offset', offset_num=3)
    jot_dict_minus_rand = beyes_prob(path_count, total_count, random_path_count, random_total_count)
    jot_dict_minus_off1 = beyes_prob(path_count, total_count, off1_path_count, off1_total_count)
    jot_dict_minus_off2 = beyes_prob(path_count, total_count, off2_path_count, off2_total_count)
    jot_dict_minus_off3 = beyes_prob(path_count, total_count, off3_path_count, off3_total_count)

    score_rand = get_target_matching_score(path_count, jot_dict_minus_rand,
                                           target_input_path_count, target_input_total_count)
    score_off1 = get_target_matching_score(path_count, jot_dict_minus_off1,
                                           target_input_path_count, target_input_total_count)
    score_off2 = get_target_matching_score(path_count, jot_dict_minus_off2,
                                           target_input_path_count, target_input_total_count)
    score_off3 = get_target_matching_score(path_count, jot_dict_minus_off3,
                                           target_input_path_count, target_input_total_count)
    ret = {}
    ret['score_rand'] = score_rand
    ret['score_off1'] = score_off1
    ret['score_off2'] = score_off2
    ret['score_off3'] = score_off3
    return ret
    # hoipc, hoiptc, hsopc, hsotc, hgopc, hgotc = load_generated('../save_models/')


def fit_aocon(fn_path='../save_models/'):
    data = []
    with open(fn_path + 'sentricon_output.jsonl') as f:
        lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    sentiment_pairs = [da["original_sentiment_pair"] for da in data]
    sentricon_outputs = [da["self_sentricon_output"] for da in data]

    senticon_output_path_count = count_pathes(sentricon_outputs, sentiment_pairs)
    senticon_output_values = [senticon_output_path_count[key] for key in senticon_output_path_count]
    senticon_output_total_count = sum(senticon_output_values)
    print(metric_fit(senticon_output_path_count, senticon_output_total_count))

def fit_t5(fn_path='../baseline_results/'):
    pairs, texts = [], []
    with open(fn_path+"t5/commongen/test.source", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            pairs.append(line.strip().replace('generate a sentence with these concepts: ','').split(', '))
    with open(fn_path+"t5/generated_base/pseudo_trained_20/test.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            texts.append(line)
    t5_output_path_count = count_pathes(texts, pairs)
    t5_output_values = [t5_output_path_count[key] for key in t5_output_path_count]
    t5_output_total_count = sum(t5_output_values)
    print(metric_fit(t5_output_path_count, t5_output_total_count))

if __name__ == '__main__':
    # fit_aocon()
    fit_t5()