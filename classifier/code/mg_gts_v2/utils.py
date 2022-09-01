import multiprocessing
import pickle
import numpy as np
import sklearn

id2sentiment = {3: 'NEG', 4: 'NEU', 5: 'POS'}


def get_aspects(tags, length, ignore_index=-1, pred_scores=None):
    spans, span_scores = [], []
    start = -1
    token_score = []
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 1:
            if start == -1:
                start = i
            if pred_scores is not None:
                token_score.append(pred_scores[i][i])
        elif tags[i][i] != 1:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
                if pred_scores is not None:
                    span_scores.append(token_score)
                    token_score = []
    if start != -1:
        spans.append([start, length-1])
        if pred_scores is not None:
            span_scores.append(token_score)
    if pred_scores is not None:
        assert len(spans) == len(span_scores)
        return spans, span_scores
    return spans


def get_aspects_ground_scores(predicted_aspect_spans):
    span_scores = []
    token_score = []
    for aspen in predicted_aspect_spans:
        al, ar = aspen
        for indx in range(al, ar+1):
            token_score.append(1.0)
        span_scores.append(token_score)
        token_score = []
    return span_scores


def get_opinions(tags, length, ignore_index=-1, pred_scores=None):
    spans, span_scores = [], []
    start = -1
    token_score = []
    for i in range(length):
        if tags[i][i] == ignore_index: continue
        elif tags[i][i] == 2:
            if start == -1:
                start = i
            if pred_scores is not None:
                token_score.append(pred_scores[i][i])
        elif tags[i][i] != 2:
            if start != -1:
                spans.append([start, i - 1])
                start = -1
                if pred_scores is not None:
                    span_scores.append(token_score)
                    token_score = []
    if start != -1:
        spans.append([start, length-1])
        if pred_scores is not None:
            span_scores.append(token_score)
    if pred_scores is not None:
        assert len(spans) == len(span_scores)
        return spans, span_scores
    return spans

def score_aspect(predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predict_set = set()
    for i in range(len(golden)):
        golden_spans = get_aspects(golden[i], lengths[i], ignore_index)
        for l, r in golden_spans:
            golden_set.add('-'.join([str(i), str(l), str(r)]))

        predict_spans = get_aspects(predicted[i], lengths[i], ignore_index)
        for l, r in predict_spans:
            predict_set.add('-'.join([str(i), str(l), str(r)]))

    correct_num = len(golden_set & predict_set)
    precision = correct_num / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def score_opinion(predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predict_set = set()
    for i in range(len(golden)):
        golden_spans = get_opinions(golden[i], lengths[i], ignore_index)
        for l, r in golden_spans:
            golden_set.add('-'.join([str(i), str(l), str(r)]))

        predict_spans = get_opinions(predicted[i], lengths[i], ignore_index)
        for l, r in predict_spans:
            predict_set.add('-'.join([str(i), str(l), str(r)]))

    correct_num = len(golden_set & predict_set)
    precision = correct_num / len(predict_set) if len(predict_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def find_pair(tags, aspect_spans, opinion_spans):
    pairs = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            flag = False
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if tags[i][j] == 3 or tags[j][i] == 3:
                        flag = True
                        break
                if flag: break
            if flag:
                pairs.append([al, ar, pl, pr])
    return pairs


def find_pair_with_score(tags, aspect_spans, opinion_spans, asp_span_scores, opn_span_scores):
    pairs = []
    for asp, a_sc in zip(aspect_spans, asp_span_scores):
        al, ar = asp
        for opn, o_sc in zip(opinion_spans, opn_span_scores):
            pl, pr = opn
            flag = False
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if tags[i][j] == 3 or tags[j][i] == 3:
                        flag = True
                        break
                if flag: break
            if flag:
                pairs.append([[al, ar], [pl, pr], a_sc, o_sc])
    return pairs


def find_triplet(tags, aspect_spans, opinion_spans):
    triplets = []
    for al, ar in aspect_spans:
        for pl, pr in opinion_spans:
            tag_num = [0]*6
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1
            if sum(tag_num[3:]) == 0: continue
            sentiment = -1
            if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                sentiment = 5
            elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                sentiment = 4
            elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                sentiment = 3
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                input()
            triplets.append([al, ar, pl, pr, sentiment])
    return triplets

def find_triplet_with_score(tags, aspect_spans, opinion_spans,
                            asp_span_scores, opn_span_scores, to_word=False):
    triplets = []
    polarity_step=0
    for asp, a_sc in zip(aspect_spans, asp_span_scores):
        al, ar = asp
        asp_allocated = False
        for opn, o_sc in zip(opinion_spans, opn_span_scores):
            pl, pr = opn
            tag_num = [0]*6
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1
            if sum(tag_num[3:]) == 0: continue
            sentiment = -1
            if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                sentiment = 5
            elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                sentiment = 4
            elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                sentiment = 3
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                input()
            if to_word:
                triplets.append([[al, ar], [pl, pr], id2sentiment[sentiment], a_sc, o_sc])
            else:
                triplets.append([[al, ar], [pl, pr], sentiment, a_sc, o_sc])
    return triplets


def soft_find_triplet_by_asp_pol_with_score(tags, aspect_spans, opinion_spans,
                                       asp_span_scores, opn_span_scores,
                                       asp_polarity, to_word=False, allocate_default=True):
    triplets = []
    match_num = 0
    total_triple_num = 0
    for polarity_step, (asp, a_sc) in enumerate(zip(aspect_spans, asp_span_scores)):
        al, ar = asp
        asp_allocated = False
        for opn, o_sc in zip(opinion_spans, opn_span_scores):
            pl, pr = opn
            tag_num = [0]*6
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1
            if sum(tag_num[3:]) == 0: continue
            sentiment = -1
            if tag_num[5] >= tag_num[4] and tag_num[5] >= tag_num[3]:
                sentiment = 5
            elif tag_num[4] >= tag_num[3] and tag_num[4] >= tag_num[5]:
                sentiment = 4
            elif tag_num[3] >= tag_num[5] and tag_num[3] >= tag_num[4]:
                sentiment = 3
            if sentiment == -1:
                print('wrong!!!!!!!!!!!!!!!!!!!!')
                input()
            # if to_word:
            #     triplets.append([[al, ar], [pl, pr], id2sentiment[sentiment], a_sc, o_sc])
            # else:
            #     triplets.append([[al, ar], [pl, pr], sentiment, a_sc, o_sc])
            if to_word:
                triplets.append([[al, ar], [pl, pr], id2sentiment[asp_polarity[polarity_step]], a_sc, o_sc])
            else:
                triplets.append([[al, ar], [pl, pr], asp_polarity[polarity_step], a_sc, o_sc])
            asp_allocated = True
            match_num += 1
            total_triple_num += 1
        if allocate_default and not asp_allocated:  # asp需要分配，但最终没有分配的到，则默认分配它自己
            if to_word:
                triplets.append([[al, ar], [al, ar], id2sentiment[asp_polarity[polarity_step]], a_sc, a_sc])
            else:
                triplets.append([[al, ar], [al, ar], asp_polarity[polarity_step], a_sc, a_sc])
            total_triple_num += 1

    return triplets, match_num, total_triple_num


def hard_find_triplet_by_asp_pol_with_score(tags, aspect_spans, opinion_spans,
                                       asp_span_scores, opn_span_scores,
                                       asp_polarity, to_word=False, allocate_default=True):
    triplets = []
    match_num = 0
    total_triple_num = 0
    for polarity_step, (asp, a_sc) in enumerate(zip(aspect_spans, asp_span_scores)):
        al, ar = asp
        asp_allocated = False
        max_vote = 0
        cur_triple = None
        for opn, o_sc in zip(opinion_spans, opn_span_scores):
            pl, pr = opn
            tag_num = [0]*6
            senator_num = (ar - al + 1) * (pr - pl + 1)
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if al < pl:
                        tag_num[int(tags[i][j])] += 1
                    else:
                        tag_num[int(tags[j][i])] += 1
            if sum(tag_num[3:]) == 0: continue
            sentiment = asp_polarity[polarity_step]
            cur_vote = tag_num[sentiment] / senator_num
            if cur_vote > max_vote:
                if to_word:
                    cur_triple = [[al, ar], [pl, pr], id2sentiment[sentiment], a_sc, o_sc]
                else:
                    cur_triple = [[al, ar], [pl, pr], sentiment, a_sc, o_sc]
                max_vote = cur_vote
        if max_vote > 0. and cur_triple is not None:
            triplets.append(cur_triple)
            asp_allocated = True
            match_num += 1
            total_triple_num += 1
        if allocate_default and not asp_allocated:  # asp需要分配，但最终没有分配的到，则默认分配它自己
            if to_word:
                triplets.append([[al, ar], [al, ar], id2sentiment[asp_polarity[polarity_step]], a_sc, a_sc])
            else:
                triplets.append([[al, ar], [al, ar], asp_polarity[polarity_step], a_sc, a_sc])
            total_triple_num += 1

    return triplets, match_num, total_triple_num

def score_pseudo_labels(c_args, pred_scores, predicted, lengths, ignore_index=-1):
    assert len(predicted) == len(pred_scores)  # B_S_S
    result_tuples = []
    for i in range(len(predicted)):
        predicted_aspect_spans, asp_span_scores = get_aspects(predicted[i], lengths[i], ignore_index, pred_scores[i])
        predicted_opinion_spans, opn_span_scores = get_opinions(predicted[i], lengths[i], ignore_index, pred_scores[i])
        if c_args["task"] == 'pair':
            predicted_tuple = find_pair_with_score(predicted[i],
                                                   predicted_aspect_spans, predicted_opinion_spans,
                                                   asp_span_scores, opn_span_scores)
        elif c_args["task"] == 'triplet':
            predicted_tuple = find_triplet_with_score(predicted[i],
                                                      predicted_aspect_spans, predicted_opinion_spans,
                                                      asp_span_scores, opn_span_scores, to_word=True)

        else:
            raise KeyError
        result_tuples.append(predicted_tuple)

    return result_tuples


def score_pseudo_opinions(c_args, pred_scores, predicted, lengths, predicted_aspect_spans,
                          asp_polarities, ignore_index=-1):
    assert len(predicted) == len(pred_scores)  # B_S_S
    result_tuples = []
    match_nums, total_triple_nums = 0, 0
    for i in range(len(predicted)):
        predicted_aspect_span = predicted_aspect_spans[i]
        asp_polarity = asp_polarities[i]
        asp_span_scores = get_aspects_ground_scores(predicted_aspect_span)
        predicted_opinion_spans, opn_span_scores = get_opinions(predicted[i], lengths[i],
                                                                ignore_index, pred_scores[i])
        if c_args["task"] == 'pair':
            predicted_tuple = find_pair_with_score(predicted[i],
                                                   predicted_aspect_span, predicted_opinion_spans,
                                                   asp_span_scores, opn_span_scores)
        elif c_args["task"] == 'triplet':
            predicted_tuple, match_num, total_triple_num = soft_find_triplet_by_asp_pol_with_score(predicted[i],
                                                                 predicted_aspect_span, predicted_opinion_spans,
                                                                 asp_span_scores, opn_span_scores,
                                                                 asp_polarity, to_word=True)
            match_nums += match_num
            total_triple_nums += total_triple_num
        else:
            raise KeyError
        result_tuples.append(predicted_tuple)

    return result_tuples, match_nums, total_triple_nums

def score_uniontags(c_args, predicted, golden, lengths, ignore_index=-1):
    assert len(predicted) == len(golden)
    golden_set = set()
    predicted_set = set()
    for i in range(len(golden)):
        golden_aspect_spans = get_aspects(golden[i], lengths[i], ignore_index)
        golden_opinion_spans = get_opinions(golden[i], lengths[i], ignore_index)
        if c_args["task"] == 'pair':
            golden_tuple = find_pair(golden[i], golden_aspect_spans, golden_opinion_spans)
        elif c_args["task"] == 'triplet':
            golden_tuple = find_triplet(golden[i], golden_aspect_spans, golden_opinion_spans)
        for pair in golden_tuple:
            golden_set.add(str(i) + '-'+ '-'.join(map(str, pair)))

        predicted_aspect_spans = get_aspects(predicted[i], lengths[i], ignore_index)
        predicted_opinion_spans = get_opinions(predicted[i], lengths[i], ignore_index)
        if c_args["task"] == 'pair':
            predicted_tuple = find_pair(predicted[i], predicted_aspect_spans, predicted_opinion_spans)
        elif c_args["task"] == 'triplet':
            predicted_tuple = find_triplet(predicted[i], predicted_aspect_spans, predicted_opinion_spans)
        for pair in predicted_tuple:
            predicted_set.add(str(i) + '-' + '-'.join(map(str, pair)))

    correct_num = len(golden_set & predicted_set)
    precision = correct_num / len(predicted_set) if len(predicted_set) > 0 else 0
    recall = correct_num / len(golden_set) if len(golden_set) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1
