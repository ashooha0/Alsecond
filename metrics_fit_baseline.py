import copy
import math
from typing import List
import sacrebleu
import nltk
from rouge import Rouge
import json
from nltk.translate import (nist_score, meteor_score)
from utils.data_util import score_sentiment_p
baseline_rs_dir = "./baseline_results/"
detokenize_punts = {' .':'.',' ,':',',' :':':'," '":"'",' "':'"'," ?":"?"," !":"!"," ;":";"," /":"/"}
# from stanfordcorenlp import StanfordCoreNLP



def detokenize(sentence: str):
    '''
    de-tokenize for example: " ." → "."  |  " ," → ","
    :param sentence:
    :return:
    '''
    for punts in detokenize_punts:
        sentence = sentence.replace(punts, detokenize_punts[punts])
    return sentence


def evaluate_bleu(generated: List, gold: List[List]):
    gold = [gd[0] for gd in gold]
    gold = [gold]
    bleu_score = sacrebleu.corpus_bleu(generated, gold)
    return bleu_score.precisions


def evaluate_self_bleu(generated: List):
    num_gen = len(generated)
    if num_gen == 0:
        return 0., 0., 0., 0., 0.
    score_list = []
    uni_list, bi_list, tri_list, qua_list = [], [], [], []
    for i in range(num_gen):
        refs = [[generated[j]] for j in range(num_gen) if j != i]
        sys = [generated[i]]
        blue = sacrebleu.corpus_bleu(sys, refs)  # calcuate score
        uni_list.append(blue.precisions[0])
        bi_list.append(blue.precisions[1])
        tri_list.append(blue.precisions[2])
        qua_list.append(blue.precisions[3])
        score_list.append(blue.score)  # save the score
        # show each score (Comment out if not needed)
    ret_uni = sum(uni_list) / num_gen
    ret_bi = sum(bi_list) / num_gen
    ret_tri = sum(tri_list) / num_gen
    ret_qua = sum(qua_list) / num_gen
    ret_score = sum(score_list) / num_gen
    return ret_uni, ret_bi, ret_tri, ret_qua, ret_score


def evaluate_rouge(generated: List, gold: List[List]):
    gold = [gd[0] for gd in gold]
    # rouge = RougeAugmented(max_n=4)
    rouge = Rouge()
    return rouge.get_scores(generated, gold, avg=True)["rouge-l"]


def evaluate_meteor(generated: List, gold: List[List]):
    meteor_ret = [meteor_score.meteor_score(gd, stc) for gd, stc in zip(gold, generated)]
    return sum(meteor_ret) / len(meteor_ret)


def evaluate_nist(generated: List, gold: List[List]):
    gold_lists = [[nltk.word_tokenize(gd[0])] for gd in gold]
    # generated_lists = [nltk.word_tokenize(gd[0]) for gd in gold]
    generated_lists = [nltk.word_tokenize(gl) for gl in generated]
    nist_ret = nist_score.corpus_nist(gold_lists, generated_lists, n=5)
    return nist_ret


def evaluate_dist(generated: List):
    token_num = 0
    unigrams, bigrams, trigrams = {}, {}, {}
    for gen_txt in generated:
        sentence_tokens = nltk.word_tokenize(gen_txt)
        token_num += len(sentence_tokens)
        for idx in range(0, len(sentence_tokens)):
            unigrm = sentence_tokens[idx]
            if unigrm not in unigrams:
                unigrams[unigrm] = 1
            else:
                unigrams[unigrm] += 1
            if idx + 1 < len(sentence_tokens):
                bigrm = sentence_tokens[idx] + ' ' + sentence_tokens[idx + 1]
                if bigrm not in bigrams:
                    bigrams[bigrm] = 1
                else:
                    bigrams[bigrm] += 1
            if idx + 2 < len(sentence_tokens):
                trigrm = sentence_tokens[idx] + ' ' + sentence_tokens[idx + 1] + ' ' + sentence_tokens[idx + 2]
                if trigrm not in trigrams:
                    trigrams[trigrm] = 1
                else:
                    trigrams[trigrm] += 1
    return len(unigrams) / token_num, len(bigrams) / token_num, len(trigrams) / token_num,


def evaluate_coverage(generated: List, senti_units: List[List], split_word=" is "):

    return score_sentiment_p(generated, senti_units, split_word)


def evaluate_matching():
    # 默认是英文
    # nlp = StanfordCoreNLP(r'E:\stanford-corenlp\stanford-corenlp-4.4.0')

    pass


def fit_aocon(cut_prompt=False, tail_pad='<|endoftext|>'):
    source_dir = 'baseline_results/ASIN/generate_samehint/'
    # source_dir = 'save_models/'
    generated, gold_sentences, gpt2_gened, senti_units = [], [], [], []
    with open(source_dir+'sentricon_output.jsonl', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text']
            if cut_prompt:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")[
                           len(history_input):].lstrip()
                sentricon_str = js_data['self_sentricon_output'][len(history_input):].lstrip()
                gpt2_str = js_data['sc_gpt2_ar_gen'][len(history_input):].lstrip()

            else:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")
                sentricon_str = js_data['self_sentricon_output']
                gpt2_str = js_data['sc_gpt2_ar_gen']
            senti_units.append(js_data['original_sentiment_pair'])
            gold_sentences.append([gold_str])
            gpt2_gened.append(gpt2_str)
            generated.append(sentricon_str)
    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(generated, gold_sentences)
    rouge_l = evaluate_rouge(generated, gold_sentences)
    meteor_v = evaluate_meteor(generated, gold_sentences)
    cov = evaluate_coverage(generated, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(generated)
    dist1, dist2, dist3 = evaluate_dist(generated)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}

def fit_gpt2base(cut_prompt=False, tail_pad='<|endoftext|>'):
    source_dir = baseline_rs_dir+'gpt2base/'
    gold_sentences, gpt2_gened, senti_units = [], [], []
    with open(source_dir+'sentricon_output.jsonl', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text']
            if cut_prompt:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")[
                           len(history_input):].lstrip()
                # sentricon_str = js_data['self_sentricon_output'][len(history_input):].lstrip()
                gpt2_str = js_data['sc_gpt2_ar_gen'][len(history_input):].lstrip()

            else:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")
                # sentricon_str = js_data['self_sentricon_output']
                gpt2_str = js_data['sc_gpt2_ar_gen']
            senti_units.append(js_data['original_sentiment_pair'])
            gold_sentences.append([gold_str])
            gpt2_gened.append(gpt2_str)
            # generated.append(sentricon_str)
    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(gpt2_gened, gold_sentences)
    rouge_l = evaluate_rouge(gpt2_gened, gold_sentences)
    meteor_v = evaluate_meteor(gpt2_gened, gold_sentences)
    cov = evaluate_coverage(gpt2_gened, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(gpt2_gened)
    dist1, dist2, dist3 = evaluate_dist(gpt2_gened)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}

def fit_pplm(cut_prompt=False, tail_pad='<|endoftext|>'):
    source_dir = baseline_rs_dir+'pplm/'
    gold_sentences, pplm_gened, senti_units = [], [], []
    with open(source_dir+'sentricon_output.jsonl', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text']
            if cut_prompt:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")[
                           len(history_input):].lstrip()
                # sentricon_str = js_data['self_sentricon_output'][len(history_input):].lstrip()
                gened_str = js_data['pplm_out'][len(history_input):].lstrip()
            else:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")
                # sentricon_str = js_data['self_sentricon_output']
                gened_str = js_data['pplm_out']
            senti_units.append(js_data['original_sentiment_pair'])
            gold_sentences.append([gold_str])
            pplm_gened.append(gened_str)
            # generated.append(sentricon_str)
    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(pplm_gened, gold_sentences)
    rouge_l = evaluate_rouge(pplm_gened, gold_sentences)
    meteor_v = evaluate_meteor(pplm_gened, gold_sentences)
    cov = evaluate_coverage(pplm_gened, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(pplm_gened)
    dist1, dist2, dist3 = evaluate_dist(pplm_gened)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}

def fit_gpt2app(cut_prompt=False, tail_pad='<|endoftext|>'):
    source_dir = baseline_rs_dir+'gpt2app/'
    gold_sentences, gpt2_gened, senti_units = [], [], []
    with open(source_dir+'sentricon_output.jsonl', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text']
            if cut_prompt:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")[
                           len(history_input):].lstrip()
                # sentricon_str = js_data['self_sentricon_output'][len(history_input):].lstrip()
                gpt2_str = js_data['sc_gpt2_ar_gen'][len(history_input):].lstrip()

            else:
                gold_str = js_data['original_input_text'].replace(tail_pad, "")
                # sentricon_str = js_data['self_sentricon_output']
                gpt2_str = js_data['sc_gpt2_ar_gen']
            senti_units.append(js_data['original_sentiment_pair'])
            gold_sentences.append([gold_str])
            gpt2_gened.append(gpt2_str)
            # generated.append(sentricon_str)
    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(gpt2_gened, gold_sentences)
    rouge_l = evaluate_rouge(gpt2_gened, gold_sentences)
    meteor_v = evaluate_meteor(gpt2_gened, gold_sentences)
    cov = evaluate_coverage(gpt2_gened, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(gpt2_gened)
    dist1, dist2, dist3 = evaluate_dist(gpt2_gened)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}

def fit_t5():
    # source_dir = baseline_rs_dir+'t5/commongen_pairs/'
    source_dir = baseline_rs_dir + 't5/commongen/'
    # target_dir = baseline_rs_dir+'t5/generated_base/pairs_20/'
    target_dir = baseline_rs_dir + 't5/generated_base/pseudo_trained_20/'
    generated, gold_sentences, senti_units = [], [], []
    with open(source_dir+'test.source', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            senti_unit = line[len("generate a sentence with these concepts: "):].rstrip('\n').split(', ')
            senti_units.append(senti_unit)
    with open(source_dir+'test.target', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            gold_sentences.append([detokenize(line.rstrip('\n'))])
    with open(target_dir+'test.txt', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            txt = line.rstrip('\n')
            if txt == '' or len(txt) == 0:
                txt = "'"
            txt = detokenize(txt)
            generated.append(txt)

    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(generated, gold_sentences)
    rouge_l = evaluate_rouge(generated, gold_sentences)
    meteor_v = evaluate_meteor(generated, gold_sentences)
    cov = evaluate_coverage(generated, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(generated)
    dist1, dist2, dist3 = evaluate_dist(generated)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}


def fit_bertgen():
    source_dir = baseline_rs_dir+'bertgen/commongen/'
    target_dir = baseline_rs_dir+'bertgen/generated/pseudo_trained/50/'
    generated, gold_sentences, senti_units = [], [], []
    with open(source_dir+'commongen.test.json', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_d = json.loads(line)
            senti_unit = js_d['src'].split(', ')
            senti_units.append(senti_unit)
            gold_sentences.append([detokenize(js_d['tgt'])])

    with open(target_dir+'test.txt', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            generated.append(detokenize(line.rstrip('\n')))

    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(generated, gold_sentences)
    rouge_l = evaluate_rouge(generated, gold_sentences)
    meteor_v = evaluate_meteor(generated, gold_sentences)
    cov = evaluate_coverage(generated, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(generated)
    dist1, dist2, dist3 = evaluate_dist(generated)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}



def fit_unilmv2(pairs=False):
    if pairs:
        source_dir = baseline_rs_dir + 'unilmv2/commongen_pairs/'
        target_dir = baseline_rs_dir + 'unilmv2/generated_pairs/pseudo_trained/50/'
    else:
        source_dir = baseline_rs_dir+'unilmv2/commongen/'
        target_dir = baseline_rs_dir+'unilmv2/generated/pseudo_trained/2_70/'
    generated, gold_sentences, senti_units = [], [], []
    with open(source_dir+'commongen.test.json', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_d = json.loads(line)
            senti_unit = js_d['src'].split(', ')
            senti_units.append(senti_unit)
            gold_sentences.append([detokenize(js_d['tgt'])])

    with open(target_dir+'test.txt', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            generated.append(detokenize(line.rstrip('\n')))

    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(generated, gold_sentences)
    rouge_l = evaluate_rouge(generated, gold_sentences)
    meteor_v = evaluate_meteor(generated, gold_sentences)
    cov = evaluate_coverage(generated, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(generated)
    dist1, dist2, dist3 = evaluate_dist(generated)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}


def fit_unilm():
    source_dir = baseline_rs_dir+'unilm/commongen/'
    target_dir = baseline_rs_dir+'unilm/generated/pseudo_trained/2_100/'
    generated, gold_sentences, senti_units = [], [], []
    with open(source_dir+'commongen.test.json', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            js_d = json.loads(line)
            senti_unit = js_d['src'].split(', ')
            senti_units.append(senti_unit)
            gold_sentences.append([detokenize(js_d['tgt'])])

    with open(target_dir+'test.txt', 'r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            txt = line.rstrip('\n')
            if txt == '' or len(txt) == 0:
                txt = "'"
            txt = detokenize(txt)
            generated.append(txt)

    bleu_1, bleu_2, bleu_3, bleu_4 = evaluate_bleu(generated, gold_sentences)
    rouge_l = evaluate_rouge(generated, gold_sentences)
    meteor_v = evaluate_meteor(generated, gold_sentences)
    cov = evaluate_coverage(generated, senti_units)
    # nist = evaluate_nist(generated, senti_units)
    self_b_uni, self_b_bi, self_b_tri, self_b_qua, self_b_score = evaluate_self_bleu(generated)
    dist1, dist2, dist3 = evaluate_dist(generated)
    return {"bleu1": bleu_1, "bleu2": bleu_2, "bleu3": bleu_3, "bleu4": bleu_4,
            "dist1": dist1, "dist2": dist2, "dist3": dist3,
            "self_bleu_1": self_b_uni, "self_bleu_2": self_b_bi, "self_bleu_3": self_b_tri,
            "self_bleu_4": self_b_qua, "self_bleu_score": self_b_score,
            "rouge_l": rouge_l, "meteor_v": meteor_v, "cov": cov}


def fit_k2t():
    pass

if __name__ == '__main__':
    print({"t5": fit_t5()})
    print({"bertgen": fit_bertgen()})
    print({"unilmv2": fit_unilmv2()})
    print({"unilmv2-pairs": fit_unilmv2(pairs=True)})
    print({"gpt2app": fit_gpt2app()})
    print({"aocon": fit_aocon()})