import numpy as np
import argparse
import json
import nltk
from nltk import RegexpTokenizer
from tqdm import tqdm, trange
import os
from model.transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    SenTriConBlock,
    HDiscriminator,
)

def tokenize_yelp(fp, word2index_pth, index2word_pth):
    MODEL_CLASSES = {
        "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    }
    parser = argparse.ArgumentParser()
    parser.add_argument(  # 配置参数
        "--config",
        help="path to json config",
        default="../data/yelp_config.json"
    )
    args = parser.parse_args()
    arg_config = json.load(open(args.config, 'r'))   # 加载参数
    stc_data = json.load(open(fp, 'r'))
    stc_data = [line["sentence"] for line in stc_data]
    sem_eval_data = json.load(open("../data/crop/SemEval/train.json"))
    sem_eval_data += json.load(open("../data/crop/SemEval/dev.json"))
    sem_eval_data += json.load(open("../data/crop/SemEval/test.json"))
    sem_eval_data = [line["sentence"] for line in sem_eval_data]
    stc_data = sem_eval_data + stc_data
    # tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    stc_data = [line.split() for line in stc_data]

    word_count = {}
    word2index = {}
    index2word = {}
    glove_token_list, glove_vector_list = [], []
    with open(glove_300_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()[0]
            glove_token_list.append(word)
    word2index['<pad>'] = 0
    index2word[0] = '<pad>'
    word2index['<unk>'] = 1
    index2word[1] = '<unk>'
    cur_index = 2
    for idx in trange(0, len(stc_data)):
        line = stc_data[idx]
        for token in line:
            if token not in word_count:
                word_count[token] = 1
            else:
                word_count[token] += 1
            if word_count[token] > 24 and token not in word2index and token in glove_token_list:
                word2index[token] = cur_index
                index2word[cur_index] = token
                cur_index += 1
    json.dump(word2index, open(word2index_pth, 'w'))
    json.dump(index2word, open(index2word_pth, 'w'))

    # print(stc_data)

def glove_item_filter(word2index_pth, index2word_pth, glove_300_path, output_path):
    word2index = json.load(open(word2index_pth, 'r'))
    embeds = np.zeros((len(word2index), 300))
    with open(glove_300_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        for line in lines:
            word, vector = line.split()[0], line.split()[-300:]
            if word in word2index:
                embeds[word2index[word]] = np.array(vector)
    np.save(output_path, embeds)

if __name__ == "__main__":
    yelp_path = "../data/crop/yelp_academic_dataset_review.json"
    word2index_pth = "../data/crop/word2index.json"
    index2word_pth = "../data/crop/index2word.json"
    glove_300_path = "../classifier/data/doubleembedding/glove.840B.300d.txt"
    # if not (os.path.exists(word2index_pth) and os.path.exists(index2word_pth)):
    tokenize_yelp(yelp_path, word2index_pth, index2word_pth)
    filter_output_path = "../classifier/data/doubleembedding/glove_filtered_300d.npy"
    glove_item_filter(word2index_pth, index2word_pth, glove_300_path, filter_output_path)
    # word2index = json.load(open(word2index_pth, 'r'))
    # index2word = json.load(open(index2word_pth, 'r'))


