"""
# 统计POS标记
import json 
import stanza 

pos_idx = {}

def processFiles(dataset, mode):
    global count
    sentence_packs = json.load(open(dataset + '/' + mode + '.json'))
    for sentence_pack in sentence_packs:
        # print(sentence_pack)
        sentence = sentence_pack['sentence']
        pos_tags = sentence_pack['pos_tags']
        # print(sentence, pos_tags)

        for i in range(len(pos_tags)):
            # print(pos_tags[i])
            if pos_tags[i] not in pos_idx:
                pos_idx[pos_tags[i]] = count
                count += 1

if __name__ == "__main__":
    count = 2

    # data preprocessing
    processFiles("res14", "train")
    processFiles("res14", "dev")
    processFiles("res14", "test")

    processFiles("res15", "train")
    processFiles("res15", "dev")
    processFiles("res15", "test")

    processFiles("res16", "train")
    processFiles("res16", "dev")
    processFiles("res16", "test")

    processFiles("lap14", "train")
    processFiles("lap14", "dev")
    processFiles("lap14", "test")

    file = open("pos_idx.json",'w')
    file.write(json.dumps(pos_idx))
    file.close()
    # print(pos_idx)
"""

import json
import numpy as np
from mittens import GloVe

class Instance(object):
    def __init__(self, sentence_pack):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.pos_tags = sentence_pack['pos_tags']
        self.heads = sentence_pack['heads']
        self.triples = sentence_pack['triples']

def loadFile():
    res14_sentence_packs = json.load(open('res14/train.json'))
    res15_sentence_packs = json.load(open('res15/train.json'))
    res16_sentence_packs = json.load(open('res16/train.json'))
    lap14_sentence_packs = json.load(open('lap14/train.json'))

    instances = list()
    for sentence_pack in res14_sentence_packs:
        instances.append(Instance(sentence_pack))
    for sentence_pack in res15_sentence_packs:
        instances.append(Instance(sentence_pack))
    for sentence_pack in res16_sentence_packs:
        instances.append(Instance(sentence_pack))
    for sentence_pack in lap14_sentence_packs:
        instances.append(Instance(sentence_pack))
    return instances

def get_cor_matrix(dataset, cor_matrix, pos2index, window=3):
    for sentence_pack in dataset:
        sentence = sentence_pack.sentence
        pos_tags = sentence_pack.pos_tags
        length = len(pos_tags)
        for i in range(length):
            main_pos = pos_tags[i]
            j = i + window if i + window < length else length
            for k in range(i+1, j):
                sub_pos = pos_tags[k]
                cor_matrix[pos2index[main_pos]][pos2index[sub_pos]] += 1
                cor_matrix[pos2index[sub_pos]][pos2index[main_pos]] += 1
    return cor_matrix

if __name__=='__main__':
    window = 2
    pos2index = json.load(open('doubleembedding/pos_idx.json'))
    dataset = loadFile()

    cor_matrix = np.zeros((2+len(pos2index), 2+len(pos2index)), int)
    cor_matrix = get_cor_matrix(dataset, cor_matrix, pos2index, window)

    glove_model = GloVe(n=25, max_iter=100000)
    pos_embeddings = glove_model.fit(cor_matrix)
    np.save('doubleembedding/pos_embeddings_window2.vec.npy', pos_embeddings)
