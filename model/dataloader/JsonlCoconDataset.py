import copy
import os
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from model.pretrained_model.tokenization_utils import PreTrainedTokenizer
import pickle
import logging
import json
from tqdm import tqdm, trange
import random
import torch
from model.transformers import GPT2Tokenizer
logger = logging.getLogger(__name__)

class JsonlCoconDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, arg_config, file_path: str, cs_len, hs_len, tis_len,
                 block_size=None, text_json_key="sentence", text_triple_key="triples", pseudo_label=False, # TODO👈 20220219
                 evaluate=False,
                 use_labeled_data=True, prepended_text_to_remove=None, retrieve_opinion=False):
        print(file_path)
        assert os.path.isfile(file_path)

        self.cs_len = cs_len
        self.hs_len = hs_len
        self.tis_len = tis_len

        if block_size is None:
            block_size = hs_len + max(cs_len, tis_len)
        self.block_size = block_size
        self.pseudo_label = pseudo_label
        directory, filename = os.path.split(file_path)
        if evaluate and text_json_key != 'sentence':
            cached_features_file = os.path.join(
                directory,
                arg_config["model"]["model_type"] + "_cached_"+text_triple_key+"_" + str(block_size) + text_json_key + "_" + filename
            )
        else:
            cached_features_file = os.path.join(
                directory,
                arg_config["model"]["model_type"] + "_cached_"+text_triple_key+"_" + str(block_size) + "_" + filename
            )

        if os.path.exists(cached_features_file) and not arg_config["data"]["overwrite_cache"]:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                if pseudo_label:
                    (self.examples, self.sentiment_examples, self.triple_examples, self.hint_matrixes_a, self.hint_matrixes_o,self.posi_senti_masks, self.pseudo_confidences) = pickle.load(handle)
                else:
                    (self.examples, self.sentiment_examples, self.triple_examples, self.hint_matrixes_a, self.hint_matrixes_o,self.posi_senti_masks) = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            if prepended_text_to_remove is not None:
                if ';' in prepended_text_to_remove:
                    prepended_texts = prepended_text_to_remove.split(';')
                    logger.info("prepended_texts: {}".format(prepended_texts))
                else:
                    prepended_texts = [prepended_text_to_remove]
            else:
                prepended_texts = None

            lines = []
            sentiment_units = []
            aspects, opinions, polarities = [], [], []
            a_confidence, o_confidence = [], []
            posi_senti_masks = []
            hint_matrixes_a = []
            hint_matrixes_o = []
            if retrieve_opinion:
                sem_triple_dict = get_triple_dict(arg_config["data"]["retrieve_opn_sem_fn"])
                mams_triple_dict = get_triple_dict(arg_config["data"]["retrieve_opn_mams_fn"])
                retrieve_opn_triple_dicts = (sem_triple_dict, mams_triple_dict)
            else:
                retrieve_opn_triple_dicts = None
            with open(file_path, encoding="utf-8") as f:
                json_list = json.load(f)
                for json_dict in tqdm(json_list):
                    # json_dict = json.loads(jsonl)
                    line = json_dict[text_json_key]
                    line_splitted = line.split()
                    if pseudo_label:
                        triples = json_dict['triples']
                    else:
                        triples = json_dict[text_triple_key]
                    if pseudo_label:
                        processed_line,sentiment_unit,aspect,opinion,polarity,hint_mtrx_a,hint_mtrx_o, posi_senti_mask,a_cfdc,o_cfdc = process_triples(
                                                                line,
                                                                triples,
                                                                arg_config,
                                                                tokenizer=tokenizer,
                                                                end_punc=
                                                                arg_config["end_segment_id"],
                                                                tuple=text_triple_key,
                                                                pseudo_label=True)
                        a_confidence.append(a_cfdc)
                        o_confidence.append(o_cfdc)
                    else:
                        processed_line, sentiment_unit, aspect, opinion, polarity,hint_mtrx_a,hint_mtrx_o, posi_senti_mask = process_triples(
                                                                line,
                                                                triples,
                                                                arg_config,
                                                                tokenizer=tokenizer,
                                                                end_punc=arg_config["end_segment_id"],
                                                                tuple=text_triple_key,
                                                                retrieve_opn_triple_dicts=retrieve_opn_triple_dicts)

                    lines.append(processed_line)  # B_S'
                    sentiment_units.append(sentiment_unit)  # sentiment_unit: B_A'_T'
                    aspects.append(aspect)  # aspects, opinions, polarities: B_A'_T'
                    opinions.append(opinion)
                    polarities.append(polarity)
                    posi_senti_masks.append(posi_senti_mask)
                    hint_matrixes_a.append(hint_mtrx_a)
                    hint_matrixes_o.append(hint_mtrx_o)

            logger.info("Encoding with tokenizer")
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=None)["input_ids"]
            sentiment_triples = []
            self.sentiment_examples = [tokenizer.batch_encode_plus(sentiment_unit,
                                       add_special_tokens=True,
                                       max_length=None)["input_ids"] for sentiment_unit in sentiment_units]

            aspect_examples = [tokenizer.batch_encode_plus(aspect,  # B_A'_T'
                               add_special_tokens=True,
                               max_length=None)["input_ids"] for aspect in aspects]
            opinion_examples = [tokenizer.batch_encode_plus(opinion,
                                add_special_tokens=True,
                                max_length=None)["input_ids"] for opinion in opinions]

            assert len(aspect_examples) == len(opinion_examples) == len(polarities)
            self.triple_examples = [tri for tri in zip(aspect_examples, opinion_examples, polarities)]
            logger.info("Saving features into cached file %s", cached_features_file)
            self.hint_matrixes_a = hint_matrixes_a
            self.hint_matrixes_o = hint_matrixes_o
            self.posi_senti_masks = posi_senti_masks
            if pseudo_label:
                self.pseudo_confidences = [tu for tu in zip(a_confidence, o_confidence)]
                assert len(self.examples)==len(self.triple_examples)==len(self.sentiment_examples)==len(self.pseudo_confidences)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump((self.examples, self.sentiment_examples, self.triple_examples,
                                 self.hint_matrixes_a, self.hint_matrixes_o,self.posi_senti_masks,
                                 self.pseudo_confidences),
                                handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                assert len(self.examples) == len(self.triple_examples) == len(self.sentiment_examples)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump((self.examples, self.sentiment_examples, self.triple_examples,
                                 self.hint_matrixes_a, self.hint_matrixes_o, self.posi_senti_masks),
                                handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print("ssssssssssssssssssssssssssssssssssss", self.examples.shape)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        sentiment_example = self.sentiment_examples[item]
        (aspect_example, opinion_example, polarity) = self.triple_examples[item]
        hint_matrix_a = self.hint_matrixes_a[item]
        hint_matrix_o = self.hint_matrixes_o[item]
        posi_senti_mask = self.posi_senti_masks[item]

        overflow_len = len(example) - self.block_size
        if overflow_len > 0:
            random_ind = random.randint(0, overflow_len)  # random integer between 0 and overflow_len (both inclusive)
        else:
            random_ind = 0
        example_block = example[random_ind: random_ind + self.block_size]
        hint_matrix_a = hint_matrix_a[random_ind: random_ind + self.block_size]
        hint_matrix_o = hint_matrix_o[random_ind: random_ind + self.block_size]
        posi_senti_mask = posi_senti_mask[random_ind: random_ind + self.block_size]
        if self.pseudo_label:
            # confidence 的处理
            (a_cfdcs, o_cfdcs) = self.pseudo_confidences[item]
            return example_block, sentiment_example, aspect_example, opinion_example, polarity,hint_matrix_a, hint_matrix_o, posi_senti_mask, a_cfdcs, o_cfdcs
        return example_block, sentiment_example, aspect_example, opinion_example, polarity, hint_matrix_a, hint_matrix_o, posi_senti_mask


def process_triples(sentence, triples, arg_config, tokenizer, end_punc, tuple='triples',
                    pseudo_label=False, retrieve_opn_triple_dicts=None):
    inner_jointer = arg_config["data"]["INNER_JOINTER"]
    start_end_pad = arg_config["data"]["START_END_PAD"]
    # sub_text_seg = arg_config["data"]["SUB_TEXT_SEG"]
    assert inner_jointer is not None
    assert start_end_pad is not None
    # assert sub_text_seg is not None
    if tuple == 'triples' or pseudo_label:
        triples = sorted(triples, key=lambda x: min(x[0][1], x[1][1]))
        triples = sorted(triples, key=lambda x: max(x[0][1], x[1][1]))

    sentence_splitted = sentence.split()
    sentence_encoded = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=None)["input_ids"]
    # sub_seg_list = [0]
    # continue_flag = 0
    # for jdx in range(len(sentence_encoded)):
    #     if continue_flag == 0 and sentence_encoded[jdx] in end_punc:
    #         continue_flag = 1
    #     elif continue_flag == 1 and sentence_encoded[jdx] not in end_punc and jdx not in sub_seg_list:
    #         sub_seg_list.append(jdx)
    #         continue_flag = 0
    # if len(sentence_encoded) not in sub_seg_list:
    #     sub_seg_list.append(len(sentence_encoded))
    hint_matrix_a = torch.zeros((len(sentence_encoded), len(triples)))
    hint_matrix_o = torch.zeros((len(sentence_encoded), len(triples)))
    posi_senti_mask = torch.zeros((len(sentence_encoded)))
    # for idx in range(1, len(sentence_splitted)+1):
    #     ret = tokenizer.decode(sentence_encoded[:idx], clean_up_tokenization_spaces=False)
    #     print(ret)
    # triples = [tri for idx, tri in enumerate(triples) if idx not in del_set]
    aspects, opinions, polarities = [], [], []
    asp_confidence, opn_confidence = [], []
    triple_units = []

    # left, right = 0, len(sentence_encoded) - 1
    dense_to_decoded = []
    dense_str = ""
    last_dense_str_len = len(dense_str)
    for idx in range(1, len(sentence_encoded) + 1):
        dense_str = tokenizer.decode(sentence_encoded[:idx], clean_up_tokenization_spaces=False).replace(' ','')
        dense_to_decoded.extend([idx-1]*(len(dense_str) - last_dense_str_len))
        last_dense_str_len = len(dense_str)
        # print(dense_to_decoded)
    hint_tree = [0]
    for jdx, tri in enumerate(triples):
        asp = ' '.join(sentence_splitted[tri[0][0]:tri[0][1] + 1])
        aspects.append(start_end_pad + ' ' + asp)
        asp_span = ''.join(sentence_splitted[tri[0][0]:tri[0][1] + 1])
        if asp_span not in dense_str:
            print(asp_span)
        assert asp_span in dense_str
        asp_left = dense_str.index(asp_span)
        asp_right = asp_left + len(asp_span) - 1
        asp_left, asp_right = dense_to_decoded[asp_left], dense_to_decoded[asp_right] + 1
        posi_senti_mask[asp_left: asp_right] = 1  # asp在句子中的位子置1

        hint_tree_idx = len(hint_tree) - 1
        while asp_right-1 < hint_tree[hint_tree_idx]:
            hint_tree_idx -= 1
        hint_matrix_a[hint_tree[hint_tree_idx]: asp_right-1, jdx] = 1  # 向左偏移一位，以符合训练过程
        ht_add_branch = [max(0, asp_right-1)]
        if tuple == 'triples' or pseudo_label:
            opn = ' '.join(sentence_splitted[tri[1][0]:tri[1][1] + 1])
            opinions.append(start_end_pad + ' ' + opn)
            opn_span = ''.join(sentence_splitted[tri[1][0]:tri[1][1] + 1])
            if opn_span not in dense_str:
                print(opn_span)
            assert opn_span in dense_str
            # opinion
            opn_left = dense_str.index(opn_span)
            opn_right = opn_left + len(opn_span) - 1
            opn_left, opn_right = dense_to_decoded[opn_left], dense_to_decoded[opn_right] + 1
            posi_senti_mask[opn_left: opn_right] = 1  # opn在句子中的位子置1

            hint_tree_idx = len(hint_tree) - 1
            while opn_right-1 < hint_tree[hint_tree_idx]:
                hint_tree_idx -= 1
            if tuple == 'triples' or arg_config["data"]["HINT_OPINIONS"]:
                hint_matrix_o[hint_tree[hint_tree_idx]: opn_right-1, jdx] = 1  # 向左偏移一位，以符合训练过程
                ht_add_branch = [max(0, asp_right-1), max(0, opn_right-1)]
        else:
            if pseudo_label:
                opinions.append(start_end_pad + ' ' + arg_config["data"]["polarity_placer"][tri[2]])
            elif retrieve_opn_triple_dicts is not None:
                # (sem_triple_dict, mams_triple_dict) = retrieve_opn_triple_dicts
                opn = retrieve_similar_opn(asp, tri[1], retrieve_opn_triple_dicts)
                opinions.append(start_end_pad + ' ' + opn)
            else:
                opinions.append(start_end_pad + ' ' + arg_config["data"]["polarity_placer"][tri[1]])
        hint_tree.extend(ht_add_branch)
        hint_tree = list(set(hint_tree))
        hint_tree = sorted(hint_tree)
        # confidence computation
        if pseudo_label:
            if arg_config["data"]["tuple_confidence_computation"] == "mean":
                asp_cfdc = 0. if len(tri[3]) == 0 else (sum(tri[3]) / len(tri[3]))
                asp_confidence.append(asp_cfdc)
                opn_cfdc = 0. if len(tri[4]) == 0 else (sum(tri[4]) / len(tri[4]))
                opn_confidence.append(opn_cfdc)
            elif arg_config["data"]["tuple_confidence_computation"] == "multi":
                if len(tri[3]) == 0:
                    asp_cfdc = 0.
                else:
                    asp_cfdc = 1.
                    for aa in tri[3]:
                        asp_cfdc *= aa
                asp_confidence.append(asp_cfdc)
                if len(tri[4]) == 0:
                    opn_cfdc = 0.
                else:
                    opn_cfdc = 1.
                    for oo in tri[4]:
                        opn_cfdc *= oo
                opn_confidence.append(opn_cfdc)
            else:
                raise KeyError
        if pseudo_label or tuple == 'triples':
            pol_jnt = arg_config["data"][tri[2] + "_JOINTER"]
        else:
            pol_jnt = arg_config["data"][tri[1]+"_JOINTER"]
        if tuple == 'triples':
            opn = opn
            polarities.append(arg_config["data"]["polarity_vocab"][tri[2]])
        else:
            if pseudo_label:
                if tri[1] == tri[0]:  # when aspect and opinion is the same span, replace with 'ok perfect terrible'
                    opn = arg_config["data"]["polarity_placer"][tri[2]]
                else:
                    opn = opn
                polarities.append(arg_config["data"]["polarity_vocab"][tri[2]])
            else:
                if retrieve_opn_triple_dicts is not None:
                    opn = opn
                else:
                    opn = arg_config["data"]["polarity_placer"][tri[1]]
                polarities.append(arg_config["data"]["polarity_vocab"][tri[1]])
        triple_units.append(start_end_pad + ' ' + asp + pol_jnt + opn)

    processed_sentence = ' '.join(sentence_splitted)
    if pseudo_label:
        assert len(aspects)==len(opinions)==len(asp_confidence)==len(opn_confidence)==len(triple_units)
        return processed_sentence, triple_units, aspects, opinions, polarities,hint_matrix_a, hint_matrix_o, posi_senti_mask, asp_confidence,opn_confidence,
    assert len(aspects) == len(opinions) == len(triple_units)
    return processed_sentence, triple_units, aspects, opinions, polarities, hint_matrix_a, hint_matrix_o, posi_senti_mask


def get_triple_dict(data_fn, tuple_name='triples'):
    sem_triple_dict = {}
    with open(data_fn, 'r') as f:
        sem_jd = json.load(f)
        for jd in sem_jd:
            line_split = jd['sentence'].split()
            _triples = jd[tuple_name]
            for cur_triple in _triples:
                aspect_term = ' '.join(line_split[cur_triple[0][0]: cur_triple[0][1] + 1]).lower()
                if cur_triple[0] == cur_triple[1]:
                    continue
                    # opinion_term = senti_dict[cur_triple[2]]
                else:
                    opinion_term = ' '.join(line_split[cur_triple[1][0]: cur_triple[1][1] + 1])
                polarity = cur_triple[2]
                if aspect_term not in sem_triple_dict:
                    sem_triple_dict[aspect_term] = {polarity: set([opinion_term])}
                else:
                    if polarity not in sem_triple_dict[aspect_term]:
                        sem_triple_dict[aspect_term][polarity] = set([opinion_term])
                    else:
                        sem_triple_dict[aspect_term][polarity].add(opinion_term)
        for aspect_term in sem_triple_dict:
            if 'POS' not in sem_triple_dict[aspect_term]:
                sem_triple_dict[aspect_term]['POS'] = set(['great','good','wonderful','not bad','above average'])
            if 'NEG' not in sem_triple_dict[aspect_term]:
                sem_triple_dict[aspect_term]['NEG'] = set(['awful','horrible','disgusting','below average'])
            if 'NEU' not in sem_triple_dict[aspect_term]:
                sem_triple_dict[aspect_term]['NEU'] = set(['average', 'so-so', 'nothing special'])
            # if 'POS' not in sem_triple_dict[aspect_term]:
            #     sem_triple_dict[aspect_term]['POS'] = set(['perfect','great','good','wonderful','not bad','above average'])
            # if 'NEG' not in sem_triple_dict[aspect_term]:
            #     sem_triple_dict[aspect_term]['NEG'] = set(['terrible','awful','horrible','disgusting','below average'])
            # if 'NEU' not in sem_triple_dict[aspect_term]:
            #     sem_triple_dict[aspect_term]['NEU'] = set(['ok', 'okay', 'average', 'so-so'])

    return sem_triple_dict


def minDistance(word1: str, word2: str) -> int:
    m = len(word1)
    n = len(word2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(m+1):
        dp[0][i]=i
    for j in range(n+1):
        dp[j][0]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if word1[j-1]==word2[i-1]:
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min(dp[i][j-1]+1,dp[i-1][j-1]+1,dp[i-1][j]+1)

    return dp[n][m]


def retrieve_similar_opn(aspect_term, polarity, triple_dicts):
    aspect_term = aspect_term.lower()
    for triple_d in triple_dicts:
        if aspect_term in triple_d:
            candidates = list(triple_d[aspect_term][polarity])
            len_subdic = len(candidates)-1
            return candidates[random.randint(0, len_subdic)]
    min_dist = 10000
    most_sml = ''
    for triple_d in triple_dicts:
        for key in triple_d:
            cur_dist = minDistance(aspect_term, key)
            if cur_dist < min_dist:
                min_dist = cur_dist
                most_sml = key
                candidates = list(triple_d[most_sml][polarity])
    len_subdic = len(candidates)-1
    return candidates[random.randint(0, len_subdic)]


# if __name__ == '__main__':
#     arg_config = {"data":{"INNER_JOINTER":" AND ", "START_END_PAD": "<|endoftext|>",
#                           "POS_JOINTER": " is ",
#                           "NEU_JOINTER": " is ",
#                           "NEG_JOINTER": " is ","polarity_vocab": {"NEG": 1, "NEU": 2,"POS": 3},
#                           },
#                   "model":{"model_name_or_path":"gpt2-medium", "cache_dir": None,},
#                   "end_segment_id": [11, 13, 492, 11485, 986, 1106, 12359, 16317, 25780, 2109, 23513, 14, 26, 0, 3228,
#                                      10185, 13896,
#                                      50184, 30, 3548, 28358, 9805, 19622, 1003, 20379, 9705, 837, 764, 11485, 2644,
#                                      19424, 47082,
#                                      20004, 2162, 5145, 37867, 5633, 19153, 34913, 1220, 3373, 34013]
#                   }
#     line_data = "Not only was the food outstanding , but the little ' perks ' were great ."
#     line_data1 = "Service could be improved but overall this is a place that understands the importance of little things ( the heavy , black , antique-seeming teapot , for one ) in the restaurant experience ."
#
#     triples = [[[4, 4], [5, 5], "POS"], [[11, 11], [14, 15], "POS"]]
#     triples1 =[[[0, 0], [3, 3], "NEG"], [[24, 24], [23, 23], "POS"]]
#     tokenizer = GPT2Tokenizer.from_pretrained(arg_config['model']['model_name_or_path'],
#                                                 cache_dir=arg_config['model']['cache_dir'])
#     process_triples(line_data1, triples1, arg_config, tokenizer, arg_config["end_segment_id"])