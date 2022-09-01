

import csv
import json
import logging
import os
from typing import Dict, List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
# from rouge import Rouge
from model.dataloader.JsonlCoconDataset import JsonlCoconDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sacrebleu
import nltk
from nltk.translate import (nist_score, meteor_score)
from sacrebleu.metrics import BLEU, CHRF, TER
# from stanfordcorenlp import StanfordCoreNLP

# stf_nlp = StanfordCoreNLP(r'E:\stanford-corenlp\stanford-corenlp-4.4.0')
from model.transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    SenTriConBlock
)
logger = logging.getLogger(__name__)


def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False

def remove_pad_token(padded_data: List, padding_tokens=[50256]):
    return list(filter(lambda x: x not in padding_tokens, padded_data))


def remove_empty_sequence(seq_data: List):
    return list(filter(lambda x: len(x) > 0, seq_data))


def constract_batch_mask(batch: List[List[List]], stce_sub_seg=None, sentence_len=None):
    len_batch = len(batch)
    max_asp = max([len(asps) for asps in batch])
    max_sentence = max([len(stc) for asps in batch for stc in asps])
    mask = torch.ones((len_batch, max_asp, max_sentence))
    for i, asps in enumerate(batch):  # mask the rest part of aspect_opinion pairs
        mask[i, len(asps):] = 0.
        for j, sentence in enumerate(asps):  # mask the rest part of inner pair
            mask[i, j, len(sentence):] = 0.
    if stce_sub_seg is None or sentence_len is None:
        return mask  # B_A_T
    else:
        assert len_batch == len(stce_sub_seg)
        mask = mask.unsqueeze(dim=1)  # B_1_A_T
        mask = mask.repeat(1, sentence_len, 1, 1)   # B_S_A_T
        # mask = mask.expand(mask.shape[0], sentence_len, mask.shape[2], mask.shape[3])  # B_S_A_T
        for idx in range(0, len_batch):
            last_seg = 0
            for jdx in range(0, len(stce_sub_seg[idx])):
                cur_seg = stce_sub_seg[idx][jdx]
                mask[idx, last_seg: cur_seg, jdx+1:] = 0.
                mask[idx, cur_seg:, jdx] = 0.
                last_seg = cur_seg
        return mask


def load_and_cache_examples(arg_config, tokenizer, evaluate=False, file_path=None,
                            use_labeled_data=True, generate=False, line_by_line=False,
                            prepend_bos_token=False, text_json_key="sentence", pseudo_data=False,
                            prepended_text_to_remove=None, retrieve_opinion=False):
    if generate:
        cs_len = arg_config["data"]["gen_cs_len"]
        hs_len = arg_config["data"]["gen_hs_len"]
        tis_len = arg_config["data"]["gen_tis_len"]
    else:
        cs_len = arg_config["data"]["cs_len"]
        hs_len = arg_config["data"]["hs_len"]
        tis_len = arg_config["data"]["tis_len"]

    if file_path is None:
        if use_labeled_data:
            file_path = arg_config["data"]["labeled_test"] if evaluate else arg_config["data"]["labeled"]
        else:
            file_path = arg_config["data"]["unlabeled_test"] if evaluate else arg_config["data"]["unlabeled"]

    if pseudo_data:
        logger.info("Creating JsonlCoconDataset from pseudo dataset")
        return JsonlCoconDataset(tokenizer, arg_config, file_path=file_path,
                                 use_labeled_data=False, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len,
                                 text_triple_key=arg_config["data"]["INPUT_SENTI_TUPLE"],
                                 pseudo_label=True)
    if evaluate:
        logger.info("Creating JsonlCoconDataset for eval")
        return JsonlCoconDataset(tokenizer, arg_config, file_path=file_path, use_labeled_data=use_labeled_data,
                                 # block_size=arg_config["model"]["block_size"],
                                 text_json_key=text_json_key, cs_len=cs_len, hs_len=hs_len, tis_len=tis_len,
                                 text_triple_key=arg_config["data"]["INPUT_SENTI_TUPLE"],
                                 evaluate=True, prepended_text_to_remove=prepended_text_to_remove,
                                 retrieve_opinion=retrieve_opinion)
    else:
        return JsonlCoconDataset(tokenizer, arg_config, file_path=file_path, use_labeled_data=use_labeled_data,
                                 cs_len=cs_len, hs_len=hs_len, tis_len=tis_len,
                                 text_triple_key=arg_config["data"]["INPUT_SENTI_TUPLE"],)


def generate_alsecond_sample(original_transform_input_seq, original_history_seq, original_context_seq, inputs,
                             senti_triple_inputs, alsecond_output_file_path, arg_config,
                             model, tokenizer,
                             token_type_ids, senti_attention_mask=None, aspect_pad_mask=None,
                             alsecond_output_jsonl_file_path=None, transform_h_after_layernorm=False,
                             prepend_history_seq=False,
                             original_dia_history_seq=None, dia_context_seq=None, original_dia_context_seq=None,
                             end_of_text_id=None, single_generation=False):
    with torch.no_grad():
        encoded_prompt = original_history_seq

        if arg_config["data"]["line_by_line_hs"] == False and original_context_seq is not None:
            alsecond_gen_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=arg_config["data"]["generate_length"] + len(encoded_prompt[0]),
                temperature=arg_config["data"]["temperature"],
                top_k=arg_config["data"]["k"],
                top_p=arg_config["data"]["p"],
                repetition_penalty=arg_config["data"]["repetition_penalty"],
                do_sample=arg_config["data"]["do_sample"],
                num_beams=arg_config["data"]["num_beams"],
                num_return_sequences=arg_config["data"]["num_return_sequences"],
                tokenizer=tokenizer,
                token_type_ids=None,
                sequence_mask=senti_attention_mask,
                aspect_pad_mask=aspect_pad_mask,
                time_step_mask=None,
                senti_triple_inputs=senti_triple_inputs,
                arg_config=arg_config,
                cocon_context_inputs=original_context_seq,
                cocon_history_inputs=original_history_seq,
                split_senti_key_and_value_hidden=arg_config["model"]["split_key_and_value_hidden"],
                senti_hidden_use_only_key=arg_config["model"]["senti_hidden_use_only_key"],
                transform_h_after_layernorm=transform_h_after_layernorm,
                context_attn_bias=arg_config["data"]["context_attn_bias"],
                hint_generation=arg_config["data"]["use_hint_matrix"],
                strategy_generation=arg_config["evaluate"]["strategy_generation"]
            )
            if len(alsecond_gen_output_sequences.shape) > 2:
                alsecond_gen_output_sequences.squeeze_()
            sentiment_input_len = original_context_seq.shape[1] * original_context_seq.shape[2]

        alsecond_output_text_lines_dict = {}
        for generated_sequence_idx, generated_sequence in enumerate(alsecond_gen_output_sequences):
            if alsecond_output_jsonl_file_path is not None:
                alsecond_jsonl_output_dict = {}
            # Decode and log original_input_text
            original_input_sequence = inputs[generated_sequence_idx]
            original_input_sequence = remove_pad_token(original_input_sequence.tolist())
            original_input_text = tokenizer.decode(original_input_sequence, clean_up_tokenization_spaces=True)
            alsecond_output_text_lines_dict[generated_sequence_idx] = [
                "original_input_text: {} \n".format(original_input_text)]
            if alsecond_output_jsonl_file_path is not None:
                alsecond_jsonl_output_dict["original_input_text"] = original_input_text

            # Decode and log original_history_seq
            original_history_sequence = original_history_seq[generated_sequence_idx]
            original_history_sequence = original_history_sequence.tolist()
            original_history_text = tokenizer.decode(original_history_sequence, clean_up_tokenization_spaces=True)
            alsecond_output_text_lines_dict[generated_sequence_idx].append(
                "original_history_text: {} \n".format(original_history_text))
            if alsecond_output_jsonl_file_path is not None:
                alsecond_jsonl_output_dict["original_history_text"] = original_history_text

            # Decode and log original_context_seq
            if arg_config["data"]["line_by_line_hs"] == False and original_context_seq is not None:
                original_context_sequence = original_context_seq[generated_sequence_idx]
                original_context_sequence = [remove_pad_token(ocr.tolist())
                                             for ocr in original_context_sequence]
                original_context_sequence = remove_empty_sequence(original_context_sequence)
                shaped_context_text = [tokenizer.decode(ocr, clean_up_tokenization_spaces=True)
                                       for ocr in original_context_sequence]
                # shaped_context_text = shaped_context_text.reshape(-1, original_context_sequence_shape[-1]).tolist() # reshape to A_T
                # shaped_context_text = [remove_pad_token(sct) for sct in shaped_context_text] # TODO 其他self生成的也去除一下'!'
                alsecond_output_text_lines_dict[generated_sequence_idx].append(                 # TODO 然后写一下困惑度和n-gram的衡量方法
                    "original_context_text: {} \n".format(shaped_context_text))
                if alsecond_output_jsonl_file_path is not None:
                    alsecond_jsonl_output_dict["original_sentiment_pair"] = shaped_context_text
            else:
                alsecond_output_text_lines_dict[generated_sequence_idx].append("original_sentiment_pair: None \n")

            # Decode and log alsecond generated text
            if arg_config["data"]["line_by_line_hs"] == False and original_context_seq is not None:
                self_alsecond_gen_sequence = alsecond_gen_output_sequences[generated_sequence_idx]
                if senti_attention_mask is not None:
                    self_alsecond_gen_sequence = self_alsecond_gen_sequence[sentiment_input_len:]
                self_alsecond_gen_sequence = remove_pad_token(self_alsecond_gen_sequence.tolist())
                # self_alsecond_gen_sequence = self_alsecond_gen_sequence.tolist()
                self_alsecond_gen_output_text = tokenizer.decode(self_alsecond_gen_sequence,
                                                                 clean_up_tokenization_spaces=True)
                alsecond_output_text_lines_dict[generated_sequence_idx].append(
                    "AlSeCond output: {} \n".format(self_alsecond_gen_output_text))
                if alsecond_output_jsonl_file_path is not None:
                    alsecond_jsonl_output_dict["self_sentricon_output"] = self_alsecond_gen_output_text

            # Sanity check (SC) prependgpt2_gen_ar_output_sequences: Decode and log AR generated text


            if alsecond_output_jsonl_file_path is not None:
                with open(alsecond_output_jsonl_file_path, "a") as f:
                    json.dump(alsecond_jsonl_output_dict, f)
                    f.write('\n')

    alsecond_output_text_lines = []
    for sample_ind in range(inputs.shape[0]):
        alsecond_output_text_lines = alsecond_output_text_lines + alsecond_output_text_lines_dict[sample_ind] + ["----------\n"]

    with open(alsecond_output_file_path, "a", encoding='utf-8') as f:
        f.writelines(alsecond_output_text_lines)

    return original_input_text

def generate_alsecond_compute(arg_config, eval_output_dir, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                              prefix="", random_sample_data=False,
                              use_only_first_context_source_batch=False,
                              use_only_first_custom_mu_s_input_batch=False, transform_h_after_layernorm=False,
                               prepend_history_seq=False) -> Dict:
    # eval_output_dir = arg_config["model"]["output_dir"]

    sentricon_output_file_path = os.path.join(eval_output_dir,
                                              arg_config["evaluate"]["sentricon_output_filename"])
    if os.path.exists(sentricon_output_file_path):
        if arg_config["evaluate"]["append_sentricon_output_files"]:
            logger.info("Append to existing sentricon output file")
        else:
            logger.info("Removing existing sentricon output file")
            os.remove(sentricon_output_file_path)
    else:
        logger.info("Creating new sentricon output file")

    if arg_config["evaluate"]["sentricon_output_jsonl_filename"] is not None:
        sentricon_output_jsonl_file_path = os.path.join(eval_output_dir,
                                                        arg_config["evaluate"]["sentricon_output_jsonl_filename"])
        if os.path.exists(sentricon_output_jsonl_file_path):
            if arg_config["evaluate"]["append_sentricon_output_files"]:
                logger.info("Append to existing sentricon output jsonl file")
            else:
                logger.info("Removing existing sentricon output jsonl file")
                os.remove(sentricon_output_jsonl_file_path)
        else:
            logger.info("Creating new alsecond output jsonl file")
    else:
        sentricon_output_jsonl_file_path = None


    if arg_config["data"]["INPUT_SENTI_TUPLE"] == "triples":
        sentence_source_dataset = load_and_cache_examples(arg_config, tokenizer,
                                                          evaluate=True,
                                                          generate=True)
    else:
        sentence_source_dataset = load_and_cache_examples(arg_config, tokenizer,
                                                          evaluate=True,
                                                          generate=True,
                                                          retrieve_opinion=True)

    if arg_config["data"]["local_rank"] in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    arg_config["data"]["eval_batch_size"] = arg_config["data"]["per_gpu_eval_batch_size"] * \
                                            max(1, arg_config['data']['n_gpu'])

    # Note that DistributedSampler samples randomly

    if random_sample_data == True:
        sentence_source_sampler = RandomSampler(sentence_source_dataset) \
            if arg_config["data"]["local_rank"] == -1 else DistributedSampler(sentence_source_dataset)
        # context_source_sampler = RandomSampler(context_source_dataset) if arg_config["data"]["local_rank"] == -1 else DistributedSampler(
        #     context_source_dataset)
    else:
        sentence_source_sampler = SequentialSampler(sentence_source_dataset)
        # context_source_sampler = SequentialSampler(context_source_dataset)

    def collate_P1(examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        def pad_sentiment(sentiment_units, stce_sub_seg=None, sentence_len=None, start_p=1, end_p=None,  # start_cut, end_cut stands for start and end length
                          batch_first=True, padding_value=arg_config["data"]["START_END_IDX"],
                          senti_token_type_id=arg_config["data"]["senti_type_id"]):  # pad B * A * T(batch, aspect/opinion, token)
            second_lens = [len(unit) for unit in sentiment_units]
            data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]

            # data = [[bigram[0][start_p:end_p]+jointer+bigram[1][start_p:end_p] for bigram in zip(tup[0], tup[1])]
            #         for tup in zip(aspts, opns)]
            data = [[senti[start_p: end_p] for senti in unit] for unit in sentiment_units]  # B_A'_T'
            batch_mask = constract_batch_mask(data)
            type_ids = torch.full_like(batch_mask, senti_token_type_id)
            type_ids = (type_ids * batch_mask).long()
            data = [torch.tensor(d) for da in data for d in da]  # BA'_T'

            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # BA'_T
            data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]
            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # B_A_T
            assert batch_mask.shape == data.shape
            return data, batch_mask, type_ids             # 生成P1的mask: B_A_T
        stcs = [torch.tensor(st[0], dtype=torch.long) for st in examples]
        senti_units = [st[1] for st in examples]
        aspts = [st[2] for st in examples]
        opns = [st[3] for st in examples]
        plrts = [torch.tensor(st[4], dtype=torch.long) for st in examples]
        # joint_of_aspect_opinion = arg_config["data"]["joint_of_aspect_opinion"]  # defult: "--------"
        stce_sub_seg_position = [[indx+1 for indx, tk in enumerate(stc) if tk == arg_config["data"]["SUB_TEXT_SEG_ID"]]
                                 for stc in stcs]
        batched_senti_words = [[tokenizer.decode(stc, clean_up_tokenization_spaces=False).split()
                                for stc in trip]
                               for trip in senti_units]
        if tokenizer._pad_token is None:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=arg_config["data"]["START_END_IDX"])
            senti_inputs, senti_batch_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                               stce_sub_seg=stce_sub_seg_position,
                                                               sentence_len=sentences.shape[1],
                                                               batch_first=True,
                                                               padding_value=arg_config["data"]["START_END_IDX"])
            plrts = pad_sequence(plrts, batch_first=True)
        else:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=tokenizer.pad_token_id)
            senti_inputs, senti_batch_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                               stce_sub_seg=stce_sub_seg_position,
                                                               sentence_len=sentences.shape[1],
                                                               batch_first=True,
                                                               padding_value=tokenizer.pad_token_id)
            plrts = pad_sequence(plrts, batch_first=True, padding_value=tokenizer.pad_token_id)
        asp_pad_mask = torch.clone(senti_batch_mask)
        if arg_config["data"]["INPUT_SENTI_TUPLE"] == 'triples' or arg_config["data"]["HINT_OPINIONS"]:
            for idx in range(sentences.shape[0]):
                for jdx in range(len(aspts[idx])):
                    asp_pad_mask[idx, jdx, len(aspts[idx][jdx]) - 1:] = 0  # B_A_T, -1 for 'start_token'
        return sentences, sentiment_token_types, senti_inputs, senti_batch_mask, asp_pad_mask, aspts, opns, plrts, batched_senti_words

    sentence_source_dataloader = DataLoader(
        sentence_source_dataset, sampler=sentence_source_sampler, batch_size=arg_config["data"]["eval_batch_size"],
        collate_fn=collate_P1
    )
    # context_source_dataloader = DataLoader(
    #     sentence_source_dataset, sampler=sentence_source_sampler, batch_size=arg_config["data"]["eval_batch_size"], collate_fn=collate
    # )
    # context_source_dataloader_iter = iter(context_source_dataloader)

    # multi-gpu evaluate
    if arg_config['data']['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Generate alsecond samples!
    logger.info("***** Running alsecond generation {} *****".format(prefix))
    logger.info("  Batch size = %d", arg_config["data"]["eval_batch_size"])
    eval_loss = 0.0
    nb_generate_alsecond_steps = 0
    model.eval()

    # if use_only_first_context_source_batch and args.use_history_source_as_context_source_for_gen == False:
    #     context_source_batch = next(context_source_dataloader_iter)
    #     context_source_inputs = context_source_batch

    for batch_ind, batch in enumerate(tqdm(sentence_source_dataloader, desc="Generating")):
        inputs = batch[0]
        token_type_ids = batch[1]
        token_type_ids = token_type_ids.to(arg_config['data']['device'])
        sentiment_inputs = batch[2]
        if arg_config['data']['use_attn_mask']:
            senti_btc_mask = batch[3]  # mask 利用一下
            aspect_pad_mask = batch[4]
            senti_btc_mask = senti_btc_mask.to(arg_config['data']['device'])
            aspect_pad_mask = aspect_pad_mask.to(arg_config['data']['device'])
        else:
            senti_btc_mask = None
            aspect_pad_mask = None
        aspects = batch[5]
        opinions = batch[6]
        polarities = batch[7]
        senti_triple_inputs = batch[8]

        inputs = inputs.to(arg_config['data']['device'])
        original_context_seq = sentiment_inputs.to(arg_config['data']['device'])
        original_history_seq = inputs[:, :arg_config["data"]["gen_hs_len"]]
        original_transform_input_seq = inputs[:, arg_config["data"]["gen_hs_len"]:arg_config["data"]["gen_hs_len"] +
                                                                                  arg_config["data"]["gen_tis_len"]]
        sentiment_inputs = original_context_seq.view(original_context_seq.shape[0], -1)  # B_A_T → B_AT
        if arg_config["G1"]["num_train_epochs"] > 0:  # 如果使用append的方式
            original_history_seq = torch.cat((sentiment_inputs, original_history_seq), dim=1)  # B_(AT+S)
        else:  # 不用append的话，mask要去掉
            senti_btc_mask = None
        original_history_seq = original_history_seq.to(arg_config['data']['device'])

        # if arg_config["data"]["line_by_line_cs"]:
        #     context_seq = context_source_inputs
        # else:
        #     context_seq = context_source_inputs[:, arg_config["data"]["gen_hs_len"]:arg_config["data"]["gen_hs_len"] + args.gen_cs_len]
        # context_seq = context_seq.to(args.device)
        original_context_seq = original_context_seq.to(arg_config['data']['device'])
        with open(sentricon_output_file_path, "a", encoding='utf-8') as f:
            f.writelines("***HS #{}***\n".format(batch_ind))

        generate_alsecond_sample(original_transform_input_seq=original_transform_input_seq,
                                 original_history_seq=original_history_seq,
                                 original_context_seq=original_context_seq,
                                 inputs=inputs,
                                 senti_triple_inputs=senti_triple_inputs,
                                 alsecond_output_file_path=sentricon_output_file_path,
                                 arg_config=arg_config,
                                 model=model,
                                 tokenizer=tokenizer,
                                 token_type_ids=token_type_ids,
                                 senti_attention_mask=senti_btc_mask,
                                 aspect_pad_mask=aspect_pad_mask,
                                 alsecond_output_jsonl_file_path=sentricon_output_jsonl_file_path,
                                 transform_h_after_layernorm=transform_h_after_layernorm,
                                 prepend_history_seq=prepend_history_seq)

        if nb_generate_alsecond_steps >= arg_config['evaluate']['num_alsecond_generate'] - 1:
            break

        nb_generate_alsecond_steps += 1

    return nb_generate_alsecond_steps


def compute_bleu(predict: List, gold: List[List]):

    bleu_score = sacrebleu.corpus_bleu(predict, gold)

    return bleu_score.precisions


def evaluate_BLEU(generated_file_path,  prefix="", sub_seg='</', tail_pad='<|endoftext|>', cut_prompt=False):
    generated_file_path = prefix + generated_file_path
    gold, prompt, input_senti, sentricon_out = [], [], [], []
    # gold_sss = []
    with(open(generated_file_path, 'r', encoding='utf-8-sig')) as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text'].replace(sub_seg, "")
            if cut_prompt:
                gold.append(js_data['original_input_text'].replace(sub_seg, "").replace(tail_pad, "")[len(history_input):].lstrip())
                sentricon_out.append(js_data['self_sentricon_output'].replace(sub_seg, "")[len(history_input):].lstrip())
                # gpt2_out.append(js_data['sc_gpt2_ar_gen'].replace(sub_seg, "")[len(history_input):].lstrip())
            else:
                gold.append(js_data['original_input_text'].replace(sub_seg, "").replace(tail_pad, ""))
                sentricon_out.append(js_data['self_sentricon_output'].replace(sub_seg, ""))
                # gpt2_out.append(js_data['sc_gpt2_ar_gen'].replace(sub_seg, ""))
            prompt.append(history_input)
            input_senti.append(js_data['original_sentiment_pair'])
    # gold_sss = gold
    gold = [gold]
    (bleu_1, bleu_2, bleu_3,bleu_4) = compute_bleu(sentricon_out, gold=gold)
    # (gpt_bleu_1, gpt_bleu_2, gpt_bleu_3, gpt_bleu_4) = compute_bleu(gpt2_out, gold=gold)
    # (gd_bleu_1, gd_bleu_2, gd_bleu_3, gd_bleu_4) = compute_bleu(gold_sss, gold=gold)
    return bleu_1, bleu_2, bleu_3,bleu_4
            # bleu_4 = 0.08  gpt_bleu_4 = 0.10

def evaluate_NIST_AND_METEOR_AND_ROUGE(generated_file_path,  prefix="", sub_seg='</', tail_pad='<|endoftext|>', cut_prompt=False):
    generated_file_path = prefix + generated_file_path
    gold, gold_unsqez, prompt, input_senti, sentricon_out, gpt2_out = [], [], [], [], [], []
    gold_lists, sentricon_out_lists, gpt2_out_lists = [], [], []
    # gold_sss = []
    # rouge = RougeAugmented(max_n=4)
    # rouge = Rouge()
    with(open(generated_file_path, 'r', encoding='utf-8-sig')) as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text'].replace(sub_seg, "")
            if cut_prompt:
                gold_str = js_data['original_input_text'].replace(sub_seg, "").replace(tail_pad, "")[len(history_input):].lstrip()
                sentricon_str = js_data['self_sentricon_output'].replace(sub_seg, "")[len(history_input):].lstrip()
                gpt2_str = js_data['sc_gpt2_ar_gen'].replace(sub_seg, "")[len(history_input):].lstrip()

            else:
                gold_str = js_data['original_input_text'].replace(sub_seg, "").replace(tail_pad, "")
                sentricon_str = js_data['self_sentricon_output'].replace(sub_seg, "")
                gpt2_str = js_data['sc_gpt2_ar_gen'].replace(sub_seg, "")
            gold_list = nltk.word_tokenize(gold_str)
            sentricon_list = nltk.word_tokenize(sentricon_str)
            gpt2_list = nltk.word_tokenize(gpt2_str)

            # gold_sss.append(gold_str)
            gold.append([gold_str])  # add a dim
            gold_unsqez.append(gold_str)
            sentricon_out.append(sentricon_str)
            gpt2_out.append(gpt2_str)

            gold_lists.append([gold_list])  # add a dim
            sentricon_out_lists.append(sentricon_list)
            gpt2_out_lists.append(gpt2_list)
            prompt.append(history_input)
            input_senti.append(js_data['original_sentiment_pair'])
    nist_sentricon = nist_score.corpus_nist(gold_lists, sentricon_out_lists, n=5)
    nist_gpt2 = nist_score.corpus_nist(gold_lists, gpt2_out_lists, n=5)
    meteor_sentricon = [meteor_score.meteor_score(gd, stc) for gd, stc in zip(gold, sentricon_out)]
    meteor_sentricon = sum(meteor_sentricon) / len(meteor_sentricon)
    meteor_gpt2 = [meteor_score.meteor_score(gd, stc) for gd, stc in zip(gold, gpt2_out)]
    meteor_gpt2 = sum(meteor_gpt2) / len(meteor_gpt2)
    # rouge_sentricon = rouge.get_scores(sentricon_out, gold_unsqez, avg=True)["rouge-l"]
    # rouge_gpt2 = rouge.get_scores(gpt2_out, gold_unsqez, avg=True)["rouge-l"]
    # return scores
    return nist_sentricon, nist_gpt2, meteor_sentricon, meteor_gpt2,# rouge_sentricon, rouge_gpt2 # , meteor_gold


def score_sentiment_p(texts, senti_units, split_word=' is '):
    assert len(texts) == len(senti_units)
    unit_numbers = 0.
    gen_true_aspect_num, gen_true_opinion_num, gen_true_unit_num=0., 0., 0.
    texts = [text.lower() for text in texts]
    senti_units = [[unit.lower().strip().split(split_word) for unit in senti] for senti in senti_units]
    for idx in range(0, len(texts)):
        unit_numbers += len(senti_units[idx])
        for unit in senti_units[idx]:
            hit_num=0
            if unit[0] in texts[idx]:
                gen_true_aspect_num += 1.
                hit_num+=1
            if len(unit) <= 1 or len(texts) <= idx:
                print(idx)
            if unit[1] in texts[idx]:
                gen_true_opinion_num += 1.
                hit_num+=1
            if hit_num >= 2:
                gen_true_unit_num += 1
    score_asp = 0. if unit_numbers == 0. else gen_true_aspect_num / unit_numbers
    score_opn = 0. if unit_numbers == 0. else gen_true_opinion_num / unit_numbers
    score_unit = 0. if unit_numbers == 0. else gen_true_unit_num / unit_numbers
    return score_asp, score_opn, score_unit


def evaluate_sentiment_hits(generated_file_path,  prefix="", split_word=' is ', sub_seg='</',
                            tail_pad='<|endoftext|>', cut_prompt=False):
    generated_file_path = prefix + generated_file_path
    senti_texts, gpt2_texts, senti_units = [], [], []
    with(open(generated_file_path, 'r', encoding='utf-8-sig')) as f:
        lines = f.readlines()
        for line in lines:
            js_data = json.loads(line)
            history_input = js_data['original_history_text'].replace(sub_seg, "")
            if cut_prompt:
                sentricon_str = js_data['self_sentricon_output'].replace(sub_seg, "")[len(history_input):].lstrip()
                gpt2_str = js_data['sc_gpt2_ar_gen'].replace(sub_seg, "")[len(history_input):].lstrip()

            else:
                sentricon_str = js_data['self_sentricon_output'].replace(sub_seg, "")
                gpt2_str = js_data['sc_gpt2_ar_gen'].replace(sub_seg, "")
            units = js_data['original_sentiment_pair']
            senti_texts.append(sentricon_str)
            gpt2_texts.append(gpt2_str)
            senti_units.append(units)
    senti_score_asp, senti_score_opn, senti_score_unit = score_sentiment_p(senti_texts, senti_units, split_word)
    gpt2_score_asp, gpt2_score_opn, gpt2_score_unit = score_sentiment_p(gpt2_texts, senti_units, split_word)
    return senti_score_asp, senti_score_opn, senti_score_unit, gpt2_score_asp, gpt2_score_opn, gpt2_score_unit

