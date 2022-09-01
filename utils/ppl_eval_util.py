import math
import os
import shutil
from typing import List, Tuple

import torch
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from model.loss_function.SentimentControlLoss import SentimentControlLoss, SentimentReinforceControlLoss
from model.transformers import WEIGHTS_NAME
from utils.data_util import load_and_cache_examples, constract_batch_mask
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import glob

def set_seed(arg_config):
    random.seed(arg_config['data']['random_seed'])
    np.random.seed(arg_config['data']['random_seed'])
    torch.manual_seed(arg_config['data']['random_seed'])
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(arg_config['data']['random_seed'])

class PPLEvalField(object):
    def __init__(self, g_args, tokenizer):
        self.g_args = g_args
        self.tokenizer = tokenizer
        self.eval_dataloader = self.collect_dataset()

    def collect_dataset(self):
        if self.g_args['data']['gen_cs_len'] is None:
            self.g_args['data']['gen_cs_len'] = self.g_args['data']['cs_len']
        if self.g_args['data']['gen_hs_len'] is None:
            self.g_args['data']['gen_hs_len'] = self.g_args['data']['hs_len']
        if self.g_args['data']['gen_tis_len'] is None:
            self.g_args['data']['gen_tis_len'] = self.g_args['data']['tis_len']
        eval_dataset = load_and_cache_examples(self.g_args, self.tokenizer,
                                               evaluate=True,
                                               generate=True,
                                               retrieve_opinion=True)
        gen_args = self.g_args
        tokenizer = self.tokenizer
        def collate_P1(examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
            def pad_sentiment(sentiment_units, stce_sub_seg=None, sentence_len=None, start_p=1, end_p=None,
                              # start_cut, end_cut stands for start and end length
                              batch_first=True, padding_value=gen_args["data"]["START_END_IDX"],
                              senti_token_type_id=gen_args["data"][
                                  "senti_type_id"]):  # pad B * A * T(batch, aspect/opinion, token)
                second_lens = [len(unit) for unit in sentiment_units]
                data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]
                data = [[senti[start_p: end_p] for senti in unit] for unit in sentiment_units]  # B_A'_T'
                senti_unit_pad_mask = constract_batch_mask(data)
                type_ids = torch.full_like(senti_unit_pad_mask, senti_token_type_id)
                type_ids = (type_ids * senti_unit_pad_mask).long()
                data = [torch.tensor(d) for da in data for d in da]  # BA'_T'
                data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # BA'_T
                data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]
                data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # B_A_T
                return data, senti_unit_pad_mask, type_ids  # , batch_mask   # senti_pad_mask: B_A_T batch_mask: B_S_A_T

            stcs = [torch.tensor(st[0], dtype=torch.long) for st in examples]
            senti_units = [st[1] for st in examples]
            aspts = [st[2] for st in examples]
            opns = [st[3] for st in examples]
            plrts = [torch.tensor(st[4], dtype=torch.long) for st in examples]
            hint_matrix_a = [st[5] for st in examples]
            hint_matrix_o = [st[6] for st in examples]
            posi_senti_mask = [st[7] for st in examples]
            stce_sub_seg_position = [[indx for indx, tk in enumerate(stc) if tk == gen_args["data"]["SUB_TEXT_SEG_ID"]]
                                     for stc in stcs]
            if tokenizer._pad_token is None:
                sentences = pad_sequence(stcs, batch_first=True, padding_value=gen_args["data"]["START_END_IDX"])
                senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                                                          stce_sub_seg=stce_sub_seg_position,
                                                                                          sentence_len=sentences.shape[
                                                                                              1],
                                                                                          batch_first=True,
                                                                                          padding_value=
                                                                                          gen_args["data"][
                                                                                              "START_END_IDX"])
                plrts = pad_sequence(plrts, batch_first=True)
            else:
                sentences = pad_sequence(stcs, batch_first=True, padding_value=tokenizer.pad_token_id)
                senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                                                          stce_sub_seg=stce_sub_seg_position,
                                                                                          sentence_len=sentences.shape[
                                                                                              1],
                                                                                          batch_first=True,
                                                                                          padding_value=tokenizer.pad_token_id, )
                plrts = pad_sequence(plrts, batch_first=True, padding_value=tokenizer.pad_token_id)
            b_len = sentences.shape[0]
            s_len = sentences.shape[1]
            a_len = senti_inputs.shape[1]
            asp_pad_mask = torch.clone(senti_units_pad_mask)
            hint_matrix_padded_a = torch.zeros((b_len, s_len, a_len))
            hint_matrix_padded_o = torch.zeros((b_len, s_len, a_len))
            posi_senti_mask_padded = torch.zeros((b_len, s_len))
            for idx in range(b_len):
                hint_matrix_padded_a[idx, :hint_matrix_a[idx].shape[0], :hint_matrix_a[idx].shape[1]] = hint_matrix_a[
                                                                                                            idx][
                                                                                                        :, :]
                hint_matrix_padded_o[idx, :hint_matrix_o[idx].shape[0], :hint_matrix_o[idx].shape[1]] = hint_matrix_o[
                                                                                                            idx][
                                                                                                        :, :]
                posi_senti_mask_padded[idx, :posi_senti_mask[idx].shape[0]] = posi_senti_mask[idx][:]
                if gen_args["data"]["INPUT_SENTI_TUPLE"] == 'triples' or gen_args["data"]["HINT_OPINIONS"]:
                    for jdx in range(len(aspts[idx])):
                        asp_pad_mask[idx, jdx, len(aspts[idx][jdx]) - 1:] = 0  # B_A_T, -1 for 'start_token'
            hint_matrix_padded = torch.cat((hint_matrix_padded_a, hint_matrix_padded_o), dim=2)
            return sentences, sentiment_token_types, senti_inputs, senti_units_pad_mask, asp_pad_mask, aspts, opns, plrts, hint_matrix_padded, posi_senti_mask_padded

        eval_sampler = RandomSampler(eval_dataset) if gen_args["data"]["local_rank"] == -1 else DistributedSampler(
            eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler,
            batch_size=gen_args['training']['per_gpu_train_batch_size'], collate_fn=collate_P1
        )
        return eval_dataloader

    def fit(self, gen_args, model):
        model.eval()
        lm_eval_loss = 0.0
        eval_loss = 0.0
        gen_loss = 0.0
        enhance_loss = 0.0
        positional_loss = 0.0
        with torch.no_grad():
            sentiment_control_loss = SentimentControlLoss(mask_mean=gen_args["data"]["ctrl_mask_mean"])
            sentiment_reinforce_control_loss = SentimentReinforceControlLoss()
            for batch in tqdm(self.eval_dataloader, desc="Evaluating PPL..."):
                inputs, lm_labels = (batch[0], batch[0])
                token_type_ids = batch[1]
                token_type_ids = token_type_ids.to(gen_args['data']['device'])
                sentiment_inputs = batch[2]
                if gen_args['data']['use_attn_mask']:
                    senti_pad_mask = batch[3]
                    aspect_pad_mask = batch[4]
                    senti_pad_mask = senti_pad_mask.to(gen_args['data']['device'])
                    aspect_pad_mask = aspect_pad_mask.to(gen_args['data']['device'])
                else:
                    senti_pad_mask = None
                    aspect_pad_mask = None
                aspects = batch[5]
                opinions = batch[6]
                if gen_args["data"]["use_hint_matrix"]:
                    hint_matrix = batch[8]
                    hint_matrix = hint_matrix.to(gen_args['data']['device'])
                else:
                    hint_matrix = None
                reinforce_control_masks = batch[9]
                reinforce_control_masks = reinforce_control_masks.to(gen_args['data']['device'])
                if inputs.shape[1] < gen_args['data']['hs_len']:
                    # "inputs.shape[1] < arg_config['data']['hs_len'], skipping batch"
                    continue

                sentiment_inputs = sentiment_inputs.view(sentiment_inputs.shape[0], -1)  # B_A_T → B_AT
                inputs = torch.cat((sentiment_inputs, inputs), dim=1)  # B_(AT+S)
                lm_labels = torch.cat((sentiment_inputs, lm_labels), dim=1)  # B_(AT+S)
                # lm_labels = lm_labels[:, :hs_len + tis_len]
                inputs = inputs.to(gen_args['data']['device'])
                lm_labels = lm_labels.to(gen_args['data']['device'])

                lm_logit_first_index = sentiment_inputs.shape[1]  # AT
                lm_labels_first_index = lm_logit_first_index + 1
                outputs = model(inputs, labels=lm_labels, query_hint_matrix=hint_matrix,
                                hint_lbd=gen_args["data"]["hint_lambda"],
                                aspect_attention_mask=aspect_pad_mask,
                                context_attention_mask=senti_pad_mask,
                                lm_logit_first_index=lm_logit_first_index,
                                lm_labels_first_index=lm_labels_first_index)

                self_cocon_lm_loss = outputs[0]
                # if gen_args["training"]["lambda_self_cocon_lm_loss"] > 0:
                eval_loss += self_cocon_lm_loss.item()
                gen_loss += self_cocon_lm_loss.item()
                if gen_args["training"]["lambda_aspects_control_loss"] > 0:
                    aspects_control_loss = sentiment_control_loss(gen_args,
                                                                  lm_labels[:, :],
                                                                  outputs[1][:, lm_logit_first_index:-1],
                                                                  sentiment_atoms=aspects)
                    # eval_loss += gen_args["training"]["lambda_aspects_control_loss"] * aspects_control_loss.item()
                    eval_loss += aspects_control_loss.item()
                    enhance_loss += aspects_control_loss.item()
                if gen_args["training"]["lambda_opinions_control_loss"] > 0:
                    opinions_control_loss = sentiment_control_loss(gen_args,
                                                                   lm_labels[:, :],
                                                                   outputs[1][:, lm_logit_first_index:-1],
                                                                   sentiment_atoms=opinions)
                    # eval_loss += gen_args["training"]["lambda_opinions_control_loss"] * opinions_control_loss.item()
                    eval_loss += opinions_control_loss.item()
                    enhance_loss += opinions_control_loss.item()
                if gen_args["training"]["lambda_positional_sentiment_control_loss"] > 0:
                    positional_reinforce_loss = sentiment_reinforce_control_loss(
                        lm_logit_first_index=lm_logit_first_index,
                        lm_labels_first_index=lm_labels_first_index,
                        input_labels=lm_labels[:, :],
                        generation_logits=outputs[1][:, :-1],
                        reinforce_control_masks=reinforce_control_masks
                        )
                    # eval_loss += gen_args["training"][
                    #                   "lambda_positional_sentiment_control_loss"] * positional_reinforce_loss.item()
                    eval_loss += positional_reinforce_loss.item()
                    positional_loss += positional_reinforce_loss.item()
            eval_loss = eval_loss / len(self.eval_dataloader)
            perplexity = math.exp(eval_loss)
            print("gen ppl: {}".format(math.exp(gen_loss / len(self.eval_dataloader))))
            print("enhance ppl: {}".format(math.exp(enhance_loss / len(self.eval_dataloader))))
            print("positional ppl: {}".format(math.exp(positional_loss / len(self.eval_dataloader))))
            return perplexity

    def eval_2_save(self, g_args, save_dir, model):
        # save_dir = '../' + save_dir
        os.makedirs(save_dir, exist_ok=True)
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(save_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0]
            min_ppl_num = float(checkpoint.split("-")[-1])
        else:
            checkpoint = ''
            min_ppl_num = 1000000.
        cur_ppl_num = self.fit(g_args, model)
        print(cur_ppl_num)
        if cur_ppl_num < min_ppl_num:
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            ready_to_save_dir = save_dir + '/ppl-'+str(cur_ppl_num)
            os.makedirs(ready_to_save_dir, exist_ok=True)
            model_to_save.save_pretrained(ready_to_save_dir)
            self.tokenizer.save_pretrained(ready_to_save_dir)

            if checkpoint != '':
                shutil.rmtree(checkpoint)