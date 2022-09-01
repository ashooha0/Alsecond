import argparse
import json
import math

import numpy as np
import logging
import os
import random
import torch
import glob

from classifier.code.mg_gts_v2.data import load_data_instances, DataIterator
from classifier.code.mg_gts_v2.main import reset_params, classifier_eval, classifier_generate_pseudo, \
    train_classifier_with_reconstruction_reward, classifier_generate_pseudo_opinion_span
import torch.nn.functional as F
from classifier.code.mg_gts_v2.model import MG_GTS
from model.transformers.modeling_gpt2 import QueryHintedGPT2LMHeadModel
from trainer import train_lm, fix_state_dict_naming, train_with_pseudo_labels, train_hinted_gpt2
from utils import data_util
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import get_rank, get_world_size
from tqdm import tqdm, trange
from collections import OrderedDict
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

from utils.data_util import load_and_cache_examples, constract_batch_mask, evaluate_BLEU, \
    evaluate_NIST_AND_METEOR_AND_ROUGE
from utils.ppl_eval_util import PPLEvalField
from utils.train_util import evaluate, _clear_checkpoints, _rotate_checkpoints


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, QueryHintedGPT2LMHeadModel),
}
logger = logging.getLogger(__name__)

def set_seed(arg_config):
    random.seed(arg_config['data']['random_seed'])
    np.random.seed(arg_config['data']['random_seed'])
    torch.manual_seed(arg_config['data']['random_seed'])
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(arg_config['data']['random_seed'])

def train_C0(c_args):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='%s/train_log' % working_dir,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    model_path = c_args["model_dir"] + c_args["model"] + c_args["task"] + '.pt'
    if os.path.exists(model_path) and c_args["use_existing"]:
        logging.info("Classifier exists loading from it...")
        model = torch.load(model_path).to(c_args["device"])  # 重新加载最佳的模型到此
        return model
    logging.info("Start training classifier...")
    # load double embedding
    word2index = json.load(open(c_args["prefix"] + 'doubleembedding/word2index.json'))
    # general_embedding = numpy.load(c_args["prefix"] + 'doubleembedding/gen.vec.npy')
    general_embedding = np.load(c_args["prefix"] + 'doubleembedding/glove_filtered_300d.npy')
    general_embedding = torch.from_numpy(general_embedding)
    # domain_embedding = numpy.load(c_args["prefix"] +'doubleembedding/'+c_args["dataset"]+'_emb.vec.npy')
    domain_embedding = np.load(c_args["prefix"] + 'doubleembedding/glove_filtered_300d.npy')
    domain_embedding = torch.from_numpy(domain_embedding)

    # load dataset
    train_sentence_packs = json.load(open(c_args["prefix"] + c_args["dataset"] + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(c_args["prefix"] + c_args["dataset"] + '/dev.json'))

    instances_train = load_data_instances(train_sentence_packs, word2index, c_args)
    instances_dev = load_data_instances(dev_sentence_packs, word2index, c_args)

    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, c_args)
    devset = DataIterator(instances_dev, c_args)

    if not os.path.exists(c_args["model_dir"]):
        os.makedirs(c_args["model_dir"])

    # build model
    model = MG_GTS(general_embedding, domain_embedding, c_args).to(c_args["device"])

    parameters = list(model.parameters())
    parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = torch.optim.Adam(parameters, lr=c_args["lr"], weight_decay=c_args["weight_decay"])
    reset_params(c_args, model)

    # training
    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(c_args["c0_epochs"]):
        print('Epoch:{}'.format(i))
        for j in trange(trainset.batch_count):
            _, sentence_tokens, lengths, masks, aspect_tags, _, tags = trainset.get_batch(j)
            predictions = model(sentence_tokens, lengths, masks)

            loss = 0.
            tags_flatten = tags[:, :lengths[0], :lengths[0]].reshape([-1])
            prediction_flatten = predictions.reshape([-1, predictions.shape[3]])
            loss = F.cross_entropy(prediction_flatten, tags_flatten, ignore_index=-1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        joint_precision, joint_recall, joint_f1 = classifier_eval(model, devset, c_args)

        if joint_f1 > best_joint_f1:
            # model_path = c_args["model_dir"] + c_args["model"] + c_args["task"] + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, c_args["task"], best_joint_f1))
    model_path = c_args["model_dir"] + c_args["model"] + c_args["task"] + '.pt'
    model = torch.load(model_path).to(c_args["device"]) # 重新加载最佳的模型到此
    return model


def train_G0(g_args, c_args, classifier, ppl_eval_field):
    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='%s/train_log' % working_dir,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Reading data ...')
    config_class, model_class, tokenizer_class, query_hinted_model_class = MODEL_CLASSES[g_args['model']['model_type']]
    config = config_class.from_pretrained(g_args['model']['model_name_or_path'],
                                          cache_dir=g_args['model']['cache_dir'])
    logging.info("Loading tokenizer from pretrained, {}".format(g_args['model']['model_name_or_path']))
    tokenizer = tokenizer_class.from_pretrained(g_args['model']['model_name_or_path'],
                                                cache_dir=g_args['model']['cache_dir'])
    logging.info('...done!')
    word2index = json.load(open(c_args["prefix"] + 'doubleembedding/word2index.json'))

    if g_args['model']['block_size'] <= 0:
        g_args['model']['block_size'] = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        g_args['model']['block_size'] = min(g_args['model']['block_size'], tokenizer.max_len)

    logging.info("Loading language model from pretrained, {}".format(g_args['model']['model_name_or_path']))
    if g_args['model']['model_name_or_path'] and ('gpt2' in g_args['model']['model_name_or_path']):
        pretrained_model = model_class.from_pretrained(
            g_args['model']['model_name_or_path'],
            config=config,
            cache_dir=g_args['model']['cache_dir'],
            output_meanvars=True,
            compute_meanvars_before_layernorm=g_args['model']['compute_meanvars_before_layernorm']
        )
        model = query_hinted_model_class(config, output_meanvars=True,
                                         compute_meanvars_before_layernorm=g_args['model']['compute_meanvars_before_layernorm'])
        pretrained_dict = pretrained_model.state_dict()
        query_hinted_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in query_hinted_dict}
        query_hinted_dict.update(pretrained_dict)
        model.load_state_dict(query_hinted_dict)
        pretrained_model = None
        del pretrained_model
    else:
        raise AssertionError

    model.to(g_args['data']['device'])

    if not g_args["training"]["only_lm"]:
        if g_args["data"]["INPUT_SENTI_TUPLE"] == 'pairs':
            # load unlabeled data
            half_labeled_sentence_packs = json.load(open(c_args["half_labeled_data_file"]))
            random.shuffle(half_labeled_sentence_packs)
            instances_half_labeled = load_data_instances(half_labeled_sentence_packs, word2index, c_args, labeled='Half')
            # random.shuffle(instances_half_labeled)
            half_labeled_set = DataIterator(instances_half_labeled, c_args)

            pseudo_opinion_data_path = c_args["pseudo_opinion_output_path"]
            if not os.path.exists(pseudo_opinion_data_path):
                os.mkdir(pseudo_opinion_data_path)
            cur_pseudo_opinion_data_file = pseudo_opinion_data_path + c_args["pseudo_opinion_output_filename"]
            if not os.path.exists(cur_pseudo_opinion_data_file):
                pseudo_data = classifier_generate_pseudo_opinion_span(classifier, half_labeled_set, c_args)
                # classifier generate pseudo label
                json.dump(pseudo_data, open(cur_pseudo_opinion_data_file, 'w', encoding='utf-8'))
            train_dataset = load_and_cache_examples(g_args, tokenizer, file_path=cur_pseudo_opinion_data_file,
                                                    evaluate=False, pseudo_data=True)
        else:
            train_dataset = load_and_cache_examples(g_args, tokenizer, evaluate=False, use_labeled_data=True)

        # assert train_dataset
        if g_args["training"]["num_lm_train_epochs"] > 0:
            logger.info("start gpt2 lm training")
            lm_global_step, lm_tr_loss, model = train_lm(g_args, train_dataset, model, tokenizer,
                                                         query_hinted_model_class, tokenizer_class,
                                                         g_args["training"]["num_lm_train_epochs"],
                                                         g_args["model"]["output_dir"],
                                                         g_args["model"]["lm_checkpoint_prefix"])  # 单训练LM
            logger.info("lm global_step = %s, average loss = %s", lm_global_step, lm_tr_loss)

        logger.info("start cat block training")
        if g_args["training"]["num_train_epochs"] > 0:
            global_step, tr_loss, model = train_hinted_gpt2(g_args, train_dataset, model, tokenizer,
                                                            g_args["model"]["output_dir"],
                                                            query_hinted_model_class, tokenizer_class,
                                                            num_train_epochs=g_args["training"]["num_train_epochs"],
                                                            checkpoint_prefix=g_args["model"]["checkpoint_prefix"],
                                                            model_config=config,
                                                            ppl_eval_field=ppl_eval_field,
                                                            transform_h_after_layernorm=g_args["training"][
                                                             "transform_h_after_layernorm"])
            logger.info("sentricon global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if g_args["training"]["do_train"] and g_args["data"]["local_rank"] == -1:
            # Create output directory if needed
            if g_args["data"]["local_rank"] in [-1, 0]:
                os.makedirs(g_args["model"]["output_dir"], exist_ok=True)

            logger.info("Saving model checkpoint to %s", g_args["model"]["output_dir"])
            if g_args["training"]["num_lm_train_epochs"] > 0:
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(g_args["model"]["output_dir"])
                tokenizer.save_pretrained(g_args["model"]["output_dir"])

                # Load a trained model and vocabulary that you have fine-tuned
                model = query_hinted_model_class.from_pretrained(g_args["model"]["output_dir"])
                tokenizer = tokenizer_class.from_pretrained(g_args["model"]["output_dir"])
                model.to(g_args['data']['device'])

            # Good practice: save your training arguments together with the trained model
            torch.save(g_args, os.path.join(g_args["model"]["output_dir"], "training_args.bin"))

    return model


def train_following_generations(c_args, g_args, classifier, generator,
                                tokenizer, max_generation=2, ppl_eval_field=None):
    word2index = json.load(open(c_args["prefix"] + 'doubleembedding/word2index.json'))
    general_embedding = np.load(c_args["prefix"] + 'doubleembedding/glove_filtered_300d.npy')
    general_embedding = torch.from_numpy(general_embedding)
    domain_embedding = np.load(c_args["prefix"] + 'doubleembedding/glove_filtered_300d.npy')
    domain_embedding = torch.from_numpy(domain_embedding)

    config_class, model_class, tokenizer_class, query_hinted_model_class = MODEL_CLASSES[g_args['model']['model_type']]
    # load unlabeled data
    unlabeled_sentence_packs = json.load(open(c_args["unlabeled_data_file"]))
    random.shuffle(unlabeled_sentence_packs)
    instances_unlabeled = load_data_instances(unlabeled_sentence_packs, word2index, c_args, labeled='False')
    random.shuffle(instances_unlabeled)
    unlabeled_set = DataIterator(instances_unlabeled, c_args)
    for gen in trange(1, max_generation+1):
        pseudo_data_path = c_args["pseudo_output_path"] + "C" + str(gen) + "/"
        if not os.path.exists(pseudo_data_path):
            os.mkdir(pseudo_data_path)
        cur_pseudo_data_file = pseudo_data_path+c_args["pseudo_output_filename"]
        if not os.path.exists(cur_pseudo_data_file):
            pseudo_data = classifier_generate_pseudo(classifier, unlabeled_set, c_args)
            # classifier generate pseudo label
            json.dump(pseudo_data, open(cur_pseudo_data_file, 'w', encoding='utf-8'))
        pseudo_dataset = load_and_cache_examples(g_args, tokenizer, file_path=cur_pseudo_data_file,
                                                 evaluate=False, pseudo_data=True)
        if g_args["following"]["train_lm_on_pseudo"] and g_args["following"]["num_lm_train_epochs"] >0:  # train language model with unlabeled data
            logger.info("start gpt2 lm training on pseudo")
            lm_global_step, lm_tr_loss, generator = train_lm(g_args, pseudo_dataset, generator, tokenizer,
                                                             query_hinted_model_class, tokenizer_class,
                                                             g_args["following"]["num_lm_train_epochs"],
                                                             g_args["following"]["saving_dir"],
                                                             g_args["model"]["lm_checkpoint_prefix"],
                                                             restart_from_latest_checkpoint=True)  # 单训练LM
            logger.info("lm global_step = %s, average loss = %s", lm_global_step, lm_tr_loss)
        if g_args['following']['num_train_epochs'] > 0:
            global_step, tr_loss, generator = train_hinted_gpt2(g_args, pseudo_dataset, generator, tokenizer,
                                                                g_args["following"]["saving_dir"],
                                                                query_hinted_model_class, tokenizer_class,
                                                                num_train_epochs=g_args["following"]["num_train_epochs"],
                                                                checkpoint_prefix=g_args["model"]["checkpoint_prefix"],
                                                                model_config=config,
                                                                ppl_eval_field=ppl_eval_field,
                                                                transform_h_after_layernorm=g_args["training"][
                                                                 "transform_h_after_layernorm"])
        if g_args["data"]["INPUT_SENTI_TUPLE"] == 'pairs':
            # load unlabeled data
            half_labeled_sentence_packs = json.load(open(c_args["half_labeled_data_file"]))
            random.shuffle(half_labeled_sentence_packs)
            instances_half_labeled = load_data_instances(half_labeled_sentence_packs, word2index, c_args, labeled='Half')
            random.shuffle(instances_half_labeled)
            half_labeled_set = DataIterator(instances_half_labeled, c_args)
            pseudo_opinion_data_path = c_args["pseudo_opinion_output_path"]
            if not os.path.exists(pseudo_opinion_data_path):
                os.mkdir(pseudo_opinion_data_path)
            cur_pseudo_opinion_data_file = pseudo_opinion_data_path + c_args["pseudo_opinion_output_filename"]
            if not os.path.exists(cur_pseudo_opinion_data_file):
                pseudo_data = classifier_generate_pseudo_opinion_span(classifier, half_labeled_set, c_args)
                # classifier generate pseudo label
                json.dump(pseudo_data, open(cur_pseudo_opinion_data_file, 'w', encoding='utf-8'))
            train_dataset = load_and_cache_examples(g_args, tokenizer, file_path=cur_pseudo_opinion_data_file,
                                                    evaluate=False, pseudo_data=True)
        else:
            train_dataset = load_and_cache_examples(g_args, tokenizer, evaluate=False, use_labeled_data=True)
        # train_dataset = load_and_cache_examples(g_args, tokenizer, evaluate=False, use_labeled_data=True)

        # assert train_dataset
        if g_args["G1"]["num_lm_train_epochs"] > 0:
            logger.info("start gpt2 lm training")
            lm_global_step, lm_tr_loss, generator = train_lm(g_args, train_dataset, generator, tokenizer,
                                                             query_hinted_model_class, tokenizer_class,
                                                             g_args["G1"]["num_lm_train_epochs"],
                                                             g_args["G1"]["saving_dir"],
                                                             g_args["model"]["lm_checkpoint_prefix"],
                                                             restart_from_latest_checkpoint=False)
            logger.info("lm global_step = %s, average loss = %s", lm_global_step, lm_tr_loss)

        if g_args["G1"]["num_train_epochs"] > 0:
            logger.info("start sentricon block training")
            global_step, tr_loss, generator = train_hinted_gpt2(g_args, train_dataset, generator, tokenizer,
                                                                g_args["G1"]["saving_dir"],
                                                                query_hinted_model_class, tokenizer_class,
                                                                num_train_epochs=g_args["G1"]["num_train_epochs"],
                                                                checkpoint_prefix=g_args["model"]["checkpoint_prefix"],
                                                                model_config=config,
                                                                ppl_eval_field=ppl_eval_field,
                                                                transform_h_after_layernorm=g_args["training"][
                                                                    "transform_h_after_layernorm"],
                                                                save_stops=True)
            logger.info("sentricon global_step = %s, average loss = %s", global_step, tr_loss)


    # generator load pseudo data

    # train generator using pseudo data

    # train classifier with reconstruction reward
    return classifier, generator


def load_c_pseudo_data(file_path):
    return None



def test_generator(g_args, generator_model, sentricon_block):
    return None


def evaluate_ppl(gen_args):
    device = torch.device("cuda" if torch.cuda.is_available() and not gen_args["data"]["no_cuda"] else "cpu")
    gen_args['data']['device'] = device
    set_seed(gen_args)
    config_class, model_class, tokenizer_class, query_hinted_model_class = MODEL_CLASSES[gen_args['model']['model_type']]
    config = config_class.from_pretrained(gen_args['model']['model_name_or_path'],
                                          cache_dir=gen_args['model']['cache_dir'])
    logging.info("Loading tokenizer from pretrained, {}".format(gen_args['model']['model_name_or_path']))
    hinted_generator_model = query_hinted_model_class.from_pretrained(gen_args["model"]["result_path"])
    tokenizer = tokenizer_class.from_pretrained(gen_args["model"]["result_path"])
    logging.info('...done!')
    # sentricon_block = SenTriConBlock(config.n_ctx, config, scale=True)
    # sentricon_block_weights_name = "sentricon_block_pytorch_model.bin"
    # output_sentricon_block_model_file = os.path.join(gen_args["model"]["result_path"],
    #                                                  sentricon_block_weights_name)
    # sentricon_state_dict = torch.load(output_sentricon_block_model_file)
    # sentricon_block.load_state_dict(sentricon_state_dict)

    hinted_generator_model.to(gen_args['data']['device'])
    # sentricon_block.to(gen_args['data']['device'])

    if gen_args['data']['gen_cs_len'] is None:
        gen_args['data']['gen_cs_len'] = gen_args['data']['cs_len']
    if gen_args['data']['gen_hs_len'] is None:
        gen_args['data']['gen_hs_len'] = gen_args['data']['hs_len']
    if gen_args['data']['gen_tis_len'] is None:
        gen_args['data']['gen_tis_len'] = gen_args['data']['tis_len']
    eval_dataset = load_and_cache_examples(gen_args, tokenizer,
                                           evaluate=True,
                                           generate=True)
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
        stce_sub_seg_position = [[indx for indx, tk in enumerate(stc) if tk == gen_args["data"]["SUB_TEXT_SEG_ID"]]
                                 for stc in stcs]
        if tokenizer._pad_token is None:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=gen_args["data"]["START_END_IDX"])
            senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                                                      stce_sub_seg=stce_sub_seg_position,
                                                                                      sentence_len=sentences.shape[1],
                                                                                      batch_first=True,
                                                                                      padding_value=gen_args["data"][
                                                                                          "START_END_IDX"])
            plrts = pad_sequence(plrts, batch_first=True)
        else:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=tokenizer.pad_token_id)
            senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                                                      stce_sub_seg=stce_sub_seg_position,
                                                                                      sentence_len=sentences.shape[1],
                                                                                      batch_first=True,
                                                                                      padding_value=tokenizer.pad_token_id, )
            plrts = pad_sequence(plrts, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_len = sentences.shape[0]
        s_len = sentences.shape[1]
        a_len = senti_inputs.shape[1]
        asp_pad_mask = torch.clone(senti_units_pad_mask)
        hint_matrix_padded_a = torch.zeros((b_len, s_len, a_len))
        hint_matrix_padded_o = torch.zeros((b_len, s_len, a_len))
        for idx in range(b_len):
            hint_matrix_padded_a[idx, :hint_matrix_a[idx].shape[0], :hint_matrix_a[idx].shape[1]] = hint_matrix_a[idx][
                                                                                                    :, :]
            hint_matrix_padded_o[idx, :hint_matrix_o[idx].shape[0], :hint_matrix_o[idx].shape[1]] = hint_matrix_o[idx][
                                                                                                    :, :]

            for jdx in range(len(aspts[idx])):
                asp_pad_mask[idx, jdx, len(aspts[idx][jdx]) - 1:] = 0  # B_A_T, -1 for 'start_token'
        hint_matrix_padded = torch.cat((hint_matrix_padded_a, hint_matrix_padded_o), dim=2)
        return sentences, sentiment_token_types, senti_inputs, senti_units_pad_mask, asp_pad_mask, aspts, opns, plrts, hint_matrix_padded

    # eval_sampler = RandomSampler(eval_dataset) if gen_args["data"]["local_rank"] == -1 else DistributedSampler(
    #     eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, #sampler=eval_sampler,
        batch_size=gen_args['training']['per_gpu_train_batch_size'], collate_fn=collate_P1
    )
    hinted_generator_model.eval()
    # sentricon_block.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating PPL..."):
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
            if gen_args["data"]["use_hint_matrix"]:
                hint_matrix = batch[8]
                hint_matrix = hint_matrix.to(gen_args['data']['device'])
            else:
                hint_matrix = None

            if inputs.shape[1] < gen_args['data']['hs_len']:
                # "inputs.shape[1] < arg_config['data']['hs_len'], skipping batch"
                continue
            # hs_len = gen_args['data']['hs_len']
            # tis_len = gen_args['data']['tis_len']
            # lm_labels = lm_labels[:, :hs_len + tis_len]
            # inputs = inputs.to(gen_args['data']['device'])
            # lm_labels = lm_labels.to(gen_args['data']['device'])

            original_context_seq = sentiment_inputs.to(gen_args['data']['device'])

            sentiment_inputs = sentiment_inputs.view(sentiment_inputs.shape[0], -1)  # B_A_T → B_AT
            inputs = torch.cat((sentiment_inputs, inputs), dim=1)  # B_(AT+S)
            lm_labels = torch.cat((sentiment_inputs, lm_labels), dim=1)  # B_(AT+S)
            # lm_labels = lm_labels[:, :hs_len + tis_len]
            inputs = inputs.to(gen_args['data']['device'])
            lm_labels = lm_labels.to(gen_args['data']['device'])

            lm_logit_first_index = sentiment_inputs.shape[1]  # AT
            lm_labels_first_index = lm_logit_first_index + 1
            outputs = hinted_generator_model(inputs, labels=lm_labels, query_hint_matrix=hint_matrix,
                                             hint_lbd=gen_args["data"]["hint_lambda"],
                                             aspect_attention_mask=aspect_pad_mask,
                                             context_attention_mask=senti_pad_mask,
                                             lm_logit_first_index=lm_logit_first_index,
                                             lm_labels_first_index=lm_labels_first_index)
            self_alsecond_lm_loss = outputs[0]

            eval_loss += self_alsecond_lm_loss.mean().item()
            # break
            # lm_outputs = generator_model(inputs, labels=inputs)
            # lm_loss = lm_outputs[0]
            # lm_eval_loss += lm_loss.mean().item()
    eval_loss = eval_loss / len(eval_dataloader)
    # lm_eval_loss = lm_eval_loss / len(eval_dataloader)
    perplexity = math.exp(eval_loss)
    # lm_perplexity = math.exp(lm_eval_loss)

    result = {"perplexity_aocon": perplexity}
    output_eval_file = os.path.join(gen_args["model"]["result_path"], gen_args['evaluate']['ppl_output_file'])
    json.dump(result, open(output_eval_file, 'w'))

    return result

def evaluate_scores(evaluate_args):
    # Evaluate with BLEU
    generated_fn = evaluate_args["model"]["result_path"] + '/' + evaluate_args["evaluate"]["sentricon_output_jsonl_filename"]
    bleu1, bleu2, bleu3, bleu4 = evaluate_BLEU(generated_fn)
    bleu_dict = {"bleu_1": bleu1, "bleu_2": bleu2, "bleu_3": bleu3, "bleu_4": bleu4,
                 # "bleu_gpt_1": bleu_gpt1, "bleu_gpt_2": bleu_gpt2, "bleu_gpt_3": bleu_gpt3, "bleu_gpt_4": bleu_gpt4
                 }
    json.dump(bleu_dict, open(evaluate_args["model"]["result_path"] + '/' + evaluate_args["evaluate"]["bleu_output_file"], 'w'))

    # Evaluate with NIST and METEOR
    nist_sentricon, nist_gpt2, meteor_sentricon, meteor_gpt2, rouge_sentricon, rouge_gpt2 = evaluate_NIST_AND_METEOR_AND_ROUGE(
        generated_fn)
    nist_dict = {"nist_4": nist_sentricon, "nist_gpt_4": nist_gpt2}
    meteor_dict = {"meteor_4": meteor_sentricon, "meteor_gpt_4": meteor_gpt2}
    rouge_dict = {"rouge-l": rouge_sentricon, "rouge-l-gpt-2": rouge_gpt2}
    json.dump(nist_dict, open(evaluate_args["model"]["result_path"] + '/' + evaluate_args["evaluate"]["nist_output_file"], 'w'))
    json.dump(meteor_dict, open(evaluate_args["model"]["result_path"] + '/' + evaluate_args["evaluate"]["meteor_output_file"], 'w'))
    json.dump(rouge_dict, open(evaluate_args["model"]["result_path"] + '/' + evaluate_args["evaluate"]["rouge_output_file"], 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(  # 配置参数
        "--generator_config",
        help="path to generator json config",
        default="data/generator_config.json"
    )
    parser.add_argument(
        "--classifier_config",
        help="path to classifier json config",
        default="classifier/data/classifier_config.json"
    )
    args = parser.parse_args()
    g_args = json.load(open(args.generator_config, 'r'))  # 加载参数
    c_args = json.load(open(args.classifier_config, 'r'))

    working_dir = g_args['data']['working_dir']  #
    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    g_config_path = os.path.join(working_dir, 'generator_config.json')
    c_config_path = os.path.join(working_dir, 'classifier_config.json')
    if not os.path.exists(g_config_path):
        with open(g_config_path, 'w') as f:
            json.dump(g_args, f)
    if not os.path.exists(c_config_path):
        with open(c_config_path, 'w') as f:
            json.dump(c_args, f)
    if g_args["data"]["local_rank"] == -1 or g_args["data"]["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not g_args["data"]["no_cuda"] else "cpu")
        g_args['data']['n_gpu'] = torch.cuda.device_count()
        # args.n_gpu =
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(g_args["data"]["local_rank"])
        device = torch.device("cuda", g_args["data"]["local_rank"])
        torch.distributed.init_process_group(backend="nccl")
        g_args['data']['n_gpu'] = 1
    device = torch.device("cuda" if torch.cuda.is_available() and not g_args["data"]["no_cuda"] else "cpu")
    g_args['data']['device'] = device
    c_args['device'] = device
    g_args['training']['train_batch_size'] = g_args['training']['per_gpu_train_batch_size'] * max(1, g_args['data']['n_gpu'])

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    c_args['initializer'] = initializers[c_args['initializer']]
    set_seed(g_args)

    config_class, model_class, tokenizer_class, hinted_model_class = MODEL_CLASSES[g_args['model']['model_type']]
    config = config_class.from_pretrained(g_args['model']['model_name_or_path'],
                                          cache_dir=g_args['model']['cache_dir'])
    logging.info("Loading tokenizer from pretrained, {}".format(g_args['model']['model_name_or_path']))
    tokenizer = tokenizer_class.from_pretrained(g_args['model']['model_name_or_path'],
                                                cache_dir=g_args['model']['cache_dir'])
    logging.info('...done!')
    ########################
    # train_dataset = load_and_cache_examples(g_args, tokenizer, evaluate=False, use_labeled_data=True)
    ppl_eval_field = PPLEvalField(g_args=g_args, tokenizer=tokenizer)
    ########################
    classify_model = train_C0(c_args)
    # generator_model, senti_block = None, None
    generator_model = train_G0(g_args=g_args, c_args=c_args, classifier=classify_model,
                                            ppl_eval_field=ppl_eval_field)

    classify_model, generator_model = train_following_generations(c_args=c_args, g_args=g_args,
                                                                  classifier=classify_model,
                                                                  generator=generator_model,
                                                                  tokenizer=tokenizer,
                                                                  max_generation=g_args["following"]["max_generation"],
                                                                  ppl_eval_field=ppl_eval_field)
    # test_result = test_generator(g_args, generator_model=generator_model, sentricon_block=senti_block)
    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if g_args["training"]["do_train"] and g_args["data"]["local_rank"] == -1:
        # Create output directory if needed
        if g_args["data"]["local_rank"] in [-1, 0]:
            os.makedirs(g_args["model"]["result_path"], exist_ok=True)

        logger.info("Saving model checkpoint to %s", g_args["model"]["result_path"])
        if g_args["training"]["num_lm_train_epochs"] > 0:
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                generator_model.module if hasattr(generator_model, "module") else generator_model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(g_args["model"]["result_path"])
            tokenizer.save_pretrained(g_args["model"]["result_path"])

            # Load a trained model and vocabulary that you have fine-tuned
            generator_model = hinted_model_class.from_pretrained(g_args["model"]["result_path"])
            tokenizer = tokenizer_class.from_pretrained(g_args["model"]["result_path"])
            generator_model.to(g_args['data']['device'])

        # Good practice: save your training arguments together with the trained model
        torch.save(g_args, os.path.join(g_args["model"]["result_path"], "training_args.bin"))