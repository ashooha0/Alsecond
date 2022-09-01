import argparse
import json
import numpy as np
import logging
import os
import random
import torch
import glob

from model.loss_function.ConfidencedLoss import ConfidenceSentimentControlLoss
from model.loss_function.SentimentControlLoss import SentimentControlLoss, SentimentReinforceControlLoss
from model.transformers.modeling_gpt2 import QueryHintedGPT2LMHeadModel
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
from utils.data_util import load_and_cache_examples, generate_alsecond_compute, constract_batch_mask
from utils.train_util import evaluate, _clear_checkpoints, _rotate_checkpoints, _latest_checkpoint

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, QueryHintedGPT2LMHeadModel),
}
logger = logging.getLogger(__name__)

def fix_state_dict_naming(state_dict):
    return state_dict

def set_seed(arg_config):
    random.seed(arg_config['data']['random_seed'])
    np.random.seed(arg_config['data']['random_seed'])
    torch.manual_seed(arg_config['data']['random_seed'])
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(arg_config['data']['random_seed'])


def train_lm(arg_config, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
             model_class, tokenizer_class,num_lm_train_epochs,
             saving_dir, checkpoint_prefix, para_output_fn="parameters.pt",
             restart_from_latest_checkpoint=False) -> Tuple[int, float, PreTrainedModel]:
    """ Train the model """

    tb_writer = SummaryWriter()

    if arg_config['training']['per_gpu_train_lm_batch_size'] <= 0:
        arg_config['training']['per_gpu_train_lm_batch_size'] = arg_config['training']['per_gpu_train_batch_size']
    arg_config['training']['train_lm_batch_size'] =\
        arg_config['training']['per_gpu_train_lm_batch_size'] * max(1, arg_config['data']['n_gpu'])

    def collate(examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        def pad_chaos(data, batch_first=True, padding_value=arg_config["data"]["START_END_IDX"]):  # pad B * A * T(batch, aspect/opinion, token)
            second_lens = [len(tup) for tup in data]
            data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]
            data = [torch.tensor(tks, dtype=torch.long) for tup in data for tks in tup]
            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)
            data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]
            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)
            return data
        sentences = [torch.tensor(st[0], dtype=torch.long) for st in examples]
        # aspects = [st[1] for st in examples]
        # opinions = [st[2] for st in examples]
        polarities = [torch.tensor(st[4], dtype=torch.long) for st in examples]
        if tokenizer._pad_token is None:
            sentences = pad_sequence(sentences, batch_first=True, padding_value=arg_config["data"]["START_END_IDX"])
            # aspects = pad_chaos(aspects, batch_first=True)
            # opinions = pad_chaos(opinions, batch_first=True)
            polarities = pad_sequence(polarities, batch_first=True)
        else:
            sentences = pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id)
            # aspects = pad_chaos(aspects, batch_first=True, padding_value=tokenizer.pad_token_id)
            # opinions = pad_chaos(opinions, batch_first=True, padding_value=tokenizer.pad_token_id)
            polarities = pad_sequence(polarities, batch_first=True, padding_value=tokenizer.pad_token_id)
        return sentences, polarities   # , aspects, opinions, polarities


    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=arg_config["data"]["batch_size"],
        collate_fn=collate
    )

    if arg_config["training"]["lm_max_steps"] > 0:
        t_total = arg_config["training"]["lm_max_steps"]
        num_lm_train_epochs =\
            arg_config["training"]["lm_max_steps"] // (len(train_dataloader) // arg_config["training"]["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) // arg_config["training"]["gradient_accumulation_steps"] * num_lm_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": arg_config["training"]["weight_decay"]
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=arg_config["training"]["learning_rate"],
                      eps=arg_config["training"]["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=arg_config["training"]["warmup_steps"], num_training_steps=t_total
    )

    # Train
    logger.info("***** Running LM training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_lm_train_epochs)
    # logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        arg_config['training']['train_lm_batch_size']
        * arg_config["training"]["gradient_accumulation_steps"]
        * (torch.distributed.get_world_size() if arg_config["data"]["local_rank"] != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", arg_config["training"]["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_global = 0

    # check if need to reload from saved optimizer, scheduler and block states and global step
    if restart_from_latest_checkpoint and _latest_checkpoint(saving_dir, checkpoint_prefix):
        # Load in optimizer and scheduler states from latest checkpoint
        latest_checkpoint_pth = _latest_checkpoint(saving_dir, checkpoint_prefix)
        logger.info("Reload LM from latest trained: ", latest_checkpoint_pth)
        latest_checkpoint = torch.load(os.path.join(latest_checkpoint_pth, para_output_fn))

        # load optimizer and scheduler and sentricon weight and global step
        optimizer.load_state_dict(latest_checkpoint["optimizer"])
        scheduler.load_state_dict(latest_checkpoint["scheduler"])
        steps_trained_global = latest_checkpoint["step"]
        global_step = steps_trained_global
        print("Now skip {} steps...".format(steps_trained_global))
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(latest_checkpoint_pth)
        tokenizer = tokenizer_class.from_pretrained(latest_checkpoint_pth)
        model.to(arg_config['data']['device'])

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_lm_train_epochs), desc="Epoch",
        disable=arg_config["data"]["local_rank"] not in [-1, 0]
    )
    set_seed(arg_config)  # Added here for reproducibility
    first_save = True
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=arg_config["data"]["local_rank"] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_global > 0:
                steps_trained_global -= 1
                continue

            inputs, labels = (batch[0], batch[0])
            inputs = inputs.to(arg_config['data']['device'])
            labels = labels.to(arg_config['data']['device'])
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if arg_config['model']['output_meanvars']:
                all_meanvars = outputs[-1]
                all_meanvars_tensor = []

                for block_ind, meanvars_in_block in enumerate(all_meanvars):
                    for layer_ind, meanvars_in_layer in enumerate(meanvars_in_block):
                        for stats_ind, stats in enumerate(meanvars_in_layer): # stats.shape: [batch_size, n_embd], mean & var
                            all_meanvars_tensor.append(stats)

                all_meanvars = torch.stack(all_meanvars_tensor, dim=1)

            if arg_config['data']['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if arg_config["training"]["gradient_accumulation_steps"] > 1:
                loss = loss / arg_config["training"]["gradient_accumulation_steps"]

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % arg_config["training"]["gradient_accumulation_steps"] == 0:

                torch.nn.utils.clip_grad_norm_(model.parameters(), arg_config["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (arg_config["data"]["local_rank"] in [-1, 0] and
                        arg_config["data"]["logging_steps"] > 0 and
                        global_step % arg_config["data"]["logging_steps"] == 0):
                    # Log metrics
                    if (
                        arg_config["data"]["local_rank"] == -1 and arg_config["training"]["evaluate_during_training"]
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(arg_config, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("LM/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("LM/loss", (tr_loss - logging_loss) / arg_config["data"]["logging_steps"], global_step)
                    logging_loss = tr_loss

            if arg_config["training"]["save_steps"] > 0\
                    and (global_step+1) % arg_config["training"]["save_steps"] == 0:

                if first_save:
                    _clear_checkpoints(saving_dir, checkpoint_prefix)
                    first_save = False
                # Save model checkpoint
                output_dir = os.path.join(saving_dir,
                                          "{}-{}".format(checkpoint_prefix, global_step+1))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                _rotate_checkpoints(saving_dir, arg_config["data"]["save_total_limit"],
                                    checkpoint_prefix)
                logger.info("Saving LM model checkpoint to %s", output_dir)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                checkpoint = {"step": global_step+1,
                              "arg_config": arg_config,
                              "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(),
                              }
                torch.save(checkpoint, os.path.join(output_dir, para_output_fn))
                del checkpoint
                logger.info("Saving LM optimizer and scheduler states to %s", output_dir)

            if arg_config["training"]["lm_max_steps"] > 0 and global_step > arg_config["training"]["lm_max_steps"]:
                epoch_iterator.close()
                break
        if arg_config["training"]["lm_max_steps"] > 0 and global_step > arg_config["training"]["lm_max_steps"]:
            train_iterator.close()
            break

    if arg_config["data"]["local_rank"] in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, model


def train_hinted_gpt2(arg_config, train_dataset, model, tokenizer, saving_dir,
                      hinted_model_class, tokenizer_class,
                      num_train_epochs, checkpoint_prefix, model_config=None, para_output_fn="parameters.pt",
                      ppl_eval_field=None, transform_h_after_layernorm=False,
                      save_stops=False):
    """ Train the model """
    if arg_config["data"]["local_rank"] in [-1, 0]:
        tb_log_dir = os.path.join(saving_dir, 'runs')
        tb_writer = SummaryWriter(tb_log_dir)

    arg_config['training']['train_batch_size'] =\
        arg_config['training']['per_gpu_train_batch_size'] * max(1, arg_config['data']['n_gpu'])

    # check if max/min_hs_tis_split_offset is out of range of hs_len or tis_len
    offset_hs_tis_split = False
    if arg_config['data']['min_hs_tis_split_offset'] != 0:
        offset_hs_tis_split = True
        if (arg_config['data']['min_hs_tis_split_offset'] + arg_config['data']['hs_len'] < 0):
            raise ValueError(
                "min_hs_tis_split_offset is out of bound"
            )
    if arg_config['data']['max_hs_tis_split_offset'] != 0:
        offset_hs_tis_split = True
        if (min(arg_config['data']['cs_len'],
                arg_config['data']['tis_len']) - arg_config['data']['max_hs_tis_split_offset'] < 0):
            raise ValueError(
                "max_hs_tis_split_offset is out of bound"
            )

    def collate_P1(examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        def pad_sentiment(sentiment_units, stce_sub_seg=None, sentence_len=None, start_p=1, end_p=None,  # start_cut, end_cut stands for start and end length
                          batch_first=True, padding_value=arg_config["data"]["START_END_IDX"],
                          senti_token_type_id=arg_config["data"]["senti_type_id"]):  # pad B * A * T(batch, aspect/opinion, token)
            second_lens = [len(unit) for unit in sentiment_units]
            data_cut_point = [sum(second_lens[:i]) for i in range(0, len(second_lens) + 1)]

            data = [[senti[start_p: end_p] for senti in unit] for unit in sentiment_units]  # B_A'_T'
            senti_unit_pad_mask = constract_batch_mask(data)
            type_ids = torch.full_like(senti_unit_pad_mask, senti_token_type_id)
            type_ids = (type_ids * senti_unit_pad_mask).long()
            # batch_mask = constract_batch_mask(data, stce_sub_seg, sentence_len=sentence_len)
            data = [torch.tensor(d) for da in data for d in da]  # BA'_T'

            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # BA'_T
            data = [data[data_cut_point[i]:data_cut_point[i + 1]] for i in range(0, len(data_cut_point) - 1)]
            data = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)  # B_A_T
            # assert batch_mask.shape[-2:] == data.shape[-2:]
            return data, senti_unit_pad_mask, type_ids  # , batch_mask   # senti_pad_mask: B_A_T batch_mask: B_S_A_T
        stcs = [torch.tensor(st[0], dtype=torch.long) for st in examples]
        senti_units = [st[1] for st in examples]
        aspts = [st[2] for st in examples]
        opns = [st[3] for st in examples]
        plrts = [torch.tensor(st[4], dtype=torch.long) for st in examples]
        hint_matrix_a = [st[5] for st in examples]
        hint_matrix_o = [st[6] for st in examples]
        posi_senti_mask = [st[7] for st in examples]
        # joint_of_aspect_opinion = arg_config["data"]["joint_of_aspect_opinion"]  # defult: "--------"
        stce_sub_seg_position = [[indx for indx, tk in enumerate(stc) if tk == arg_config["data"]["SUB_TEXT_SEG_ID"]]
                                 for stc in stcs]

        if tokenizer._pad_token is None:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=arg_config["data"]["START_END_IDX"])
            senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                             stce_sub_seg=stce_sub_seg_position,
                                                             sentence_len=sentences.shape[1],
                                                             batch_first=True,
                                                             padding_value=arg_config["data"]["START_END_IDX"])
            plrts = pad_sequence(plrts, batch_first=True)
        else:
            sentences = pad_sequence(stcs, batch_first=True, padding_value=tokenizer.pad_token_id)
            senti_inputs, senti_units_pad_mask, sentiment_token_types = pad_sentiment(sentiment_units=senti_units,
                                                             stce_sub_seg=stce_sub_seg_position,
                                                             sentence_len=sentences.shape[1],
                                                             batch_first=True,
                                                             padding_value=tokenizer.pad_token_id,)
            plrts = pad_sequence(plrts, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_len = sentences.shape[0]
        s_len = sentences.shape[1]
        a_len = senti_inputs.shape[1]
        asp_pad_mask = torch.clone(senti_units_pad_mask)
        hint_matrix_padded_a = torch.zeros((b_len, s_len, a_len))
        hint_matrix_padded_o = torch.zeros((b_len, s_len, a_len))
        posi_senti_mask_padded = torch.zeros((b_len, s_len))
        for idx in range(b_len):
            hint_matrix_padded_a[idx, :hint_matrix_a[idx].shape[0], :hint_matrix_a[idx].shape[1]] = hint_matrix_a[idx][:, :]
            hint_matrix_padded_o[idx, :hint_matrix_o[idx].shape[0], :hint_matrix_o[idx].shape[1]] = hint_matrix_o[idx][:, :]
            posi_senti_mask_padded[idx, :posi_senti_mask[idx].shape[0]] = posi_senti_mask[idx][:]
            if arg_config["data"]["INPUT_SENTI_TUPLE"] == 'triples' or arg_config["data"]["HINT_OPINIONS"]:
                for jdx in range(len(aspts[idx])):
                    asp_pad_mask[idx, jdx, len(aspts[idx][jdx])-1:] = 0  # B_A_T, -1 for 'start_token'
        hint_matrix_padded = torch.cat((hint_matrix_padded_a, hint_matrix_padded_o), dim=2)
        return sentences, sentiment_token_types, senti_inputs, senti_units_pad_mask, asp_pad_mask, aspts, opns, plrts, hint_matrix_padded, posi_senti_mask_padded

    train_sampler = RandomSampler(train_dataset) if arg_config["data"]["local_rank"] == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler,
        batch_size=arg_config['training']['train_batch_size'], collate_fn=collate_P1
    )

    if arg_config['training']['max_steps'] > 0:
        t_total = arg_config['training']['max_steps']
        num_train_epochs =\
            arg_config['training']['max_steps'] //\
            (len(train_dataloader) // arg_config["training"]["gradient_accumulation_steps"]) + 1
    else:
        t_total = len(train_dataloader) //\
                  arg_config["training"]["gradient_accumulation_steps"] * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay) for lm model
    # GPT2
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": arg_config["training"]["weight_decay"],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=arg_config["training"]["learning_rate"],
                      eps=arg_config["training"]["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=arg_config["training"]["warmup_steps"],
        num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if arg_config['data']['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
        # if args.lambda_adv > 0:
        #     disc_model = torch.nn.DataParallel(disc_model)

    # Distributed training (should be after apex fp16 initialization)
    if arg_config["data"]["local_rank"] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[arg_config["data"]["local_rank"]], output_device=arg_config["data"]["local_rank"], find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", arg_config['training']['per_gpu_train_batch_size'])
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        arg_config['training']['train_batch_size']
        * arg_config["training"]["gradient_accumulation_steps"]
        * (torch.distributed.get_world_size() if arg_config["data"]["local_rank"] != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", arg_config["training"]["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_global = 0

    tr_loss, logging_loss = 0.0, 0.0
    disc_loss, logging_disc_loss = 0.0, 0.0

    # check if need to reload from saved optimizer, scheduler and block states and global step
    if (arg_config["training"]["restart_from_latest_checkpoint"]
            and _latest_checkpoint(saving_dir,
                                   checkpoint_prefix)):
        # Load in optimizer and scheduler states from latest checkpoint
        latest_checkpoint_pth = _latest_checkpoint(saving_dir,
                                                   checkpoint_prefix)
        logger.info("Reload from latest trained: ", latest_checkpoint_pth)
        latest_checkpoint = torch.load(os.path.join(latest_checkpoint_pth, para_output_fn))

        # load optimizer and scheduler and sentricon weight and global step
        optimizer.load_state_dict(latest_checkpoint["optimizer"])
        scheduler.load_state_dict(latest_checkpoint["scheduler"])
        steps_trained_global = latest_checkpoint["step"]
        global_step = steps_trained_global
        print("Now skip {} steps...".format(steps_trained_global))
        model = hinted_model_class.from_pretrained(latest_checkpoint_pth)
        tokenizer = tokenizer_class.from_pretrained(latest_checkpoint_pth)
        model.to(arg_config['data']['device'])

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_train_epochs), desc="Epoch",
        disable=arg_config["data"]["local_rank"] not in [-1, 0]
    )
    set_seed(arg_config)  # Added here for reproducibility

    first_save = True
    sentiment_control_loss = SentimentControlLoss(mask_mean=arg_config["data"]["ctrl_mask_mean"])
    sentiment_reinforce_control_loss = SentimentReinforceControlLoss()
    for epoch_ind in train_iterator:
        logger.info("epoch_ind: {}".format(epoch_ind))

        if arg_config['training']['lambda_alsecond_recon_lm_loss'] > 0 \
                and arg_config['training']['per_gpu_train_alsecond_recon_batch_size'] is not None\
                and epoch_ind == arg_config['training']['epoch_ind_to_start_alsecond_recon']:
            arg_config['training']['train_alsecond_recon_batch_size'] =\
                arg_config['training']['per_gpu_train_alsecond_recon_batch_size']\
                * max(1, arg_config['data']['n_gpu'])
            logger.info("Changing train_batch_size to {} due to start_alsecond_recon".format(
                arg_config['training']['train_alsecond_recon_batch_size']))

            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler,
                batch_size=arg_config['training']['train_alsecond_recon_batch_size'],
                collate_fn=collate_P1
            )

        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=arg_config["data"]["local_rank"] not in [-1, 0])
        aspect_losses = []
        positional_sentiment_losses = []
        generation_losses = []
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_global > 0:
                steps_trained_global -= 1
                continue

            inputs, lm_labels = (batch[0], batch[0])
            token_type_ids = batch[1]
            token_type_ids = token_type_ids.to(arg_config['data']['device'])
            sentiment_inputs = batch[2]
            if arg_config['data']['use_attn_mask']:
                senti_pad_mask = batch[3]
                aspect_pad_mask = batch[4]
                # senti_btc_mask = batch[4]
                # senti_btc_mask = senti_btc_mask.to(arg_config['data']['device'])
                senti_pad_mask = senti_pad_mask.to(arg_config['data']['device'])
                aspect_pad_mask = aspect_pad_mask.to(arg_config['data']['device'])
            else:
                # senti_btc_mask = None
                senti_pad_mask = None
                aspect_pad_mask = None
            aspects = batch[5]
            opinions = batch[6]
            polarities = batch[7]
            if arg_config["data"]["use_hint_matrix"]:
                hint_matrix = batch[8]
                hint_matrix = hint_matrix.to(arg_config['data']['device'])
            else:
                hint_matrix = None
            reinforce_control_masks = batch[9]
            reinforce_control_masks = reinforce_control_masks.to(arg_config['data']['device'])

            # Skip batch if seq len is shorter than hs_len, i.e. no tis or cs text
            if inputs.shape[1] < arg_config['data']['hs_len']:
                logger.info("inputs.shape[1] < arg_config['data']['hs_len'], skipping batch")
                continue

            sentiment_inputs = sentiment_inputs.view(sentiment_inputs.shape[0], -1)  # B_A_T → B_AT
            inputs = torch.cat((sentiment_inputs, inputs), dim=1)  # B_(AT+S)
            lm_labels = torch.cat((sentiment_inputs, lm_labels), dim=1)  # B_(AT+S)
            # lm_labels = lm_labels[:, :hs_len + tis_len]
            inputs = inputs.to(arg_config['data']['device'])
            lm_labels = lm_labels.to(arg_config['data']['device'])

            model.train()
            lm_logit_first_index = sentiment_inputs.shape[1]  # AT
            lm_labels_first_index = lm_logit_first_index + 1
            outputs = model(inputs, labels=lm_labels, query_hint_matrix=hint_matrix,
                            hint_lbd=arg_config["data"]["hint_lambda"],
                            aspect_attention_mask=aspect_pad_mask,
                            context_attention_mask=senti_pad_mask,
                            lm_logit_first_index=lm_logit_first_index,
                            lm_labels_first_index=lm_labels_first_index)
            self_lm_loss = outputs[0]
            if arg_config['model']['output_meanvars']:
                all_meanvars = outputs[-1]
                all_meanvars_tensor = []

                for block_ind, meanvars_in_block in enumerate(all_meanvars):
                    for layer_ind, meanvars_in_layer in enumerate(meanvars_in_block):
                        for stats_ind, stats in enumerate(
                                meanvars_in_layer):  # stats.shape: [batch_size, n_embd], mean & var
                            all_meanvars_tensor.append(stats)

                all_meanvars = torch.stack(all_meanvars_tensor, dim=1)
            generation_losses.append(self_lm_loss.item())
            if arg_config["training"]["lambda_alsecond_lm_loss"] > 0:
                total_loss = arg_config["training"]["lambda_alsecond_lm_loss"] * self_lm_loss
            else:
                total_loss = 0
            if arg_config["training"]["lambda_aspects_control_loss"] > 0:
                aspects_control_loss = sentiment_control_loss(arg_config,
                                                              lm_labels[:, :],
                                                              outputs[1][:, lm_logit_first_index:-1],
                                                              sentiment_atoms=aspects)
            if arg_config["training"]["lambda_opinions_control_loss"] > 0:
                opinions_control_loss = sentiment_control_loss(arg_config,
                                                               lm_labels[:, :],
                                                               outputs[1][:, lm_logit_first_index:-1],
                                                               sentiment_atoms=opinions)
            if arg_config["training"]["lambda_positional_sentiment_control_loss"] > 0:
                positional_reinforce_loss = sentiment_reinforce_control_loss(lm_logit_first_index=lm_logit_first_index,
                                                                             lm_labels_first_index=lm_labels_first_index,
                                                                             input_labels=lm_labels[:, :],
                                                                             generation_logits=outputs[1][:, :-1],
                                                                             reinforce_control_masks=reinforce_control_masks
                                                                             )
            if arg_config["training"]["lambda_aspects_control_loss"] > 0:
                aspect_losses.append(aspects_control_loss.item())

                total_loss += arg_config["training"]["lambda_aspects_control_loss"] * aspects_control_loss

            if arg_config["training"]["lambda_opinions_control_loss"] > 0:
                total_loss += arg_config["training"]["lambda_opinions_control_loss"] * opinions_control_loss
            if arg_config["training"]["lambda_positional_sentiment_control_loss"] > 0:
                positional_sentiment_losses.append(positional_reinforce_loss.item())
                total_loss += arg_config["training"]["lambda_positional_sentiment_control_loss"] * positional_reinforce_loss

            if arg_config['data']['n_gpu'] > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
            if arg_config["training"]["gradient_accumulation_steps"] > 1:
                total_loss = total_loss / arg_config["training"]["gradient_accumulation_steps"]

            total_loss.backward()
            tr_loss += total_loss.item()

            if (step + 1) % arg_config["training"]["gradient_accumulation_steps"] == 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), arg_config["training"]["max_grad_norm"])
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # global_step += 1

                if (arg_config["data"]["local_rank"] in [-1, 0] and
                        arg_config["data"]["logging_steps"] > 0 and
                        global_step % arg_config["data"]["logging_steps"] == 0):
                    # Log metrics
                    if (
                            arg_config["data"]["local_rank"] == -1 and arg_config["training"][
                        "evaluate_during_training"]
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(arg_config, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("LM/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("LM/loss", (tr_loss - logging_loss) / arg_config["data"]["logging_steps"],
                                         global_step)
                    logging_loss = tr_loss

            # Save model
            if arg_config["training"]["save_steps"] > 0 \
                    and (global_step + 1) % arg_config["training"]["save_steps"] == 0:
                if first_save:
                    _clear_checkpoints(saving_dir, checkpoint_prefix)
                    first_save = False

                # Save model checkpoint
                output_dir = os.path.join(saving_dir,
                                          "{}-{}".format(checkpoint_prefix, (global_step + 1)))
                # print("output_dir is : ", output_dir)
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                _rotate_checkpoints(saving_dir, arg_config["data"]["save_total_limit"],
                                    checkpoint_prefix)
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                checkpoint = {"step": global_step + 1,
                              "arg_config": arg_config,
                              "optimizer": optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(),
                              }
                torch.save(checkpoint, os.path.join(output_dir, para_output_fn))
                del checkpoint
            if save_stops and arg_config["G1"]["saving_stop_steps"] > 0 and \
                    (global_step + 1) % arg_config["G1"]["saving_stop_steps"] == 0:
                # Save model stop
                output_dir = os.path.join(saving_dir,
                                          "stop-{}".format(global_step + 1))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                _rotate_checkpoints(saving_dir, 30, "stop")
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
            global_step += 1

            if (arg_config['training']['max_steps'] > 0 and global_step > arg_config['training']['max_steps']) or (
                    arg_config['training']['epoch_max_steps'] > 0 and step > arg_config['training']['epoch_max_steps']):
                epoch_iterator.close()
                break
        if arg_config['data']['eval_ppl_save_per_n_epoch'] > 0 and\
                epoch_ind % arg_config['data']['eval_ppl_save_per_n_epoch'] == 0:
            ppl_eval_field.eval_2_save(arg_config, arg_config["model"]["best_save_dir"], model)
        if arg_config['training']['max_steps'] > 0 and global_step > arg_config['training']['max_steps']:
            train_iterator.close()
            break
        if len(generation_losses) > 0:
            print("generating_loss: max:{}, mean:{}\n".format(max(generation_losses),
                                                              sum(generation_losses)/len(generation_losses)))
        if len(aspect_losses) > 0:
            print("aspects_control_loss: max:{}, mean:{}".format(max(aspect_losses),
                                                                 sum(aspect_losses)/len(aspect_losses)))
        if len(positional_sentiment_losses) > 0:
            print("positional_sentiment_control_loss: max:{}, mean:{}".format(max(positional_sentiment_losses),
                                                                 sum(positional_sentiment_losses) / len(positional_sentiment_losses)))

    if arg_config["data"]["local_rank"] in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, model