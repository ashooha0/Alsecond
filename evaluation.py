import glob
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import argparse
import json
import logging
from torch.distributed import get_rank
from main_framework import MODEL_CLASSES, evaluate_scores, evaluate_ppl
from model.transformers import SenTriConBlock, WEIGHTS_NAME
from utils.data_util import generate_alsecond_compute, evaluate_BLEU, evaluate_NIST_AND_METEOR_AND_ROUGE, \
    constract_batch_mask, load_and_cache_examples


def generate_samples(model, sentricon_block, arg_config):
    pass


def set_seed(arg_config):
    random.seed(arg_config['data']['random_seed'])
    np.random.seed(arg_config['data']['random_seed'])
    torch.manual_seed(arg_config['data']['random_seed'])
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(arg_config['data']['random_seed'])

def regenerate_samples(gen_args):
    if gen_args["data"]["local_rank"] == -1 or gen_args["data"]["no_cuda"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not gen_args["data"]["no_cuda"] else "cpu")
        gen_args['data']['n_gpu'] = torch.cuda.device_count()
        # args.n_gpu =
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(gen_args["data"]["local_rank"])
        device = torch.device("cuda", gen_args["data"]["local_rank"])
        torch.distributed.init_process_group(backend="nccl")
        gen_args['data']['n_gpu'] = 1
    device = torch.device("cuda" if torch.cuda.is_available() and not gen_args["data"]["no_cuda"] else "cpu")
    gen_args['data']['device'] = device
    set_seed(gen_args)
    if gen_args['data']['gen_cs_len'] is None:
        gen_args['data']['gen_cs_len'] = gen_args['data']['cs_len']
    if gen_args['data']['gen_hs_len'] is None:
        gen_args['data']['gen_hs_len'] = gen_args['data']['hs_len']
    if gen_args['data']['gen_tis_len'] is None:
        gen_args['data']['gen_tis_len'] = gen_args['data']['tis_len']
    config_class, model_class, tokenizer_class, query_hinted_model_class = MODEL_CLASSES[gen_args['model']['model_type']]
    config = config_class.from_pretrained(gen_args['model']['model_name_or_path'],
                                          cache_dir=gen_args['model']['cache_dir'])
    logging.info("Loading tokenizer from pretrained, {}".format(gen_args['model']['model_name_or_path']))
    tokenizer = tokenizer_class.from_pretrained(gen_args["model"]["result_path"])
    if not gen_args['evaluate']['eval_compute_without_checkpoint']:
        # checkpoints = ["pretrained"]
        # g_args["G1"]["saving_dir"],gen_args["model"]["best_save_dir"]
        checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(g_args["G1"]["saving_dir"] + "/**/" + WEIGHTS_NAME, recursive=True))
        )
    else:
        checkpoints = [gen_args["model"]["result_path"]]
    for checkpoint in checkpoints:
        # print("#"*10, checkpoint)
        if not gen_args['evaluate']['eval_compute_without_checkpoint']:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            if gen_args['model']['output_meanvars']:
                generator_model = query_hinted_model_class.from_pretrained(
                    checkpoint,
                    output_meanvars=True,
                    compute_meanvars_before_layernorm=gen_args['model']['compute_meanvars_before_layernorm']
                )
            else:
                generator_model = query_hinted_model_class.from_pretrained(checkpoint)
        else:
            generator_model = query_hinted_model_class.from_pretrained(gen_args["model"]["result_path"])
            global_step = 0
            prefix = ""

        generator_model.to(gen_args['data']['device'])

        generate_steps = generate_alsecond_compute(gen_args, checkpoint, generator_model,
                                                tokenizer,
                                                prefix=prefix,
                                                transform_h_after_layernorm=gen_args["training"][
                                                    "transform_h_after_layernorm"])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(  # 配置参数
        "--generator_config",
        help="path to generator json config",
        default="data/generator_config.json"
    )
    args = parser.parse_args()
    g_args = json.load(open(args.generator_config, 'r'))  # 加载参数

    if g_args["evaluate"]["regenerate"]:
        regenerate_samples(g_args)
    evaluate_ppl(g_args)
    evaluate_scores(g_args)