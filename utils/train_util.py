from utils.data_util import load_and_cache_examples

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import shutil
# from transformers import (
#     PreTrainedModel,
#     PreTrainedTokenizer,
# )
from model.transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import logging
import re
logger = logging.getLogger(__name__)
# FIXME
# FIXME
def evaluate(arg_config, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = arg_config["model"]["output_dir"]
    eval_dataset = load_and_cache_examples(arg_config,
                                           tokenizer, evaluate=True,
                                           text_json_key="sentence",
                                           prepended_text_to_remove=arg_config["data"]["prepended_text_to_remove"])

    if arg_config["data"]["local_rank"] in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    arg_config["data"]["eval_batch_size"] = \
        arg_config["data"]["per_gpu_eval_batch_size"] * max(1, arg_config['data']['n_gpu'])
    # Note that DistributedSampler samples randomly

    def collate(examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
        def pad_chaos(data, batch_first=True, padding_value=0.0):  # pad B * A * T(batch, aspect/opinion, token)
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
        # print("BEFORE", sentences)
        if tokenizer._pad_token is None:
            sentences = pad_sequence(sentences, batch_first=True)
            # aspects = pad_chaos(aspects, batch_first=True)
            # opinions = pad_chaos(opinions, batch_first=True)
            polarities = pad_sequence(polarities, batch_first=True)
        else:
            sentences = pad_sequence(sentences, batch_first=True, padding_value=tokenizer.pad_token_id)
            # aspects = pad_chaos(aspects, batch_first=True, padding_value=tokenizer.pad_token_id)
            # opinions = pad_chaos(opinions, batch_first=True, padding_value=tokenizer.pad_token_id)
            polarities = pad_sequence(polarities, batch_first=True, padding_value=tokenizer.pad_token_id)
        # print("AFTER", sentences)
        return sentences, polarities   # , aspects, opinions,

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler,
        batch_size=arg_config["data"]["eval_batch_size"], collate_fn=collate
    )

    # multi-gpu evaluate
    if arg_config['data']['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", arg_config["data"]["eval_batch_size"])
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch[0], batch[0])
        # print(labels.shape)
        inputs = inputs.to(arg_config['data']['device'])
        labels = labels.to(arg_config['data']['device'])

        if labels.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, arg_config['data']['eval_output_filename'])
    with open(output_eval_file, "w") as writer:
        logger.info("***** PPL Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def _latest_checkpoint(checkpoint_dir, checkpoint_prefix="checkpoint", use_mtime=False):
    """
    find the latest checkpoint path
    :param checkpoint_dir:
    :param checkpoint_prefix:
    :return: Str(checkpoint) if exist, else: None
    """
    checkpoints_sorted = _sorted_checkpoints(checkpoint_dir, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) == 0:
        return None
    return checkpoints_sorted[-1]

# FIXME
# FIXME
def _sorted_checkpoints(output_dir, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

# FIXME
# FIXME
def _clear_checkpoints(output_dir, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(output_dir, checkpoint_prefix, use_mtime)

    for checkpoint in checkpoints_sorted:
        logger.info("Deleting older checkpoint [{}] before rerunning training".format(checkpoint))
        shutil.rmtree(checkpoint)

# FIXME
# FIXME
def _rotate_checkpoints(output_dir, save_total_limit, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not save_total_limit:
        return
    if save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(output_dir, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)
