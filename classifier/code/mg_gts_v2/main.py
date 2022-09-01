#coding utf-8

import json, os, math
import random
import argparse

import numpy
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np

from classifier.code.mg_gts_v2.data import load_data_instances, DataIterator
from classifier.code.mg_gts_v2.model import MG_GTS
import classifier.code.mg_gts_v2.utils as utils
from utils.data_util import is_chinese


def reset_params(c_args, model):
    for child in model.children():
        for p in child.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    c_args["initializer"](p)
                else:    
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

def train(args):
    # load double embedding
    word2index = json.load(open(args.prefix + 'doubleembedding/word2index.json'))
    # general_embedding = numpy.load(args.prefix + 'doubleembedding/gen.vec.npy')
    general_embedding = numpy.load(args.prefix + 'doubleembedding/glove_filtered_300d.npy')
    general_embedding = torch.from_numpy(general_embedding)
    # domain_embedding = numpy.load(args.prefix +'doubleembedding/'+args.dataset+'_emb.vec.npy') # TODO 这里，领域内嵌入解决下
    domain_embedding = numpy.load(args.prefix + 'doubleembedding/glove_filtered_300d.npy')
    domain_embedding = torch.from_numpy(domain_embedding)

    # load dataset
    train_sentence_packs = json.load(open(args.prefix + args.dataset + '/train.json'))
    random.shuffle(train_sentence_packs)
    dev_sentence_packs = json.load(open(args.prefix + args.dataset + '/dev.json'))

    instances_train = load_data_instances(train_sentence_packs, word2index, args)
    instances_dev = load_data_instances(dev_sentence_packs, word2index, args)

    random.shuffle(instances_train)
    trainset = DataIterator(instances_train, args)
    devset = DataIterator(instances_dev, args)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # build model
    model = MG_GTS(general_embedding, domain_embedding, args).to(args.device)

    parameters = list(model.parameters())
    parameters = filter(lambda x: x.requires_grad, parameters)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    reset_params(args, model)

    # training
    best_joint_f1 = 0
    best_joint_epoch = 0
    for i in range(args.epochs):
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

        joint_precision, joint_recall, joint_f1 = classifier_eval(model, devset, args)

        if joint_f1 > best_joint_f1:
            model_path = args.model_dir + args.model + args.task + '.pt'
            torch.save(model, model_path)
            best_joint_f1 = joint_f1
            best_joint_epoch = i
    print('best epoch: {}\tbest dev {} f1: {:.5f}\n\n'.format(best_joint_epoch, args.task, best_joint_f1))


def classifier_eval(model, dataset, c_args):
    model.eval()
    with torch.no_grad():
        predictions=[]
        labels=[]
        all_ids = []
        all_lengths = []
        for i in range(dataset.batch_count):
            sentence_ids, sentence_tokens, lengths, mask, aspect_tags, _, tags = dataset.get_batch(i)
            prediction = model.forward(sentence_tokens, lengths, mask)  # B_S_S_C
            prediction = torch.argmax(prediction, dim=3)
            prediction_padded = torch.zeros(prediction.shape[0], c_args["max_sequence_len"], c_args["max_sequence_len"])
            prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
            predictions.append(prediction_padded)

            all_ids.extend(sentence_ids)
            labels.append(tags)
            all_lengths.append(lengths)

        predictions = torch.cat(predictions,dim=0).cpu().tolist()
        labels = torch.cat(labels,dim=0).cpu().tolist()
        all_lengths = torch.cat(all_lengths, dim=0).cpu().tolist()
        precision, recall, f1 = utils.score_uniontags(c_args, predictions, labels, all_lengths, ignore_index=-1)

        aspect_results = utils.score_aspect(predictions, labels, all_lengths, ignore_index=-1)
        opinion_results = utils.score_opinion(predictions, labels, all_lengths, ignore_index=-1)
        print('Aspect term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(aspect_results[0], aspect_results[1], aspect_results[2]))
        print('Opinion term\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}'.format(opinion_results[0], opinion_results[1], opinion_results[2]))
        print(c_args["task"]+'\tP:{:.5f}\tR:{:.5f}\tF1:{:.5f}\n'.format(precision, recall, f1))

    model.train()
    return precision, recall, f1

def train_classifier_with_reconstruction_reward(classifier, generator, block, unlabeled_dataset):
    pass  # TODO 2022 02 21 使用重构反馈机制来训练分类模型


def classifier_generate_pseudo(model, dataset, c_args):
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_tuples=[]
        print('generating pseudo labels...')
        step = 20
        for cut in trange(0, dataset.batch_count, step):
            predictions = []
            prediction_scores = []
            cut_lengths = []
            prediction_padded, prediction_score_padded = None, None
            for i in range(cut, min(dataset.batch_count, cut+step)):
                sentence_ids, sentence_tokens, lengths, mask, sentences = dataset.get_batch(i)
                prediction = model.forward(sentence_tokens, lengths, mask)  # B_S_S_C
                # 归一化
                prediction = torch.softmax(prediction, dim=3)
                # prediction = torch.sigmoid(prediction)
                # sum_up = torch.sum(prediction, dim=3, keepdim=True)
                # fill_ones = torch.ones_like(sum_up)
                # sum_up = torch.where(sum_up > 0, sum_up, fill_ones)
                # prediction = prediction / sum_up
                prediction_score = torch.max(prediction, dim=3)[0]  # B_S_S
                prediction = torch.argmax(prediction, dim=3)
                prediction_padded = torch.zeros(prediction.shape[0], c_args["max_sequence_len"], c_args["max_sequence_len"])
                prediction_score_padded = torch.clone(prediction_padded)
                prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
                prediction_score_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction_score
                predictions.append(prediction_padded)
                prediction_scores.append(prediction_score_padded)

                all_ids.extend(sentence_ids)
                cut_lengths.append(lengths)
                all_sentences.extend(sentences)

            if len(predictions) == 0:
                continue
            predictions = torch.cat(predictions,dim=0).cpu().tolist()
            prediction_scores = torch.cat(prediction_scores,dim=0).cpu().tolist()
            cut_lengths = torch.cat(cut_lengths, dim=0).cpu().tolist()
            cut_tuples = utils.score_pseudo_labels(c_args, prediction_scores, predictions,
                                                   cut_lengths, ignore_index=-1)
            all_tuples.extend(cut_tuples)
        assert len(all_ids) == len(all_sentences) == len(all_tuples)
        # 筛选掉空triple的数据
        filtered_results = list(filter(lambda x: len(x[2]) > 0, zip(all_ids, all_sentences, all_tuples)))
        filtered_results = list(filter(lambda x: not is_chinese(x[1]), filtered_results))
        filtered_results = sorted(filtered_results, key=lambda x: x[0])
        dict_results = []
        for (psd_id, psd_sentence, psd_tuple) in filtered_results:
            dict_results.append({"id": psd_id, "sentence":psd_sentence,"triples":psd_tuple})

    model.train()
    return dict_results


def classifier_generate_pseudo_opinion_span(model, dataset, c_args): # TODO 0620 11 ;33
    model.eval()
    with torch.no_grad():
        all_ids = []
        all_sentences = []
        all_tuples=[]
        print('generating pseudo opinion labels...')
        step = 20
        match_numss, total_triple_numss = 0, 0
        for cut in trange(0, dataset.batch_count, step):
            predictions = []
            prediction_scores = []
            cut_lengths = []
            sub_aspect_spans = []
            sub_polarities = []
            for i in range(cut, min(dataset.batch_count, cut+step)):
                sentence_ids, sentence_tokens, lengths, mask, sentences, predicted_aspect_spans, polarities = dataset.get_batch(i)
                prediction = model.forward(sentence_tokens, lengths, mask)  # B_S_S_C
                # 归一化
                prediction = torch.softmax(prediction, dim=3)
                # prediction = torch.sigmoid(prediction)
                # sum_up = torch.sum(prediction, dim=3, keepdim=True)
                # fill_ones = torch.ones_like(sum_up)
                # sum_up = torch.where(sum_up > 0, sum_up, fill_ones)
                # prediction = prediction / sum_up
                prediction_score = torch.max(prediction, dim=3)[0]  # B_S_S
                prediction = torch.argmax(prediction, dim=3)
                prediction_padded = torch.zeros(prediction.shape[0], c_args["max_sequence_len"], c_args["max_sequence_len"])
                prediction_score_padded = torch.clone(prediction_padded)
                prediction_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction
                prediction_score_padded[:, :prediction.shape[1], :prediction.shape[1]] = prediction_score
                predictions.append(prediction_padded)
                prediction_scores.append(prediction_score_padded)

                all_ids.extend(sentence_ids)
                cut_lengths.append(lengths)
                all_sentences.extend(sentences)
                sub_aspect_spans.extend(predicted_aspect_spans)
                sub_polarities.extend(polarities)

            if len(predictions) == 0:
                continue
            predictions = torch.cat(predictions,dim=0).cpu().tolist()
            prediction_scores = torch.cat(prediction_scores,dim=0).cpu().tolist()
            cut_lengths = torch.cat(cut_lengths, dim=0).cpu().tolist()
            print(len(predictions), len(sub_aspect_spans), len(sub_polarities))
            assert len(predictions) == len(sub_aspect_spans) == len(sub_polarities)
            cut_tuples, match_nums, total_triple_nums = utils.score_pseudo_opinions(c_args, prediction_scores, predictions,
                                                     cut_lengths, sub_aspect_spans,
                                                     sub_polarities, ignore_index=-1)
            match_numss += match_nums
            total_triple_numss += total_triple_nums

            all_tuples.extend(cut_tuples)
        assert len(all_ids) == len(all_sentences) == len(all_tuples)
        print(match_numss, total_triple_numss)
        # 筛选掉空triple的数据
        filtered_results = list(filter(lambda x: len(x[2]) > 0, zip(all_ids, all_sentences, all_tuples)))
        filtered_results = list(filter(lambda x: not is_chinese(x[1]), filtered_results))
        filtered_results = sorted(filtered_results, key=lambda x: x[0])
        dict_results = []
        for (psd_id, psd_sentence, psd_tuple) in filtered_results:
            dict_results.append({"id": psd_id, "sentence":psd_sentence,"triples":psd_tuple})

    model.train()
    return dict_results


def classifier_test(c_args):
    print("Evaluation on testset:")
    model_path = c_args["model_dir"] + c_args["model"] + c_args["task"] + '.pt'
    model = torch.load(model_path).to(c_args["device"])
    model.eval()

    word2index = json.load(open(c_args["prefix"] + 'doubleembedding/word2index.json'))
    sentence_packs = json.load(open(c_args["prefix"] + c_args["dataset"] + '/test.json'))
    instances = load_data_instances(sentence_packs, word2index, c_args)
    testset = DataIterator(instances, c_args)
    classifier_eval(model, testset, c_args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/",
                        help='dataset and embedding path prefix')
    parser.add_argument('--model_dir', type=str, default="savemodel/",
                        help='model path prefix')
    parser.add_argument('--task', type=str, default="triplet", choices=["pair", "triplet"],
                        help='option: pair, triplet')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"],
                        help='option: train, test')
    parser.add_argument('--model', type=str, default="cnn", choices=["cnn"],
                        help='option: cnn')
    parser.add_argument('--dataset', type=str, default="SemEval",
                        help='dataset')
    parser.add_argument('--max_sequence_len', type=int, default=100,
                        help='max length of a sentence')
    parser.add_argument('--device', type=str, default="cuda",
                        help='gpu or cpu')

    parser.add_argument('--lstm_dim', type=int, default=50,
                        help='dimension of lstm cell')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='dimension of lstm cell')
    parser.add_argument('--cnn_dim', type=int, default=256,
                        help='dimension of cnn')

    parser.add_argument('--weight_decay', type=float, default=2e-5,
                        help='weight decay')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='bathc size')
    parser.add_argument('--epochs', type=int, default=400,
                        help='training epoch number')
    parser.add_argument('--class_num', type=int, default=6,
                        help='label number')
    
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    # parser.add_argument('--patience', default=200, type=int)

    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    args.initializer = initializers[args.initializer]

    if args.mode == 'train':
        train(args)
        classifier_test(args)
    else:
        classifier_test(args)

