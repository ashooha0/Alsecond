import math

import torch

sentiment2id = {'NEG': 3, 'NEU': 4, 'POS': 5}
class UnlabeledInstance(object):
    def __init__(self, sentence_pack, word2index, c_args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(c_args["max_sequence_len"]).long()

        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        '''generate mask of the sentence'''
        self.mask = torch.zeros(c_args["max_sequence_len"])
        self.mask[:self.length] = 1


class Instance(object):
    def __init__(self, sentence_pack, word2index, c_args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(c_args["max_sequence_len"]).long()

        '''generate sentence tokens'''
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        self.aspect_tags = torch.zeros(c_args["max_sequence_len"]).long()
        self.opinion_tags = torch.zeros(c_args["max_sequence_len"]).long()
        self.aspect_tags[self.length:] = -1
        self.opinion_tags[self.length:] = -1
        self.tags = torch.zeros(c_args["max_sequence_len"], c_args["max_sequence_len"]).long()
        self.tags[:, :] = -1

        for i in range(self.length):
            for j in range(i, self.length):
                self.tags[i][j] = 0
        for triple in sentence_pack['triples']:
            aspect_span = triple[0]
            opinion_span = triple[1]

            l, r = aspect_span[0], aspect_span[1]
            for i in range(l, r+1):
                self.aspect_tags[i] = 1 if i == l else 2
                self.tags[i][i] = 1
                if i > l: self.tags[i-1][i] = 1
                for j in range(i, r+1):
                    self.tags[i][j] = 1
            l, r = opinion_span[0], opinion_span[1]
            for i in range(l, r+1):
                self.opinion_tags[i] = 1 if i == l else 2
                self.tags[i][i] = 2
                if i > l: self.tags[i-1][i] = 2
                for j in range(i, r+1):
                    self.tags[i][j] = 2
            al, ar = aspect_span[0], aspect_span[1]
            pl, pr = opinion_span[0], opinion_span[1]
            for i in range(al, ar+1):
                for j in range(pl, pr+1):
                    if c_args["task"] == 'pair':
                        if i > j: self.tags[j][i] = 3
                        else: self.tags[i][j] = 3
                    elif c_args["task"] == 'triplet':
                        if i > j: self.tags[j][i] = sentiment2id[triple[2]]
                        else: self.tags[i][j] = sentiment2id[triple[2]]
        '''generate mask of the sentence'''
        self.mask = torch.zeros(c_args["max_sequence_len"])
        self.mask[:self.length] = 1


class AspectLabeledInstance(object):
    def __init__(self, sentence_pack, word2index, c_args):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.sentence_tokens = torch.zeros(c_args["max_sequence_len"]).long()

        '''generate sentence tokens'''
        words = self.sentence.split()
        self.length = len(words)
        for i, w in enumerate(words):
            # word = w.lower()
            word = w
            if word in word2index:
                self.sentence_tokens[i] = word2index[word]
            else:
                self.sentence_tokens[i] = word2index['<unk>']

        aspect_spans = []
        polarities = []
        for pair in sentence_pack['pairs']:
            aspect_spans.append(pair[0])
            polarities.append(sentiment2id[pair[1]])
        self.aspect_spans = aspect_spans
        self.polarities = polarities
        '''generate mask of the sentence'''
        self.mask = torch.zeros(c_args["max_sequence_len"])
        self.mask[:self.length] = 1


def load_data_instances(sentence_packs, word2index, c_args, labeled='True'):
    instances = list()
    if labeled == 'True':
        for sentence_pack in sentence_packs:
            instances.append(Instance(sentence_pack, word2index, c_args))
    elif labeled == 'False':
        for sentence_pack in sentence_packs:
            instances.append(UnlabeledInstance(sentence_pack, word2index, c_args))
    else:
        for sentence_pack in sentence_packs:
            instances.append(AspectLabeledInstance(sentence_pack, word2index, c_args))
    return instances


class DataIterator(object):
    def __init__(self, instances, c_args):
        assert len(instances) > 0
        if isinstance(instances[0], Instance):
            self.labeled = 'True'
        elif isinstance(instances[0], UnlabeledInstance):
            self.labeled = 'False'
        else:
            self.labeled = 'Half'
        # self.labeled = True if isinstance(instances[0], Instance) else False
        self.instances = instances
        self.c_args = c_args
        self.batch_count = math.ceil(len(instances)/c_args["batch_size"])

    def get_batch(self, index):
        sentence_ids = []
        sentence_tokens = []
        lengths = []
        masks = []
        sentences = []
        aspect_tags = []
        opinion_tags = []
        aspect_spans = []
        polarities = []
        tags = []

        for i in range(index * self.c_args["batch_size"],
                       min((index + 1) * self.c_args["batch_size"], len(self.instances))):
            sentence_ids.append(self.instances[i].id)
            sentence_tokens.append(self.instances[i].sentence_tokens)
            lengths.append(self.instances[i].length)
            masks.append(self.instances[i].mask)
            sentences.append(self.instances[i].sentence)
            if self.labeled == "True":
                aspect_tags.append(self.instances[i].aspect_tags)
                opinion_tags.append(self.instances[i].opinion_tags)
                tags.append(self.instances[i].tags)
            elif self.labeled == "Half":
                aspect_spans.append(self.instances[i].aspect_spans)
                polarities.append(self.instances[i].polarities)
        indexes = list(range(len(sentence_tokens)))
        indexes = sorted(indexes, key=lambda x: lengths[x], reverse=True)

        sentence_ids = [sentence_ids[i] for i in indexes]
        sentence_tokens = torch.stack(sentence_tokens).to(self.c_args["device"])[indexes]
        sentences = [sentences[i] for i in indexes]
        lengths = torch.tensor(lengths).to(self.c_args["device"])[indexes]
        masks = torch.stack(masks).to(self.c_args["device"])[indexes]
        if self.labeled == "True":
            aspect_tags = torch.stack(aspect_tags).to(self.c_args["device"])[indexes]
            opinion_tags = torch.stack(opinion_tags).to(self.c_args["device"])[indexes]
            tags = torch.stack(tags).to(self.c_args["device"])[indexes]
            return sentence_ids, sentence_tokens, lengths, masks, aspect_tags, opinion_tags, tags
        elif self.labeled == "Half":
            aspect_spans = [aspect_spans[i] for i in indexes]
            polarities = [polarities[i] for i in indexes]
            return sentence_ids, sentence_tokens, lengths, masks, sentences, aspect_spans, polarities
        else:
            return sentence_ids, sentence_tokens, lengths, masks, sentences
