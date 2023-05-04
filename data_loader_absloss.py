import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import prettytable as pt
from gensim.models import KeyedVectors
from transformers import AutoTokenizer
import os
import utils
import requests
os.environ["TOKENIZERS_PARALLELISM"] = "false"

dis2idx = np.zeros((1000), dtype='int64')

dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9


class Vocabulary(object):
    PAD = '<pad>'
    UNK = '<unk>'
    SUC = '<suc>'

    def __init__(self):
        self.label2id = {self.PAD: 0, self.SUC: 1}
        self.id2label = {0: self.PAD, 1: self.SUC}

    def add_label(self, label):
        label = label.lower()
        if label not in self.label2id:
            self.label2id[label] = len(self.label2id)
            self.id2label[self.label2id[label]] = label

        assert label == self.id2label[self.label2id[label]]

    def __len__(self):
        return len(self.token2id)

    def label_to_id(self, label):
        label = label.lower()
        return self.label2id[label]

    def id_to_label(self, i):
        return self.id2label[i]


def collate_fn(data):  # bert tokenize化的数据
    bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_num, entity_labels, entity_label_mask2d, entity_text = map(
        list, zip(*data))

    max_tok = np.max(sent_length)
    max_entity_num = np.max(entity_num)
    sent_length = torch.LongTensor(sent_length)

    max_pie = np.max([x.shape[0] for x in bert_inputs])
    bert_inputs = pad_sequence(bert_inputs, True)
    batch_size = bert_inputs.size(0)

    def fill(data, new_data):
        for j, x in enumerate(data):
            new_data[j, :x.shape[0], :x.shape[1]] = x
        return new_data

    def fill_entity_label(data, new_data):
        for j, x in enumerate(data):
            if len(x.shape) != 1:
                new_data[j, :x.shape[0], :x.shape[1], :x.shape[2]] = x
            """
            else:
                print(x.shape)
            """

        return new_data

    dis_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    dist_inputs = fill(dist_inputs, dis_mat)

    labels_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.long)
    grid_labels = fill(grid_labels, labels_mat)

    mask2d_mat = torch.zeros((batch_size, max_tok, max_tok), dtype=torch.bool)
    grid_mask2d = fill(grid_mask2d, mask2d_mat)

    entity_labels_mat = torch.zeros(
        batch_size, max_entity_num, max_tok, max_tok)
    entity_labels = fill_entity_label(entity_labels, entity_labels_mat)
    """
    count = 0
    print(max_entity_num, max_tok)
    for i in entity_labels:
        count += 1
        print(len(i.shape), count)
    
    """

    # [batch,n]用于最终输出去除填充entity
    """
        entity_grid_mat = torch.zeros(batch_size, max_entity_num, max_tok, max_tok)
    entity_grid_mask2d = fill_entity_label(
        entity_grid_mask2d, entity_grid_mat)  # 用于恢复填充数据至原序列
    """

    entity_label_mat = torch.zeros(
        batch_size, max_entity_num, max_tok, max_tok)
    entity_label_mask2d = fill_entity_label(
        entity_label_mask2d, entity_label_mat)

    sub_mat = torch.zeros((batch_size, max_tok, max_pie), dtype=torch.bool)
    pieces2word = fill(pieces2word, sub_mat)

    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length,  entity_labels,  entity_label_mask2d, entity_text


class RelationDataset(Dataset):
    def __init__(self, bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_num, entity_labels,  entity_label_mask2d, entity_text):
        self.bert_inputs = bert_inputs
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.entity_num = entity_num
        self.entity_labels = entity_labels
        #self.entity_grid_mask2d = entity_grid_mask2d
        self.entity_label_mask2d = entity_label_mask2d
        self.entity_text = entity_text

    def __getitem__(self, item):
        return torch.LongTensor(self.bert_inputs[item]), \
            torch.LongTensor(self.grid_labels[item]), \
            torch.LongTensor(self.grid_mask2d[item]), \
            torch.LongTensor(self.pieces2word[item]), \
            torch.LongTensor(self.dist_inputs[item]), \
            self.sent_length[item], \
            self.entity_num[item],\
            torch.LongTensor(self.entity_labels[item]),\
            torch.LongTensor(self.entity_label_mask2d[item]),\
            self.entity_text[item],

    def __len__(self):
        return len(self.bert_inputs)


def process_bert(data, tokenizer, vocab):

    bert_inputs = []
    grid_labels = []
    grid_mask2d = []
    dist_inputs = []
    entity_text = []
    pieces2word = []
    sent_length = []
    entity_num = []  # 暂用不到
    entity_labels = []
    #entity_grid_mask2d = []
    entity_label_mask2d = []

    for index, instance in enumerate(data):
        if len(instance['sentence']) == 0:
            continue

        # 细看tokenize模块[['Mu', '##s', '##cle'], ['Pain'], ['.']]
        tokens = [tokenizer.tokenize(word) for word in instance['sentence']]
        # ['Mu', '##s', '##cle', 'Pain', '.']
        pieces = [piece for pieces in tokens for piece in pieces]
        _bert_inputs = tokenizer.convert_tokens_to_ids(
            pieces)  # [19569, 1116, 10536, 13304, 119]
        # [  101 19569  1116 10536 13304   119   102]
        _bert_inputs = np.array(
            [tokenizer.cls_token_id] + _bert_inputs + [tokenizer.sep_token_id])

        length = len(instance['sentence'])
        entity_length = len(instance["ner"])
        _grid_labels = np.zeros((length, length), dtype=np.int)
        _pieces2word = np.zeros((length, len(_bert_inputs)), dtype=np.bool)
        _dist_inputs = np.zeros((length, length), dtype=np.int)
        _grid_mask2d = np.ones((length, length), dtype=np.bool)
        # 用于将原outputs[b,l,l,c]-->转换成[b,e,l,l,c]
        _entity_label_mask2d = []
        _entity_labels = []
        #_entity_grid_mask2d = []

        if tokenizer is not None:
            start = 0
            for i, pieces in enumerate(tokens):
                if len(pieces) == 0:
                    continue
                pieces = list(range(start, start + len(pieces)))
                # 一个word可能对应多个piece
                _pieces2word[i, pieces[0] + 1:pieces[-1] + 2] = 1
                start += len(pieces)

        for k in range(length):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(length):
            for j in range(length):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19

        for ind, entity in enumerate(instance["ner"]):

            index = entity["index"]
            __entity_label_mask2d = np.zeros((length, length), dtype=np.bool)
            __entity_labels = np.zeros((length, length), dtype=np.int)
            for i in range(len(index)):
                if i + 1 >= len(index):
                    break

                _grid_labels[index[i], index[i + 1]] = 1  # 相邻标签
                __entity_label_mask2d[index[i], index[i+1]] = True
                __entity_labels[index[i], index[i+1]] = 1

            _grid_labels[index[-1], index[0]
                         ] = vocab.label_to_id(entity["type"])  # 头尾标签
            __entity_label_mask2d[index[-1], index[0]] = True
            __entity_labels[index[-1], index[0]
                            ] = vocab.label_to_id(entity["type"])

            _entity_label_mask2d.append(__entity_label_mask2d)
            _entity_labels.append(__entity_labels)
            # _entity_grid_mask2d.append(_grid_mask2d)

        _entity_text = set([utils.convert_index_to_text(e["index"], vocab.label_to_id(e["type"]))
                            for e in instance["ner"]])  # 文本+类型

        # [  101 19569  1116 10536 13304   119   102]
        bert_inputs.append(_bert_inputs)  # 每个句子输入bert的表示
        sent_length.append(length)  # 每个句子的长度
        entity_num.append(entity_length)
        grid_labels.append(_grid_labels)  # 相邻关系类型结尾关系标签
        grid_mask2d.append(_grid_mask2d)  # 没看到其他处理步骤，全为1
        # entity_grid_mask2d.append(_entity_grid_mask2d)
        dist_inputs.append(_dist_inputs)  # 词对间的距离
        pieces2word.append(_pieces2word)  # 切片到词的映射
        entity_text.append(_entity_text)  # 1-2-3-#-add
        # 实体标签bool值，[sentence,entity_num,length,length]
        entity_label_mask2d.append(_entity_label_mask2d)
        entity_labels.append(_entity_labels)  # 实体标签
    return bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length, entity_num, entity_labels,  entity_label_mask2d, entity_text


def fill_vocab(vocab, dataset):  # 添加标签类型返回总实体数量
    entity_num = 0
    for instance in dataset:
        for entity in instance["ner"]:
            vocab.add_label(entity["type"])
        entity_num += len(instance["ner"])
    return entity_num


def load_data_bert(config):
    with open('./data/{}/train.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open('./data/{}/dev.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open('./data/{}/test.json'.format(config.dataset), 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        config.bert_name, cache_dir="./cache/")

    vocab = Vocabulary()
    train_ent_num = fill_vocab(vocab, train_data)
    dev_ent_num = fill_vocab(vocab, dev_data)
    test_ent_num = fill_vocab(vocab, test_data)

    table = pt.PrettyTable(
        [config.dataset, 'sentences', 'entities'])  # 可视化表显示数据集长度，实体数
    table.add_row(['train', len(train_data), train_ent_num])
    table.add_row(['dev', len(dev_data), dev_ent_num])
    table.add_row(['test', len(test_data), test_ent_num])
    config.logger.info("\n{}".format(table))

    config.label_num = len(vocab.label2id)  # 艹终于找到你
    config.vocab = vocab

    train_dataset = RelationDataset(
        *process_bert(train_data, tokenizer, vocab))  # 训练数据，分词器，标签表(不包含word关系)
    dev_dataset = RelationDataset(*process_bert(dev_data, tokenizer, vocab))
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, vocab))
    return (train_dataset, dev_dataset, test_dataset), (train_data, dev_data, test_data)
