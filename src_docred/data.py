import json
import math
import random
from collections import defaultdict
import dgl
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from tqdm.std import tqdm
from utils import get_cuda, Bert
from graph_utils import build_g

IGNORE_INDEX = -100


class BERTDGLREDataset(IterableDataset):

    def __init__(self, src_file, ner2id, rel2id,
                 dataset='train', instance_in_train=None,
                 model_name='FCDS_BERT_base'):

        super(BERTDGLREDataset, self).__init__()

        if instance_in_train is None:
            self.instance_in_train = set()
        else:
            self.instance_in_train = instance_in_train
        self.data = None
        self.document_max_length = 512
        self.bert = Bert(model_name)
        self.dataset = dataset
        self.rel2id = rel2id
        self.ner2id = ner2id
        print('Reading data from {}.'.format(src_file))

        self.create_data(src_file)
        self.get_instance_in_train()

    def get_instance_in_train(self):
        for doc in self.data:
            entity_list = doc['vertexSet']
            labels = doc.get('labels', [])
            for label in labels:
                head, tail, relation = label['h'], label['t'], label['r']
                label['r'] = self.rel2id[relation]
                if self.dataset == 'train':
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            self.instance_in_train.add(mention_triple)

    def process_doc(self, doc, dataset, ner2id, bert):
        title, entity_list = doc['title'], doc['vertexSet']
        labels, sentences = doc.get('labels', []), doc['sents']

        Ls = [0]
        L = 0
        for x in sentences:
            L += len(x)
            Ls.append(L)
        for j in range(len(entity_list)):
            for k in range(len(entity_list[j])):
                sent_id = int(entity_list[j][k]['sent_id'])
                entity_list[j][k]['sent_id'] = sent_id

                dl = Ls[sent_id]
                pos0, pos1 = entity_list[j][k]['pos']
                entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

        # generate positive examples
        train_triple = []
        new_labels = []
        for label in labels:
            head, tail, relation = label['h'], label['t'], label['r']
            # label['r'] = rel2id[relation]
            train_triple.append((head, tail))
            label['in_train'] = False

            # record training set mention triples and mark for dev and test
            for n1 in entity_list[head]:
                for n2 in entity_list[tail]:
                    mention_triple = (n1['name'], n2['name'], relation)
                    if dataset != 'train':
                        if mention_triple in self.instance_in_train:
                            label['in_train'] = True
                            break

            new_labels.append(label)

        # generate negative examples
        na_triple = []
        for j in range(len(entity_list)):
            for k in range(len(entity_list)):
                if j != k and (j, k) not in train_triple:
                    na_triple.append((j, k))

        # generate document ids
        words = []
        for sentence in sentences:
            for word in sentence:
                words.append(word)

        bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(
            words)

        word_id = np.zeros((self.document_max_length,), dtype=np.int32)
        pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
        ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
        mention_id = np.zeros((self.document_max_length,), dtype=np.int32)
        word_id[:] = bert_token[0]

        entity2mention = defaultdict(list)
        mention_idx = 1
        already_exist = set()
        pos_idx = {}
        ent_idx = {}
        for idx, vertex in enumerate(entity_list, 1):
            for v in vertex:

                sent_id, ner_type = v['sent_id'], v['type']
                pos0_w, pos1_w = v['global_pos']

                pos0 = bert_starts[pos0_w]
                if pos1_w < len(bert_starts):
                    pos1 = bert_starts[pos1_w]
                else:
                    pos1 = self.document_max_length

                if (pos0, pos1) in already_exist:
                    continue

                if pos0 >= len(pos_id):
                    continue

                if idx not in pos_idx:
                    pos_idx[idx] = []
                    ent_idx[idx] = []

                pos_idx[idx].extend(range(pos0_w, pos1_w))
                ent_idx[idx].extend(range(pos0, pos1))
                pos_id[pos0:pos1] = idx
                ner_id[pos0:pos1] = ner2id[ner_type]
                mention_id[pos0:pos1] = mention_idx
                entity2mention[idx].append(mention_idx)
                mention_idx += 1
                already_exist.add((pos0, pos1))

        # ======================================================
        # compute subword to word index
        sub2word = np.zeros((
            len(bert_starts)+len(entity_list),
            self.document_max_length))
        for idx in range(len(bert_starts)-1):
            start, end = bert_starts[idx], bert_starts[idx+1]
            if start == end:
                continue
            sub2word[idx, start:end] = 1/(end-start)
        start, end = bert_starts[-1], len(bert_subwords)
        sub2word[len(bert_starts)-1, start:end] = 1/(end-start)
        # compute convertion matrix for entity
        for idx, poss in ent_idx.items():
            # print('------------>', idx, poss)
            sub2word[len(bert_starts)+idx-1, poss] = 1/len(poss)
        # ======================================================
        # compute words to sent index
        word2sent = np.zeros((len(Ls)-1, Ls[-1]))
        for i in range(1, len(Ls)):
            word2sent[i-1, Ls[i-1]:Ls[i]] = 1 / (Ls[i] - Ls[i-1])
        # ======================================================

        replace_i = 0
        idx = len(entity_list)
        if entity2mention[idx] == []:
            entity2mention[idx].append(mention_idx)
            while mention_id[replace_i] != 0:
                replace_i += 1
            mention_id[replace_i] = mention_idx
            pos_id[replace_i] = idx
            ner_id[replace_i] = ner2id[vertex[0]['type']]
            mention_idx += 1

        new_Ls = [0]
        for ii in range(1, len(Ls)):
            if Ls[ii] < len(bert_starts):
                new_Ls.append(bert_starts[Ls[ii]])
            else:
                new_Ls.append(len(bert_subwords))

        Ls = new_Ls

        graph2, path2 = build_g(sentences, pos_idx, pos_id.max())

        return {
            'title': title,
            'num_sent': len(doc['sents']),
            'entities': entity_list,
            'labels': new_labels,
            'na_triple': na_triple,
            'word_id': word_id,
            'pos_id': pos_id,
            'ner_id': ner_id,
            'sub2word': sub2word,
            'word2sent': word2sent,
            'graph2': graph2,
            'path2': path2
        }

    def create_data(self, src_file):
        with open(file=src_file, mode='r', encoding='utf-8') as fr:
            ori_data = json.load(fr)
        self.data = ori_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc = self.data[idx]
        cur_d = self.process_doc(
            doc, dataset=self.dataset, ner2id=self.ner2id, bert=self.bert)
        return cur_d

    def __iter__(self):
        return iter(self.data)


class DGLREDataloader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=False,
                 h_t_limit=1722, relation_num=97,
                 max_length=512, negativa_alpha=0.0, dataset_type='train'):
        super(DGLREDataloader, self).__init__(
            dataset, batch_size=batch_size, num_workers=4)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.negativa_alpha = negativa_alpha
        self.dataset_type = dataset_type
        self.h_t_limit = h_t_limit
        self.relation_num = relation_num
        self.order = list(range(self.length))
        self.data = []
        self.boosted = 0
        for idx in tqdm(range(self.length)):
            self.data.append(self.dataset[idx])
            self.data[idx]['idx'] = idx

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [(
            idx * self.batch_size,
            min(self.length, (idx + 1) * self.batch_size))
                for idx in range(0, batch_num)]
        self.batches_order = [self.order[
            idx * self.batch_size: min(self.length, (idx+1) * self.batch_size)]
                for idx in range(0, batch_num)]

        # begin
        context_word_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_pos_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_ner_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_word_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_word_length = torch.LongTensor(self.batch_size).cpu()
        ht_pairs = torch.LongTensor(self.batch_size, self.h_t_limit, 2).cpu()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cpu()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cpu()
        relation_example_idx = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()
        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()

        for idx, (batch_s, batch_e) in enumerate(self.batches):
            minibatch = [self.data[idx] for idx in self.order[batch_s: batch_e]]
            cur_bsz = len(minibatch)

            for mapping in [context_word_ids, context_pos_ids, context_ner_ids,
                            context_word_mask, context_word_length, ht_pairs,
                            relation_multi_label, relation_mask,
                            relation_label, relation_example_idx]:
                if mapping is not None:
                    mapping.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 0

            label_list = []
            L_vertex = []
            titles = []
            indexes = []
            graphs = []
            path2_table = []
            sub2word_list = []
            word2sent_list = []

            for i, example in enumerate(minibatch):
                entities, labels, na_triple, word_id, pos_id, ner_id = \
                    example['entities'], example['labels'], example['na_triple'], \
                    example['word_id'], example['pos_id'], example['ner_id']
                graphs.append(dgl.add_self_loop(example['graph2']).to('cuda:0'))
                path2_table.append(example['path2'])

                prewrong = example.get('wrong_predits', [])

                sub2word_list.append(torch.Tensor(example['sub2word']))
                word2sent_list.append(torch.Tensor(example['word2sent']))
                L = len(entities)
                word_num = word_id.shape[0]

                context_word_ids[i, :word_num].copy_(torch.from_numpy(word_id))
                context_pos_ids[i, :word_num].copy_(torch.from_numpy(pos_id))
                context_ner_ids[i, :word_num].copy_(torch.from_numpy(ner_id))

                idx2label = defaultdict(list)
                evid2label = defaultdict(list)
                label_set = {}
                for label in labels:
                    head, tail, relation, intrain = \
                        label['h'], label['t'], label['r'], label['in_train']
                    idx2label[(head, tail)].append(relation)
                    evid2label[(head, tail)].extend(label['evidence'])
                    label_set[(head, tail, relation)] = intrain

                label_list.append(label_set)

                if self.dataset_type == 'train':
                    train_tripe = list(idx2label.keys())
                    na_train_triple = set()
                    for j, (h_idx, t_idx) in enumerate(train_tripe):
                        ht_pairs[i, j, :] = torch.Tensor([h_idx+1, t_idx+1])
                        label = idx2label[(h_idx, t_idx)]
                        for r in label:
                            relation_multi_label[i, j, r] = 1

                        relation_mask[i, j] = 1
                        relation_label[i, j] = 1
                        relation_example_idx[i, j] = example['idx']

                    # =========================================
                    # This is for forcing selecting challenging negative pairs
                    #     if (t_idx, h_idx) not in idx2label:
                    #         na_train_triple.add((t_idx, h_idx))

                    to_sample = min(len(train_tripe), int(len(prewrong) * 0.1))
                    na_train_triple = random.sample(prewrong, to_sample)
                    self.boosted += to_sample
                    for j, (h_idx, t_idx) in enumerate(na_train_triple, len(train_tripe)):
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx+1])
                        relation_multi_label[i, j, 0] = 1
                        relation_label[i, j] = 0
                        relation_mask[i, j] = 1
                        relation_example_idx[i, j] = example['idx']
                    # =========================================

                    lower_bound = len(na_triple)
                    if self.negativa_alpha > 0.0:
                        random.shuffle(na_triple)
                        lower_bound = int(max(20, len(train_tripe) * self.negativa_alpha))
                    lower_bound -= len(na_train_triple)

                    for j, (h_idx, t_idx) in enumerate(na_triple[:lower_bound], len(train_tripe)+len(na_train_triple)):
                        ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])
                        relation_multi_label[i, j, 0] = 1
                        relation_label[i, j] = 0
                        relation_mask[i, j] = 1
                        relation_example_idx[i, j] = example['idx']

                    max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound + len(na_train_triple))
                else:
                    j = 0
                    for h_idx in range(L):
                        for t_idx in range(L):
                            if h_idx != t_idx:
                                ht_pairs[i, j, :] = torch.Tensor([h_idx + 1, t_idx + 1])
                                relation_mask[i, j] = 1
                                relation_example_idx[i, j] = example['idx']
                                j += 1

                    max_h_t_cnt = max(max_h_t_cnt, j)
                    L_vertex.append(L)
                    titles.append(example['title'])
                    indexes.append(self.batches_order[idx][i])

            context_word_mask = context_word_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()
            sub2word_list = [sw[:, :batch_max_length].cuda() for sw in sub2word_list]
            word2sent_list = [ws.cuda() for ws in word2sent_list]

            yield {'context_idxs': get_cuda(context_word_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_pos': get_cuda(context_pos_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_ner': get_cuda(context_ner_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_mask': get_cuda(context_word_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_length': get_cuda(context_word_length[:cur_bsz].contiguous()),
                   'h_t_pairs': get_cuda(ht_pairs[:cur_bsz, :max_h_t_cnt, :2]),
                   'relation_label': get_cuda(relation_label[:cur_bsz, :max_h_t_cnt]).contiguous(),
                   'relation_multi_label': get_cuda(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'relation_mask': get_cuda(relation_mask[:cur_bsz, :max_h_t_cnt]),
                   'relation_example_idx': relation_example_idx[:cur_bsz, :max_h_t_cnt],
                   'labels': label_list,
                   'graph2s': graphs,
                   'sub2words': sub2word_list,
                   'word2sents': word2sent_list,
                   'path2_table': path2_table,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'indexes': indexes,
                   }

    def feedback(self, m_preds, m_label, r_mask, h_t_pairs,
                 relation_example_idx):
        output_m = torch.argmax(m_preds, dim=-1)
        output_m = output_m.data.cpu().numpy()
        m_label = m_label.data.cpu().numpy()
        r_mask = r_mask.data.cpu().numpy()
        h_t_pairs = h_t_pairs.data.cpu().numpy()
        wrong_predits = {}
        for i in range(len(r_mask)):
            for j in range(len(r_mask[0])):
                idx = int(relation_example_idx[i, j])
                ent_0, ent_1 = h_t_pairs[i, j, 0], h_t_pairs[i, j, 1]
                if r_mask[i, j] == 1 and m_label[i, j, output_m[i, j]] == 0:
                    if idx not in wrong_predits:
                        wrong_predits[idx] = []
                    wrong_predits[idx].append((ent_0-1, ent_1-1))
        for idx in wrong_predits:
            self.data[idx]['wrong_predits'] = wrong_predits.get(idx, [])
