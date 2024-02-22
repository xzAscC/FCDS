import argparse
import json
import os
import numpy as np


def get_opt():
    data_dir = '../data/'
    parser = argparse.ArgumentParser()
    # datasets path
    parser.add_argument('--train_set', type=str, default=os.path.join(
        data_dir, 'train_annotated.json'))
    parser.add_argument('--dev_set', type=str, default=os.path.join(
        data_dir, 'dev.json'))
    parser.add_argument('--test_set', type=str, default=os.path.join(
        data_dir, 'test.json'))

    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')

    # task/Dataset-related
    parser.add_argument('--relation_nums', type=int, default=97)
    parser.add_argument('--entity_type_num', type=int, default=7)
    parser.add_argument('--max_entity_num', type=int, default=80)

    # padding
    parser.add_argument('--word_pad', type=int, default=0)
    parser.add_argument('--entity_type_pad', type=int, default=0)
    parser.add_argument('--entity_id_pad', type=int, default=0)

    # word embedding
    parser.add_argument('--word_emb_size', type=int, default=10)

    # entity type embedding
    parser.add_argument('--use_entity_type', action='store_true')
    parser.add_argument('--entity_type_size', type=int, default=20)

    # entity id embedding, i.e., coreference embedding in DocRED original paper
    parser.add_argument('--use_entity_id', action='store_true')
    parser.add_argument('--entity_id_size', type=int, default=20)

    # BiLSTM
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_dropout', type=float, default=0.1)

    # training settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--negativa_alpha', type=float, default=0.0)
    parser.add_argument('--save_model_freq', type=int, default=5)

    # gcn
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.6)
    parser.add_argument('--activation', type=str, default="relu")

    # BERT
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--coslr', action='store_true')

    # use DeBERTa / BERT encoder, default: BERT encoder
    parser.add_argument(
        '--use_model', type=str, default="bert", choices=['glove', 'bert'])

    parser.add_argument('--input_theta', type=float, default=-1)

    opt = parser.parse_args()

    data_opt = Object()
    data_opt.data_dir = data_dir
    data_opt.rel2id = json.load(open(os.path.join(
        data_opt.data_dir, 'rel2id.json'), "r"))
    data_opt.id2rel = {v: k for k, v in data_opt.rel2id.items()}
    data_opt.word2id = json.load(open(os.path.join(
        data_opt.data_dir, 'word2id.json'), "r"))
    data_opt.ner2id = json.load(open(os.path.join(
        data_opt.data_dir, 'ner2id.json'), "r"))
    data_opt.word2vec = np.load(os.path.join(data_opt.data_dir, 'vec.npy'))
    return opt, data_opt


class Object(object):
    pass
