import json
from tqdm import tqdm
import numpy as np
import torch
from config import get_opt
from data import DGLREDataloader, BERTDGLREDataset
from FCDS import FCDS_BERT
from utils import logging


def get_test_results(model, dataloader, relation_num, id2rel):
    test_result_dic = {}
    total_recall_dic = {}
    total_recall = 0

    for d in tqdm(dataloader, unit='b'):
        labels = d['labels']
        L_vertex = d['L_vertex']
        titles = d['titles']
        indexes = d['indexes']
        with torch.no_grad():
            m_preds, b_preds, _ = model(
                words=d['context_idxs'],
                src_lengths=d['context_word_length'],
                mask=d['context_word_mask'],
                entity_type=d['context_ner'],
                entity_id=d['context_pos'],
                h_t_pairs=d['h_t_pairs'],
                relation_mask=None,
                graph2s=d['graph2s'],
                path2_table=d['path2_table'],
                sub2words=d['sub2words'])
            predict_re = torch.sigmoid(m_preds)
            b_preds_re = torch.softmax(b_preds, dim=-1)

        predict_re = predict_re.data.cpu().numpy()
        b_preds_re = b_preds_re.data.cpu().numpy()

        for i in range(len(labels)):
            label = labels[i]
            L = L_vertex[i]
            title = titles[i]
            index = indexes[i]
            total_recall += len(label)
            for _, _, r in label:
                total_recall_dic[r] = total_recall_dic.get(r, 0) + 1
            j = 0

            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:
                        cur_results = []
                        for r in range(1, relation_num):
                            rel_ins = (h_idx, t_idx, r)
                            intrain = label.get(rel_ins, False)
                            cur_result = (
                                rel_ins in label,
                                float(predict_re[i, j, r]-predict_re[i, j, 0]),
                                float(b_preds_re[i, j, 1]), intrain, title,
                                id2rel[r], index, h_idx, t_idx, r)
                            # if predict_re[i, j, r] > predict_re[i, j, 0]:
                            cur_results.append(cur_result)
                        cur_results.sort(key=lambda x: x[1], reverse=True)
                        for cur_result in cur_results[:3]:
                            r = cur_result[-1]
                            if r not in test_result_dic:
                                test_result_dic[r] = []
                            test_result_dic[r].append(cur_result)
                        j += 1
    return test_result_dic, total_recall_dic, total_recall


def get_results_per_class(test_result_dic, total_recall_dic, test_theta=None):
    test_result = []
    new_test_theta = {}
    for r, results in test_result_dic.items():
        cur_pr_x = []
        cur_pr_y = []
        cur_correct = 0
        results.sort(key=lambda x: x[1] + 0.3 * x[2], reverse=True)
        # results.sort(key=lambda x: x[1], reverse=True)
        if test_theta is None:
            for i, item in enumerate(results):
                cur_correct += item[0]
                cur_pr_y.append(float(cur_correct) / (i + 1))  # Precision
                cur_pr_x.append(float(cur_correct) / total_recall_dic[r])  # Recall
            cur_pr_x = np.asarray(cur_pr_x, dtype='float32')
            cur_pr_y = np.asarray(cur_pr_y, dtype='float32')
            cur_f1_arr = (2 * cur_pr_x * cur_pr_y / (cur_pr_x + cur_pr_y + 1e-20))
            cur_f1 = cur_f1_arr.max()
            cur_f1_pos = cur_f1_arr.argmax()
            cur_theta = results[cur_f1_pos][1] + 0.3 * results[cur_f1_pos][2]
            if cur_f1 > 0.27:
                test_result.extend(results[:cur_f1_pos+1])
                new_test_theta[r] = cur_theta
        else:
            if str(r) not in test_theta:
                continue
            for i, item in enumerate(results):
                if item[1] + 0.3 * item[2] < test_theta[str(r)]:
                    break
            test_result.extend(results[:i+1])
    return test_result, new_test_theta


def test(model, dataloader, id2rel, input_theta=-1, output=False,
         test_prefix='dev', relation_num=97, test_theta=None, test_bound=None):

    test_result_dic, total_recall_dic, total_recall = get_test_results(
        model, dataloader, relation_num, id2rel)

    test_result = []
    for results in test_result_dic.values():
        test_result.extend(results)

    # test_result, test_theta = get_results_per_class(
    #     test_result_dic, total_recall_dic, test_theta)

    test_result.sort(key=lambda x: x[1], reverse=True)

    pr_x = []
    pr_y = []
    correct = 0
    w = 0

    if test_bound:
        w = test_bound
    else:
        if total_recall == 0:
            total_recall = 1

        for i, item in enumerate(test_result[:20000]):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  # Precision
            pr_x.append(float(correct) / total_recall)  # Recall
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype='float32')
        pr_y = np.asarray(pr_y, dtype='float32')
        f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
        # f1 = f1_arr[-1]
        # f1_pos = len(f1_arr) - 1
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()

        theta = test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos
            input_theta = theta

        logging('F1 {:3.4f} | P {:3.4f} | R {:3.4f}'.format(f1, pr_y[w], pr_x[w]))

    print('=========+> total: %s, selected: %s' % (len(test_result), w+1))

    if output:
        output = [{'correct': x[0], 'index': x[-4], 'h_idx': x[-3],
                   't_idx': x[-2], 'r_idx': x[-1],
                   'score': x[1], 'bscore':x[2], 'intrain': x[3],
                   'r': x[-5], 'title': x[-6]} for x in test_result[:w + 1]]
        json.dump(output, open(test_prefix + "_index.json", "w"))

    if not test_bound:
        return f1, input_theta, test_theta


if __name__ == '__main__':
    opt, data_opt = get_opt()
    if opt.use_model == 'bert':
        train_set = BERTDGLREDataset(
            opt.train_set, data_opt.ner2id, data_opt.rel2id, dataset='train',
            model_name=opt.model_name)
        dev_set = BERTDGLREDataset(
            opt.dev_set, data_opt.ner2id, data_opt.rel2id, dataset='dev',
            instance_in_train=train_set.instance_in_train,
            model_name=opt.model_name)
        test_set = BERTDGLREDataset(
            opt.test_set, data_opt.ner2id, data_opt.rel2id, dataset='test',
            instance_in_train=train_set.instance_in_train,
            model_name=opt.model_name)
        model = FCDS_BERT(opt)

    # dev_loader = DGLREDataloader(
    #     dev_set, batch_size=opt.test_batch_size, dataset_type='test')
    test_loader = DGLREDataloader(
        test_set, batch_size=opt.test_batch_size, dataset_type='test')

    chkpt = torch.load(opt.pretrain_model, map_location=torch.device('cpu'))
    model.load_state_dict(chkpt['checkpoint'])
    logging('load checkpoint from {}'.format(opt.pretrain_model))

    model = model.cuda()
    model.eval()

    # f1, input_theta, test_theta = test(
    #     model, dev_loader, id2rel=data_opt.id2rel,
    #     input_theta=opt.input_theta, output=False, test_prefix='dev')
    # json.dump([input_theta, test_theta], open("dev_theta.json", "w"))
    input_theta, test_theta = json.load(open("dev_theta.json"))
    test(
        model, test_loader, id2rel=data_opt.id2rel, input_theta=input_theta,
        output=True, test_prefix='test', test_theta=test_theta,
        relation_num=opt.relation_nums,
        # test_bound=12626)
        test_bound=20000)
    print('finished')
