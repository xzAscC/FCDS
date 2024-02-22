import time
import os
import torch
from tqdm import tqdm
from torch import nn
from torch import optim
from transformers import optimization
from config import get_opt
from data import DGLREDataloader, BERTDGLREDataset, GloveDGLREDataset
from FCDS import FCDS_BERT
from test import test
from utils import Accuracy, get_cuda, logging, set_random, ATLoss


def train(opt, data_opt):
    if opt.use_model == 'bert':
        print('Training with BERT==============')
        train_set = BERTDGLREDataset(
            opt.train_set, data_opt.ner2id, data_opt.rel2id, dataset='train',
            model_name=opt.model_name)
        dev_set = BERTDGLREDataset(
            opt.dev_set, data_opt.ner2id,
            data_opt.rel2id, dataset='dev',
            instance_in_train=train_set.instance_in_train,
            model_name=opt.model_name)

        train_loader = DGLREDataloader(
            train_set, batch_size=opt.batch_size, shuffle=True,
            negativa_alpha=opt.negativa_alpha)
        dev_loader = DGLREDataloader(
            dev_set, batch_size=opt.test_batch_size, dataset_type='dev')

        model = FCDS_BERT(opt)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load model from {}'.format(pretrain_model))
        start_epoch = chkpt['epoch'] + 1
        lr = chkpt['lr']
        logging('resume from epoch {} with lr {}'.format(start_epoch, lr))
    else:
        logging('training from scratch with lr {}'.format(lr))

    model = get_cuda(model)

    if opt.use_model == 'bert':
        bert_param_ids = list(map(id, model.bert.parameters()))
        base_params = filter(lambda p: p.requires_grad and id(p) not in bert_param_ids, model.parameters())
        optimizer = optimization.AdamW([
            {'params': model.bert.parameters(), 'lr': 0.00005},
            {'params': base_params, 'weight_decay': opt.weight_decay}
        ], lr=lr)
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=opt.weight_decay)

    pos_weight = [1] * 97
    pos_weight[0] = 0.0
    BCE = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight).cuda(), reduction='none')
    # BCE = nn.BCELoss(
    #     weight=torch.tensor(pos_weight).cuda(), reduction='none')
    EntropyLoss = nn.CrossEntropyLoss(reduction='none')
    Aloss = ATLoss()

    if opt.coslr:
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
            # optimizer, T_max=(opt.epoch // 4) + 1)
        total_steps = len(train_loader) * opt.epoch
        warmup_steps = int(total_steps * 0.06)
        scheduler = optimization.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min')

    checkpoint_dir = opt.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    best_ign_f1 = 0.0
    best_epoch = 0

    model.train()

    acc_NA, acc_not_NA, acc_total = Accuracy(), Accuracy(), Accuracy()
    logging('begin..')

    for epoch in range(start_epoch, opt.epoch + 1):
        total_loss = 0
        re_loss = 0
        re_loss2 = 0
        for acc in [acc_NA, acc_not_NA, acc_total]:
            acc.clear()

        for d in tqdm(train_loader, desc='Epoch '+str(epoch), unit='b'):
            m_label = d['relation_multi_label']
            r_mask = d['relation_mask']
            b_label = d['relation_label']
            relation_example_idx = d['relation_example_idx']

            m_preds, _, _ = model(
                words=d['context_idxs'],
                src_lengths=d['context_word_length'],
                mask=d['context_word_mask'],
                entity_type=d['context_ner'],
                entity_id=d['context_pos'],
                h_t_pairs=d['h_t_pairs'],
                relation_mask=r_mask,
                graph2s=d['graph2s'],
                path2_table=d['path2_table'],
                sub2words=d['sub2words'])

            # train_loader.feedback(
            #     m_preds, m_label, r_mask, d['h_t_pairs'], relation_example_idx)

            loss2 = Aloss(
                m_preds.view(-1, 97), m_label.view(-1, 97), r_mask.view(-1))
            loss1 = torch.sum(BCE(
                m_preds, m_label) * r_mask.unsqueeze(2)) / (
                    opt.relation_nums * torch.sum(r_mask))
            # loss2 = torch.sum(EntropyLoss(
            #     b_preds.view(-1, 2), b_label.view(-1)) * r_mask.view(-1)) / (
            #         opt.relation_nums * torch.sum(r_mask)
            # )
            loss = 0.1 * loss1 + loss2
            # loss = loss2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if opt.coslr:
                scheduler.step()

            output_m = torch.argmax(m_preds, dim=-1)
            # output_b = torch.argmax(b_preds, dim=-1)
            output_m = output_m.data.cpu().numpy()
            # output_b = output_b.data.cpu().numpy()
            m_label = m_label.data.cpu().numpy()
            b_label = b_label.data.cpu().numpy()
            for i in range(output_m.shape[0]):
                for j in range(output_m.shape[1]):
                    label = b_label[i][j]
                    if label < 0:
                        break
                    if label == 0 and output_m[i][j] == 0:
                        is_correct = True
                    elif label == 1 and m_label[i, j, output_m[i][j]] == 1:
                        is_correct = True
                    else:
                        is_correct = False
                    if label == 0:
                        acc_NA.add(is_correct)
                    else:
                        acc_not_NA.add(is_correct)
                    acc_total.add(is_correct)

            total_loss += loss.item()
            re_loss += loss1.item()
            re_loss2 += loss2.item()

        batch_num = len(train_loader)
        log_str = '| epoch {:2d} | loss_re {:5.3f} | loss_re2 {:5.3f} | ' + \
            'NA acc: {:4.2f} | noNA acc: {:4.2f}  | tot acc: {:4.2f} '
        logging(
            log_str.format(
                epoch, re_loss * 100 / batch_num, re_loss2 / batch_num, acc_NA.get(),
                acc_not_NA.get(), acc_total.get()))

        if epoch > 25 or epoch % opt.test_epoch == 0:
            logging('start testing ' + '-' * 50)
            eval_start_time = time.time()
            model.eval()
            ign_f1, _, _ = test(
                model, dev_loader, id2rel=data_opt.id2rel)
            if not opt.coslr:
                scheduler.step(ign_f1)
            model.train()
            logging('end testing | epoch {:3d} | time: {:5.2f}s'.format(
                epoch, time.time() - eval_start_time))

            if ign_f1 > best_ign_f1:
                best_ign_f1 = ign_f1
                best_epoch = epoch
                path = os.path.join(checkpoint_dir, model_name + '_best.pt')
                torch.save({
                    'epoch': epoch,
                    'checkpoint': model.state_dict(),
                    'lr': lr,
                    'best_ign_f1': ign_f1,
                    'best_epoch': epoch
                }, path)
            print('best_ign_f1 -------------------->', best_epoch, best_ign_f1)

        if epoch % opt.save_model_freq == 0:
            path = os.path.join(
                checkpoint_dir, model_name + '_{}.pt'.format(epoch))
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'checkpoint': model.state_dict()
            }, path)

    print("Finish training")
    print("Best epoch = %d | Best Ign F1 = %f" % (best_epoch, best_ign_f1))
    print("Storing best result...")
    print("Finish storing")


if __name__ == '__main__':
    set_random(432)
    opt, data_opt = get_opt()
    train(opt, data_opt)
