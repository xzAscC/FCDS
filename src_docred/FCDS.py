from collections import defaultdict
import torch
import torch.nn as nn
from transformers import BertModel
from utils import GCNLayer, AttnLayer, filter_g


class FCDS_BERT(nn.Module):
    def __init__(self, config):
        super(FCDS_BERT, self).__init__()
        self.config = config
        self.activation = nn.ReLU()
        self.entity_type_emb = nn.Embedding(
            config.entity_type_num, config.entity_type_size,
            padding_idx=config.entity_type_pad)
        self.entity_id_emb = nn.Embedding(
            config.max_entity_num + 1, config.entity_id_size,
            padding_idx=config.entity_id_pad)

        if config.model_name == 'FCDS_BERT_base':
            self.bert = BertModel.from_pretrained(
                "bert-base-cased", return_dict=False)
        else:
            self.bert = BertModel.from_pretrained(
                "bert-large-cased", return_dict=False)

        self.start_dim = config.bert_hid_size

        # if config.use_entity_type:
        self.start_dim += config.entity_type_size + config.entity_id_size

        self.gcn_dim = config.gcn_dim

        self.start_gcn = GCNLayer(self.start_dim, self.gcn_dim)

        self.GCNs = nn.ModuleList([
            GCNLayer(
                self.gcn_dim, self.gcn_dim, activation=self.activation)
            for i in range(config.gcn_layers)])

        self.Attns = nn.ModuleList([
            AttnLayer(
                self.gcn_dim, activation=self.activation)
            for _ in range(config.gcn_layers)
        ])

        self.bank_size = self.start_dim + self.gcn_dim * (
            self.config.gcn_layers + 1)
        self.dropout = nn.Dropout(self.config.dropout)

        self.rnn = nn.LSTM(
            self.bank_size, self.bank_size, 2,
            bidirectional=False, batch_first=True)

        self.path_attn = nn.MultiheadAttention(self.bank_size, 4)

        self.predict2 = nn.Sequential(
            nn.Linear(self.bank_size*5, self.bank_size*5),
            self.activation,
            self.dropout,
        )

        self.out_linear = nn.Linear(self.bank_size * 5, config.relation_nums)
        self.out_linear_binary = nn.Linear(self.bank_size * 5, 2)

    def forward(self, **params):
        words = params['words'].cuda()
        mask = params['mask'].cuda()
        bsz = words.size(0)

        encoder_outputs, sent_cls = self.bert(
            input_ids=words, attention_mask=mask)
        encoder_outputs = torch.cat([
            encoder_outputs,
            self.entity_type_emb(params['entity_type']),
            self.entity_id_emb(params['entity_id'])], dim=-1)

        graphs = params['graph2s']
        sub2words = params['sub2words']
        features = []

        for i, graph in enumerate(graphs):
            encoder_output = encoder_outputs[i]
            sub2word = sub2words[i]
            x = torch.mm(sub2word, encoder_output)
            graph = filter_g(graph, x)
            xs = [x]
            x = self.start_gcn(graph, x)
            xs.append(x)
            for GCN, Attn in zip(self.GCNs, self.Attns):
                x1 = GCN(graph, x)
                x2 = Attn(x, x1, x1)
                x = x1 + x2
                xs.append(x)
            out_feas = torch.cat(xs, dim=-1)
            features.append(out_feas)

        h_t_pairs = params['h_t_pairs']
        h_t_pairs = h_t_pairs + (h_t_pairs == 0).long() - 1
        h_t_limit = h_t_pairs.size(1)
        path_info = torch.zeros((bsz, h_t_limit, self.bank_size)).cuda()
        rel_mask = params['relation_mask']
        path_table = params['path2_table']

        path_len_dict = defaultdict(list)

        entity_num = torch.max(params['entity_id'])
        entity_bank = torch.Tensor(bsz, entity_num, self.bank_size).cuda()

        for i in range(len(graphs)):
            max_id = torch.max(params['entity_id'][i])
            entity_feas = features[i][-max_id:]
            entity_bank[i, :entity_feas.size(0)] = entity_feas
            path_t = path_table[i]
            for j in range(h_t_limit):
                h_ent = h_t_pairs[i, j, 0].item()
                t_ent = h_t_pairs[i, j, 1].item()

                if rel_mask is not None and rel_mask[i, j].item() == 0:
                    break

                if rel_mask is None and h_ent == 0 and t_ent == 0:
                    continue

                # path = path_t[(h_ent+1, t_ent+1)]
                paths = path_t[(h_ent+1, t_ent+1)]
                for path in paths:
                    path = torch.LongTensor(path).cuda()
                    cur_h = torch.index_select(features[i], 0, path)
                    path_len_dict[len(path)].append((i, j, cur_h))

        h_ent_idx = h_t_pairs[:, :, 0].unsqueeze(-1).expand(
            -1, -1, self.bank_size)
        t_ent_idx = h_t_pairs[:, :, 1].unsqueeze(-1).expand(
            -1, -1, self.bank_size)
        h_ent_feas = torch.gather(input=entity_bank, dim=1, index=h_ent_idx)
        t_ent_feas = torch.gather(input=entity_bank, dim=1, index=t_ent_idx)

        path_embedding = {}

        for items in path_len_dict.values():
            cur_hs = torch.stack([h for _, _, h in items], 0)
            cur_hs2, _ = self.rnn(cur_hs)
            cur_hs = cur_hs2.max(1)[0]
            for idx, (i, j, _) in enumerate(items):
                if (i, j) not in path_embedding:
                    path_embedding[(i, j)] = []
                path_embedding[(i, j)].append(cur_hs[idx])

        querys = h_ent_feas - t_ent_feas

        for (i, j), emb in path_embedding.items():
            query = querys[i:i+1, j:j+1]
            keys = torch.stack(emb).unsqueeze(1)
            output, attn_weights = self.path_attn(query, keys, keys)
            path_info[i, j] = output.squeeze(0).squeeze(0)

        out_feas = torch.cat([
            h_ent_feas, t_ent_feas,
            torch.abs(h_ent_feas - t_ent_feas),
            torch.mul(h_ent_feas, t_ent_feas),
            path_info], dim=-1)
        out_feas = self.predict2(out_feas)
        m_preds = self.out_linear(out_feas)
        b_preds = self.out_linear_binary(out_feas)
        return m_preds, b_preds, None
