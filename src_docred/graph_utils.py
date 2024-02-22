import spacy
import dgl
from spacy.tokens import Doc
import networkx as nx
import matplotlib.pyplot as plt

nlp = spacy.load('en_core_web_lg')
# nlp.tokenizer = nlp.tokenizer.tokens_from_list

def build_g(sentences, pos_idx, max_id):
    # sentences should be a list of word lists
    # [[sent_1], [sent_2], ..., [sent_m]]
    # senti = [w_0, w_1, ..., w_n]
    pre_roots = []
    g = nx.DiGraph()
    docs = [Doc(nlp.vocab, words=ws) for ws in sentences]
    # tokens = parser(doc)
    for tokens in nlp.pipe(docs):
        g, pre_roots = parse_sent(tokens, g, pre_roots)
    # g = add_same_words_links(g, remove_stopwords=True)
    g, start = add_entity_node(g, pos_idx, max_id)
    paths = get_entity_paths(g, max_id, start)
    g = dgl.from_networkx(g)
    return g, paths


def parse_sent(tokens, g, pre_roots):
    roots = [token for token in tokens if token.head == token]
    start = len(g.nodes())
    dic = {}
    idx = 0
    for token in tokens:
        is_root = token in roots
        g.add_node(start + idx, text=token.text, vector=token.vector,
                   is_root=is_root, tag=token.tag_, pos=token.pos_,
                   dep=token.dep_)
        dic[token] = start + idx
        idx += 1

    for token in tokens:
        g.add_edge(dic[token], dic[token.head], dep=token.dep_)
        g.add_edge(dic[token.head], dic[token], dep=token.dep_)

    for idx, root in enumerate(roots[:-1]):
        g.add_edge(dic[root], dic[roots[idx+1]], dep=token.dep_)
        g.add_edge(dic[roots[idx+1]], dic[root], dep=token.dep_)

    if pre_roots:
        pre_root_idx = pre_roots[-1]
        for root in roots[:1]:
            g.add_edge(dic[root], pre_root_idx, dep='rootconn')
            g.add_edge(pre_root_idx, dic[root], dep='rootconn')
    for pre_root_idx in pre_roots:
        # pre_root_idx = pre_roots[-2]
        # g.add_edge(dic[root], pre_root_idx, dep='rootconn')
        for root in roots[:1]:
            g.add_edge(pre_root_idx, dic[root], dep='rootconn')
    for root in roots[:1]:
        pre_roots.append(dic[root])
    return g, pre_roots


def add_same_words_links(g, remove_stopwords=True):
    names = nx.get_node_attributes(g, "text")
    name_dic = {}
    stopwords = nlp.Defaults.stop_words
    for idx, name in names.items():
        name = name.lower()
        if remove_stopwords and name in stopwords:
            continue
        if len(name) < 5:
            continue
        if name not in name_dic:
            name_dic[name] = [idx]
        else:
            for pre_idx in name_dic[name]:
                g.add_edge(idx, pre_idx)
                g.add_edge(pre_idx, idx)
            name_dic[name].append(idx)
    return g


def add_entity_node(g, pos_idx, max_id):
    start = len(g.nodes())
    for idx in range(max_id):
        g.add_node(start + idx, text='entity_%s' % idx, is_root=False)
        if idx + 1 not in pos_idx:
            continue
        for idx2, node in enumerate(pos_idx[idx + 1]):
            g.add_edge(start + idx, node)
            g.add_edge(node, start + idx)
    return g, start


def get_entity_paths(g, max_id, start):
    ent_paths = {}
    pos_data = nx.get_node_attributes(g, 'pos')
    for idx in range(max_id):
        for j in range(max_id):
            if idx != j:
                try:
                    # paths = list(nx.all_simple_paths(g, start + idx, start + j, 8))
                    # if len(paths) == 0:
                    paths = [nx.shortest_path(g, start + idx, start + j)]
                except nx.NetworkXNoPath:
                    paths = []
                new_paths = []
                # for path in paths:
                #     path = sorted(path)
                #     new_paths.append(path)

                for path in paths[:3]:
                    # add immediate neighbors for nodes on the path
                    neighbors = set()
                    # neighbors = list()
                    for n in path:
                        neighbors.add(n)
                        for cur_neigh in g.neighbors(n):
                            if cur_neigh in pos_data:
                                if pos_data[cur_neigh] in ['ADP']:
                                    neighbors.add(cur_neigh)
                                # if dep_data[cur_neigh] in ['neg']:
                                #     neighbors.add(cur_neigh)
                    path = sorted(neighbors)
                    # path = sorted(path)
                    new_paths.append(path)
                # except nx.NetworkXNoPath:
                #     new_paths = [[start + idx] + roots_path + [start + j]]
                #     # path = [start+idx] + roots_path + [start+j]
                #     # path = [start+idx, start+j]

                ent_paths[(idx + 1, j + 1)] = new_paths[:3]
    return ent_paths


def draw_graph(G):
    pos = nx.kamada_kawai_layout(G)
    node_colors = ['gray'] * len(G.nodes)
    edge_color = ['gray'] * len(G.edges)
    node_labels = nx.get_node_attributes(G, 'text')
    nx.draw_networkx(
        G, pos, node_size=30, labels=node_labels, font_size=7,
        node_color=node_colors, font_color='purple', edge_color=edge_color)
    plt.savefig('test.png')
