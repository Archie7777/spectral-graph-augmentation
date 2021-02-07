import os
import re
import time
import zipfile
import numpy as np
from math import *
import urllib.request
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from dgl.data import CitationGraphDataset
from utils import compute_ppr, normalize_adj, spectral_perturbation

graph_data = ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
node_data = ['cora', 'citeseer', 'pubmed']


class GraphData:

    def __init__(self, name):

        self.__name = name
        if name in graph_data:
            self.__graph_type = 'graph'
        elif name in node_data:
            self.__graph_type = 'node'
        else:
            print('Wrong dataset name!')
            self.__name = 'MUTAG'

        # download dataset
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        path = os.path.join(root, 'dataset', self.__graph_type)

        if not os.path.exists(path):
            os.makedirs(path)

        if not os.path.exists(os.path.join(path, self.__name)):
            if self.__name in graph_data:
                url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{0}.zip'.format(self.__name)
                zip_path = '{0}/{1}.zip'.format(path, self.__name)
                print('downloading {0}'.format(self.__name))
                urllib.request.urlretrieve(url, zip_path)

                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(path)
                os.remove(zip_path)

                # process the raw dataset
                self.graphs, self.aug_view = self.process_raw_data()
            else:
                self.cite_graph = CitationGraphDataset(name=self.__name)

    def process_raw_data(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
        prefix = os.path.join(root, 'dataset', self.__graph_type, self.__name, self.__name)
        print(prefix)

        graph_node_dict = {}
        with open('{0}_graph_indicator.txt'.format(prefix), 'r') as f:
            for idx, line in enumerate(f):
                graph_node_dict[idx + 1] = int(line.strip('\n'))
        max_nodes = Counter(graph_node_dict.values()).most_common(1)[0][1]

        node_labels = []
        if os.path.exists('{0}_node_labels.txt'.format(prefix)):
            with open('{0}_node_labels.txt'.format(prefix), 'r') as f:
                for line in f:
                    node_labels += [int(line.strip('\n'))]  # -1 ???????????
                num_unique_node_labels = max(node_labels) + 1  # ??????????????
        else:
            print('No node labels')

        node_attrs = []
        if os.path.exists('{0}_node_attributes.txt'.format(prefix)):
            with open('{0}_node_attributes.txt'.format(prefix), 'r') as f:
                for line in f:
                    node_attrs.append(
                        np.array([float(attr) for attr in re.split("[,\s]+", line.strip('\s\n')) if attr],
                                 dtype=np.float)
                    )
        else:
            print('No node attributes')

        graph_labels = []
        unique_labels = set()
        with open('{0}_graph_labels.txt'.format(prefix), 'r') as f:
            for line in f:
                val = int(line.strip('\n'))
                if val not in unique_labels:
                    unique_labels.add(val)
                graph_labels.append(val)
        label_idx_dict = {val: idx for idx, val in enumerate(unique_labels)}
        graph_labels = np.array([label_idx_dict[element] for element in graph_labels])

        adj_list = {idx: [] for idx in range(1, len(graph_labels) + 1)}  # 每个图包含的边
        index_graph = {idx: [] for idx in range(1, len(graph_labels) + 1)}  # 每个图包含的节点
        with open('{0}_A.txt'.format(prefix), 'r') as f:
            for line in f:
                u, v = tuple(map(int, line.strip('\n').split(',')))
                adj_list[graph_node_dict[u]].append((u, v))
                index_graph[graph_node_dict[u]] += [u, v]

        for k in index_graph.keys():
            index_graph[k] = [u - 1 for u in set(index_graph[k])]

        graphs, aug_view = [], []
        for idx in range(1, 1 + len(adj_list)):
            graph = nx.from_edgelist(adj_list[idx])
            if max_nodes is not None and graph.number_of_nodes() > max_nodes:
                continue

            graph.graph['label'] = graph_labels[idx - 1]
            for u in graph.nodes():
                if len(node_labels) > 0:
                    node_label_one_hot = [0] * num_unique_node_labels
                    node_label = node_labels[u - 1]
                    node_label_one_hot[node_label] = 1
                    graph.nodes[u]['label'] = node_label_one_hot
                if len(node_attrs) > 0:
                    graph.nodes[u]['feat'] = node_attrs[u - 1]
            if len(node_attrs) > 0:
                graph.graph['feat_dim'] = node_attrs[0].shape[0]

            # relabeling
            mapping = {}
            for node_idx, node in enumerate(graph.nodes()):
                mapping[node] = node_idx

            graphs.append(nx.relabel_nodes(graph, mapping))
            aug_view.append(compute_ppr(graph, alpha=0.2))

        if 'feat_dim' in graphs[0].graph:
            pass
        else:
            max_deg = max([max(dict(graph.degree).values()) for graph in graphs])
            for graph in graphs:
                for u in graph.nodes(data=True):
                    f = np.zeros(max_deg + 1)
                    f[graph.degree[u[0]]] = 1.0
                    if 'label' in u[1]:
                        f = np.concatenate((np.array(u[1]['label'], dtype=np.float), f))
                    graph.nodes[u[0]]['feat'] = f
        return graphs, aug_view

    def graph_augment(self):
        return 0

    def load_graph(self):
        return 0

    def save_graph(self):
        return 0

    def read_graph(self):
        return 0


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    a = 1
    # print(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
    ds = CitationGraphDataset(name='pubmed')
    # nx.draw(ds.graph, node_size=10)
    # plt.show()
    # print(nx.to_numpy_array(ds.graph).shape)
    print('start')
    start = time.time()
    # spectral_perturbation(ds.graph)
    compute_ppr(ds.graph, 0.2)
    finish = time.time()
    print('finish ', finish - start)
    # print(ds)
    # print(nx.to_numpy_array(ds.graph).shape)
    # print(n)
    # print(ds.features)

    # path = 'D:/VSrepos/spectral-graph-augmentation/dataset/node'
    # if not os.path.exists(path):
    #     print('create')
    #     os.makedirs(path)
    # else:
    #     print('exist')

    # GraphData('PTC_MR').process_raw_data()


