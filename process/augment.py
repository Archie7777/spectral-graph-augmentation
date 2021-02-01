import os
import re
import numpy as np
import networkx as nx
from collections import Counter
from utils import compute_ppr, normalize_adj, spectral_perturbation
import matplotlib.pyplot as plt
from math import *


class GraphData:

    def __init__(self, name):
        self.__name = name
        if name in ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
            self.__graph_type = 'graph'
        elif name in ['cora', 'citeseer', 'pubmed']:
            self.__graph_type = 'node'
        else:
            print('Wrong dataset name!')
            self.__name = 'MUTAG'

        path = os.path.dirname(__file__)

        self.download_dataset(path)


    def download_dataset(self):
        return 0

    def process(self):
        return 0

    def graph_augment(self):
        return 0

    def load_graph(self):
        return 0

    def save_graph(self):
        return 0

    def read_graph(self):
        return 0


if __name__ == '__main__':
    a = 1
    print(os.path.dirname(__file__))

