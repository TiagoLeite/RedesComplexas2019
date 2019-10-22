import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt
from pyunicorn.timeseries.visibility_graph import VisibilityGraph
from networkx.utils.rcm import *
import networkx as nx
import uuid
import random

random.seed(29)


def save_image_graph_from_series(series, label):
    g = VisibilityGraph(series)
    adj = g.adjacency
    g = nx.Graph(incoming_graph_data=adj)
    rcm = list(reverse_cuthill_mckee_ordering(g))
    adj2 = nx.adjacency_matrix(g, nodelist=rcm).toarray()
    random_img_name = 'img/' + str(label) + '/' + str(uuid.uuid4().hex)[:10] + '.png'
    plt.imsave(random_img_name, adj2, cmap='gray')
    # g2 = nx.Graph(incoming_graph_data=adj2)
    # print(nx.is_isomorphic(g, g2))


def build_image_dataset():
    data = pd.read_csv('signal.csv')
    total = len(data)
    for index, row in data.iterrows():
        print(index, 'of', total)
        label = row[-1]
        series = np.array(row[1:-1])
        save_image_graph_from_series(series, label)


build_image_dataset()
