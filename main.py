import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt
from pyunicorn.timeseries.visibility_graph import VisibilityGraph
from networkx.utils.rcm import *
import networkx.generators as gen
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


def avg_degree(g):
    degrees = nx.degree(g)
    soma = 0
    size = len(degrees)
    for x in degrees:
        soma = soma + x[1]
    degrees_mean = soma/size
    return degrees_mean


def generate_networks():

    for k in range(1000):

        print(k)

        g = gen.watts_strogatz_graph(n=178, k=random.randrange(10, 100), p=.5)
        h = gen.barabasi_albert_graph(n=178, m=random.randrange(10, 100))
        j = gen.erdos_renyi_graph(n=178, p=random.random())
        m = gen.random_regular_graph(d=random.randrange(10, 100), n=178)

        rcm = list(reverse_cuthill_mckee_ordering(g))
        g_adj = nx.adjacency_matrix(g, nodelist=rcm).toarray()

        rcm = list(reverse_cuthill_mckee_ordering(h))
        h_adj = nx.adjacency_matrix(h, nodelist=rcm).toarray()

        rcm = list(reverse_cuthill_mckee_ordering(j))
        j_adj = nx.adjacency_matrix(j, nodelist=rcm).toarray()

        rcm = list(reverse_cuthill_mckee_ordering(m))
        m_adj = nx.adjacency_matrix(m, nodelist=rcm).toarray()

        plt.imsave('img/0/'+str(k)+'.jpg', g_adj, cmap='gray')
        plt.imsave('img/1/'+str(k)+'.jpg', h_adj, cmap='gray')
        plt.imsave('img/2/'+str(k)+'.jpg', j_adj, cmap='gray')
        plt.imsave('img/3/'+str(k)+'.jpg', m_adj, cmap='gray')

        # print(nx.number_of_edges(g), nx.number_of_edges(h), nx.number_of_edges(j))

        # print(avg_degree(g), avg_degree(h), avg_degree(j), avg_degree(m))


generate_networks()












