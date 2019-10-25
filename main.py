import numpy as np
import pandas as pd
# from jgraph import *
# import seaborn as sns
import matplotlib.pyplot as plt
# from pyunicorn.timeseries.visibility_graph import VisibilityGraph
from networkx.utils.rcm import *
import networkx.generators as gen
import networkx as nx
import uuid
import random
from numpy.random import seed
# from node2vec import Node2Vec


random.seed(476)
seed(1453)


'''def save_image_graph_from_series(series, label):
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
        save_image_graph_from_series(series, label)'''


def avg_degree(g):
    degrees = nx.degree(g)
    soma = 0
    size = len(degrees)
    for x in degrees:
        soma = soma + x[1]
    degrees_mean = soma / size
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

        plt.imsave('img/0/' + str(k) + '.jpg', g_adj, cmap='gray')
        plt.imsave('img/1/' + str(k) + '.jpg', h_adj, cmap='gray')
        plt.imsave('img/2/' + str(k) + '.jpg', j_adj, cmap='gray')
        plt.imsave('img/3/' + str(k) + '.jpg', m_adj, cmap='gray')
        # print(nx.number_of_edges(g), nx.number_of_edges(h), nx.number_of_edges(j))
        # print(avg_degree(g), avg_degree(h), avg_degree(j), avg_degree(m))


def parse_files(folder_name, base_name):
    indicator = pd.read_csv(folder_name + '/' + base_name + '_graph_indicator.csv')
    labels = pd.read_csv(folder_name + '/' + base_name + '_graph_labels.csv')
    graph_ids = list((indicator['graph_id']))
    graph_class = [labels.iloc[graph_id - 1].values[0] for graph_id in graph_ids]
    indexes = list(np.arange(1, len(graph_ids) + 1))
    dataframe = pd.DataFrame(data={'node_id': indexes, 'graph_id': graph_ids, 'class': graph_class})
    dataframe.to_csv(folder_name + '/' + base_name + '_node_graph_class.csv', index=False)


def build_graphs_from_files(folder_name, base_name, n_classes):
    parse_files(folder_name, base_name)
    graph_data = pd.read_csv(folder_name + '/' + base_name + '_node_graph_class.csv')
    edges = pd.read_csv(folder_name + '/' + base_name + '_A.csv')
    total_graphs = len(graph_data['graph_id'].unique())
    print(total_graphs)
    last_id = -1
    all_graphs = [list() for _ in range(n_classes)]
    total = len(edges)
    for index, row in edges.iterrows():
        print(index, 'of', total)
        node1 = int(row['node1'])
        node2 = int(row['node2'])
        graph_id, graph_class = graph_data[graph_data['node_id'] == node1].iloc[0][['graph_id', 'class']]
        if last_id != graph_id:
            last_id = graph_id
            current_graph = nx.Graph()
            all_graphs[graph_class - 1].append(current_graph)

        if node1 not in current_graph:
            current_graph.add_node(node1)

        if node2 not in current_graph:
            current_graph.add_node(node2)

        current_graph.add_edge(node1, node2)

    return all_graphs


def build_images(all_graphs, folder_name):
    class_count = -1
    for graphs in all_graphs:  # gets all graphs from the same class
        class_count += 1
        k = 0
        for g in graphs:
            rcm = list(reverse_cuthill_mckee_ordering(g))
            g_adj = nx.adjacency_matrix(g, nodelist=rcm).toarray()
            plt.imsave(folder_name + '/img/' + str(class_count) + '/' + str(k) + '.jpg', g_adj, cmap='gray')
            k += 1
        print('Created', k, 'jpg images in folder ', folder_name + '/img/' + str(class_count) + '/')


N_CLASSES = 2
all_graphs = build_graphs_from_files('proteins', 'PROTEINS_full', N_CLASSES)
build_images(all_graphs, 'proteins')
nodes = 0
for k in range(N_CLASSES):
    graphs_class = all_graphs[k]
    nodes += np.sum([len(j.nodes) for j in graphs_class])
total_graphs = np.sum(len(graphs_class) for graphs_class in all_graphs)
nodes_mean = nodes / total_graphs
print("Total graphs:", total_graphs)
print("Avg nodes:", nodes_mean)
