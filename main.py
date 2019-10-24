import numpy as np
import pandas as pd
# from jgraph import *
import seaborn as sns
import matplotlib.pyplot as plt
# from pyunicorn.timeseries.visibility_graph import VisibilityGraph
from networkx.utils.rcm import *
import networkx.generators as gen
import networkx as nx
import uuid
import random
# from node2vec import Node2Vec


random.seed(29)


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


def parse_files():
    indicator = pd.read_csv('imdb/IMDB-MULTI_graph_indicator.csv')
    labels = pd.read_csv('imdb/IMDB-MULTI_graph_labels.csv')
    graph_ids = list((indicator['graph_id']))
    graph_class = [labels.iloc[graph_id - 1].values[0] for graph_id in graph_ids]
    indexes = list(np.arange(1, len(graph_ids) + 1))
    dataframe = pd.DataFrame(data={'node_id': indexes, 'graph_id': graph_ids, 'class': graph_class})
    dataframe.to_csv('imdb/IMDB-MULTI_node_graph_class.csv', index=False)


def build_graphs():
    graph_data = pd.read_csv('graph_data/proteins/node_graph_class.csv')
    edges = pd.read_csv('graph_data/proteins/PROTEINS_full_A.csv')
    total_graphs = len(graph_data['graph_id'].unique())
    print(total_graphs)
    last_id = -1
    all_graps1, all_graps2 = [], []
    total = len(edges)
    for index, row in edges.iterrows():
        print(index, 'of', total)
        node1 = int(row['node1'])
        node2 = int(row['node2'])
        graph_id, graph_class = graph_data[graph_data['node_id'] == node1].iloc[0][['graph_id', 'class']]
        # graph_class = graph_data[graph_data['node_id'] == node1].iloc[0]['class']
        if last_id != graph_id:
            last_id = graph_id
            current_graph = nx.Graph()

            if graph_class == 1:
                all_graps1.append(current_graph)
            else:
                all_graps2.append(current_graph)

        if node1 not in current_graph:
            current_graph.add_node(node1)

        if node2 not in current_graph:
            current_graph.add_node(node2)

        current_graph.add_edge(node1, node2)

    return all_graps1, all_graps2


parse_files()

'''
graphs1, graphs2 = build_graphs()

print("Total graphs:", len(graphs1) + len(graphs2))
#nodes_avg = [len(g.nodes) for g in graphs]
#print(sorted(nodes_avg))
#nodes_avg = sorted(nodes_avg)
#unique, counts = np.unique(nodes_avg, return_counts=True)
#print(np.asarray((unique, counts)).T)

k = 0
for g in graphs1:
    rcm = list(reverse_cuthill_mckee_ordering(g))
    g_adj = nx.adjacency_matrix(g, nodelist=rcm).toarray()
    plt.imsave('protein_img/1/' + str(k) + '.jpg', g_adj, cmap='gray')
    k += 1


k = 0
for g in graphs2:
    rcm = list(reverse_cuthill_mckee_ordering(g))
    g_adj = nx.adjacency_matrix(g, nodelist=rcm).toarray()
    plt.imsave('protein_img/2/' + str(k) + '.jpg', g_adj, cmap='gray')
    k += 1


'''

'''
for g in graphs:
    adj = nx.adjacency_matrix(g).toarray()
    print(adj)
    print(np.shape(adj))'''

# generate_networks()
'''graph = nx.barabasi_albert_graph(n=178, m=14, seed=5)
node2vec = Node2Vec(graph, dimensions=64, workers=8)
model = node2vec.fit()
emb = model.wv.get_vector('12')
print(np.shape(emb))
print(emb)'''
