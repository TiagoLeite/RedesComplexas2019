import numpy as np
import pandas as pd
from igraph import *
import seaborn as sns
import matplotlib.pyplot as plt
from pyunicorn.timeseries.visibility_graph import VisibilityGraph
from networkx.utils.rcm import *
import networkx as nx
import cv2 as cv

data = pd.read_csv('signal.csv')
# print(data.describe())
# print(data.info())
# print(data['y'].value_counts())

data1 = data[data['y'] == 1]
data2 = data[data['y'] == 2]
data3 = data[data['y'] == 3]
data4 = data[data['y'] == 4]
data5 = data[data['y'] == 5]


def plot_some_fig():
    plt.figure(figsize=(15, 10))
    plt.subplot(5, 2, 1)
    plt.plot(data1.iloc[0][1:].values)
    plt.subplot(5, 2, 2)
    plt.plot(data1.iloc[1][1:].values)

    plt.subplot(5, 2, 3)
    plt.plot(data2.iloc[0][1:].values)
    plt.subplot(5, 2, 4)
    plt.plot(data2.iloc[1][1:].values)

    plt.subplot(5, 2, 5)
    plt.plot(data3.iloc[0][1:].values)
    plt.subplot(5, 2, 6)
    plt.plot(data3.iloc[1][1:].values)

    plt.subplot(5, 2, 7)
    plt.plot(data4.iloc[0][1:].values)
    plt.subplot(5, 2, 8)
    plt.plot(data4.iloc[1][1:].values)

    plt.subplot(5, 2, 9)
    plt.plot(data5.iloc[0][1:].values)
    plt.subplot(5, 2, 10)
    plt.plot(data5.iloc[1][1:].values)

    plt.xlabel('Samples')
    plt.show()


series = data5.iloc[10][1:-1].values
print(np.shape(series))
# print(series)

g = VisibilityGraph(series)
adj = g.adjacency
print("Adj 1:", np.shape(adj))
plt.figure(figsize=(20, 10))

g1 = nx.Graph(incoming_graph_data=adj)
rcm = list(reverse_cuthill_mckee_ordering(g1))

adj2 = nx.adjacency_matrix(g1, nodelist=rcm).toarray()
g2 = nx.Graph(incoming_graph_data=adj2)

plt.subplot(2, 2, 1)
plt.imshow(adj, cmap='gray')

plt.subplot(2, 2, 2)
plt.imshow(adj2, cmap='gray')

plt.subplot(2, 2, 3)
plt.imshow(nx.adjacency_matrix(g1).toarray(), cmap='gray')

plt.imsave('g1.png', nx.adjacency_matrix(g1).toarray(), cmap='gray')
plt.imsave('g2.png', nx.adjacency_matrix(g2).toarray(), cmap='gray')


plt.subplot(2, 2, 4)
plt.imshow(nx.adjacency_matrix(g2).toarray(), cmap='gray')

unique, counts = np.unique(adj, return_counts=True)
print(unique, counts)
unique2, counts2 = np.unique(adj2, return_counts=True)
print(unique2, counts2)

print("Are they the same? ", nx.is_isomorphic(g1, g2))
print('G1:\n', nx.adjacency_matrix(g1).toarray())
print('\n')
print('G2:\n', nx.adjacency_matrix(g2).toarray())

plt.show()

print('\n=====\n', np.sum(nx.adjacency_matrix(g1).toarray() - adj))

















