import numpy as np
import math
from sklearn import datasets

iris, number_of_clusters = {}, 0
size, distance = 0, []
# Array to keep track of which point is in which cluster
cluster_index = []

def initialize():
    global iris, number_of_clusters, distance, size, cluster_index
    # Initialize centers of the clusters.
    iris = datasets.load_iris()
    # Initially each data point would be one cluster.
    number_of_clusters = len(iris['data'])

    size = (len(iris['data']), len(iris['data']))
    # Initialize the distance matrix with zeros.
    distance = np.zeros(size)
    np.fill_diagonal(distance, np.inf)

    # Initialize each point as in it's own cluster.
    for i in range(len(iris['data'])):
        cluster_index.append([i])

initialize()

def euclidean_distance(a, b):
    dist = 0
    for i in range(0, len(a)):
        dist += (a[i] - b[i]) ** 2
    return math.sqrt(dist)

def initialize_distance():
    for i in range(len(iris['data'])):
        for j in range(i+1, len(iris['data'])):
            distance[i][j] = euclidean_distance(iris['data'][i], iris['data'][j])
            distance[j][i] = distance[i][j]

initialize_distance()


def update_distance(index):
    for i in range(len(iris['data'])):
        dist1 = []
        dist2 = []
        for k in cluster_index[i]:
            dist1.append(euclidean_distance(iris['data'][k], iris['data'][index[0]]))
            dist2.append(euclidean_distance(iris['data'][k], iris['data'][index[1]]))
        distance[i][index[0]] = min(min(dist1), min(dist2))
        distance[i][index[1]] = distance[i][index[0]]
        distance[i][index[0]] = distance[index[0]][i]
        distance[i][index[1]] = distance[index[1]][i]
        # print(dist1, dist2, distance[i][j])
    for i in cluster_index[index[0]]:
        for j in cluster_index[index[1]]:
            distance[i][j] = np.inf

while number_of_clusters != 3:
    min_dist = distance.min()
    index = np.unravel_index(np.argmin(distance, axis=None), distance.shape)

    update_distance(index)
    for i in cluster_index[max(index[0], index[1])]:
        cluster_index[min(index[0], index[1])].append(i)
    np.delete(cluster_index, max(index[0], index[1]))
    number_of_clusters -= 1
    print(cluster_index)

print(cluster_index, len(cluster_index))
