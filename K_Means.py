import math 
import random
from sklearn import datasets # For iris dataset.
from sklearn.cluster import KMeans # For Trained KMeans Classifier to compare with our classifier.

# load iris dataset
iris = datasets.load_iris()

# Storing cluster centers.
centers = []

# Storing cluster data.
cluster_number = [0] * len(iris.data)

# Randomly initializing centers of the clusters.
def initialize():
	for i in range(0, 3):
    # Since index starts from 0 in python, hence generating a random index from 0 to 149
		index = random.randint(0, 149)
		centers.append(iris['data'][index])			

initialize()

# Finding euclidean distance between points.
def euclidean_distance(data, centers):
	return math.sqrt((data[0] - centers[0]) ** 2 
                   + (data[1] - centers[1]) ** 2 
                   + (data[2] - centers[2]) ** 2 
                   + (data[3] - centers[3]) ** 2)


# Updating cluster centers to the average of all data points in that cluster.
def update_cluster_centers():
	sum = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
	count = [0, 0, 0]
	for index in range(0, len(iris.data)):
		for j in range(0, 4):
			sum[cluster_number[index]][j] += iris['data'][index][j]
		count[cluster_number[index]] += 1
	for i in range(0, 3):
		for j in range(0, 4):
			if count[i] != 0:
				sum[i][j] /= count[i]
	centers.clear()
	for i in range(0, 3):
		centers.append(sum[i])

# Check if two cluster centers are equal.
def check_equal(previous_centers, centers):
	for i in range(0, 3):
		for j in range(0, 4):
			if(centers[i][j] != previous_centers[i][j]):
				return False
	return True

# Until the cluster centers are same do clustering.  
while True:
	cluster_number = [0] * len(iris.data)
	previous_centers = []
	previous_centers = centers.copy()
	for index in range(0, len(iris.data)):
		distance = []
		for i in range(0, len(centers)):
			distance.append(euclidean_distance(iris['data'][index], centers[i]))
		cluster_number[index] = distance.index(min(distance))
	update_cluster_centers()
	if check_equal(previous_centers, centers):
		break

# Function to print number of data points in each cluster. 
def print_count(a, b):
	sum = [[0, 0, 0], [0, 0, 0]]
	for j in range(0, 2):
		for i in range(0, len(iris.data)):
			if j == 0:
				sum[j][a[i]] += 1
			else:
				sum[j][b[i]] += 1
	print("Number of data points in the three cluster are (for In built KMeans Classifier): ", sum[0])
	print("Number of data points in the three cluster are (for our KMeans Classifier): ", sum[1])

# Calling In built KMeans Classifier to fit iris data.
kmeans = KMeans(n_clusters = 3).fit(iris.data)
print_count(kmeans.labels_, cluster_number)               