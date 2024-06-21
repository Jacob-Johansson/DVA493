import random

import numpy
import pandas
import sklearn
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Function for dynamically creating a matrix
def create_matrix(n, m):
    val = [0] * n
    for x in range(n):
        val[x] = [0] * m
    return val

def k_means_clustering(k, data):

    principalComponents = data.drop(labels=['Class'], axis=1)
    classes = data.loc[:, ['Class']]
    clusterIndices = pandas.DataFrame(numpy.zeros(len(principalComponents)), columns=['Cluster index'], dtype=int)

    numPrincipalComponents = len(principalComponents.columns)

    objects = pandas.concat([principalComponents, classes, clusterIndices], axis=1)
    objectClusterIndices = objects.loc[:, 'Cluster index'].values

    clusters = pandas.DataFrame(create_matrix(k, numPrincipalComponents), columns=principalComponents.columns, dtype=float)
    clusters = pandas.concat([clusters, pandas.DataFrame(numpy.zeros(k, dtype=int), columns=['Number of objects'])], axis=1)

    # Initializes clusters
    minPC = []
    maxPC = []
    for pc in principalComponents.columns.values:
        pcValues = objects.loc[:, pc].values
        minPC.append(min(pcValues))
        maxPC.append(max(pcValues))

    pcIndex = 0
    for pc in principalComponents.columns:
        pcArray = clusters.loc[:, pc].values
        for i in range(0, len(pcArray)):
            pcArray[i] = random.uniform(minPC[pcIndex], maxPC[pcIndex])
        pcIndex = pcIndex + 1

    # The algorithm
    distances = pandas.DataFrame(create_matrix(len(clusters), len(objects)), dtype=float)
    distanceColumns = distances.columns.values

    for i in range(0, 100):
        # Calculate the closest cluster from all the principal components
        for pc in principalComponents.columns:
            objectsPcArray = objects.loc[:, pc].values
            clustersPcArray = clusters.loc[:, pc].values
            for o in range(len(objectsPcArray)):
                distanceArray = distances.loc[:, distanceColumns[o]].values
                for c in range(len(clustersPcArray)):
                    distanceArray[c] = distanceArray[c] + numpy.power(objectsPcArray[o] - clustersPcArray[c], 2)

        for c in distanceColumns:
            distanceArray = distances.loc[:, distanceColumns[c]].values
            for j in range(len(distanceArray)):
                distanceArray[j] = numpy.sqrt(distanceArray[j])

        # Find the closest cluster of all calculated clusters & count number of objects belonging to each cluster
        clusterObjectCounts = clusters.loc[:, 'Number of objects'].values
        for c in range(len(clusterObjectCounts)):
            clusterObjectCounts[c] = 0

        for c in range(len(distanceColumns)):
            distanceSeries = pandas.Series(distances.loc[:, distanceColumns[c]].values)
            minIndex = distanceSeries.idxmin()
            clusterObjectCounts[minIndex] = clusterObjectCounts[minIndex] + 1
            objectClusterIndices[c] = minIndex

        # Recalculate cluster mean
        for pc in principalComponents.columns:
            objectsPcArray = objects.loc[:, pc].values
            clustersPcArray = clusters.loc[:, pc].values

            for c in range(len(clustersPcArray)):
                clustersPcArray[c] = 0
            for o in range(len(objectsPcArray)):
                clustersPcArray[objectClusterIndices[o]] = clustersPcArray[objectClusterIndices[o]] + objectsPcArray[o]

            for c in range(len(clustersPcArray)):
                if clusterObjectCounts[c] == 0:
                    clustersPcArray[c] = clustersPcArray[c] / 1
                else:
                    clustersPcArray[c] = clustersPcArray[c] / clusterObjectCounts[c]

    return clusters


def principal_component_analysis(dim, data):

    # Feature values of the data
    x = data.loc[:, data.drop(labels=['Class'], axis=1).columns].values
    # Class values of the data
    y = data.loc[:, 'Class'].values

    # Standardize the data
    x = StandardScaler().fit_transform(x)

    # Dimensional reduction
    pca = PCA(n_components=dim)
    principalComponents = pca.fit_transform(x)

    print('Variance Ratio', dim, 'dimensional PCA:', pca.explained_variance_ratio_)

    principalData = pandas.DataFrame(data=principalComponents)
    principalData = pandas.concat([principalData, data[['Class']]], axis=1)
    return principalData

def visualize_2_pc(data):
    # Visualize Projection
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = [1, 2, 3]
    colors = ['r', 'g', 'b']

    for target, color in zip(targets, colors):
        indicesToKeep = data['Class'] == int(target)
        ax.scatter(data.loc[indicesToKeep, 0],
                   data.loc[indicesToKeep, 1],
                   c=color,
                   s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

# Import data
labels = ['Class', 'Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5', 'Feature 6', 'Feature 7',
          'Feature 8', 'Feature 9', 'Feature 10', 'Feature 11', 'Feature 12', 'Feature 13']
data = pandas.read_csv('WINE.txt', sep=' ', names=labels)

# Principal component analysis
pca2Data = principal_component_analysis(2, data)
pca5Data = principal_component_analysis(5, data)
pca8Data = principal_component_analysis(8, data)

pandas.set_option("display.max_rows", 1000, "display.max_columns", 1000)

# K-means clustering
k_means_clustering_2pca = k_means_clustering(3, pca2Data)
k_means_clustering_5pca = k_means_clustering(3, pca5Data)
k_means_clustering_8pca = k_means_clustering(3, pca8Data)

k_means_clustering_2pca.to_csv('2pca.csv', index=False)
k_means_clustering_5pca.to_csv('5pca.csv', index=False)
k_means_clustering_8pca.to_csv('8pca.csv', index=False)

# Visualize 2 PCA
visualize_2_pc(pca2Data)