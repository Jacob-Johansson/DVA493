import numpy
import pandas
from astropy.io import ascii
import matplotlib.pyplot as plt
import random


# Load berlin where each row represents a location id, x and y coordinates
def load_data():
    data = open('berlin52.ascii', 'r')
    table = []
    for line in data:
        line = line.strip()
        columns = line.split()
        columns[0] = int(columns[0])
        columns[1] = float(columns[1])
        columns[2] = float(columns[2])
        table.append(columns)
    return table


def calculate_fitness(path, data):
    sum = 0
    size = len(path) - 2
    for i in range(size):
        l1 = path[i]
        l2 = path[i + 1]
        x1 = data[l1][1]
        y1 = data[l1][2]
        x2 = data[l2][1]
        y2 = data[l2][2]
        sum += numpy.sqrt(numpy.power(x2 - x1, 2) + numpy.power(y2 - y1, 2))
    return sum


def initialize_population(numPopulation, data):
    numLocations = len(data) + 1

    paths = []
    fitness = []

    # Initialize random paths
    for i in range(numPopulation):
        path = random.sample(range(0, numLocations - 1), numLocations - 1)
        oneIndex = list.index(path, 0)
        tempStart = path[0]
        path[0] = path[oneIndex]
        path[oneIndex] = tempStart
        path.append(0)
        paths.append(path)

    # Initialize fitness
    for i in range(numPopulation):
        fitness.append(calculate_fitness(paths[i], data))

    return [paths, fitness]


def selection(fitness, tournamentSize, desiredNumParents):
    parents = []

    for i in range(desiredNumParents):
        selections = random.sample(range(0, len(fitness)), tournamentSize)

        best = 0
        for j in range(tournamentSize):
            fitness1 = fitness[selections[j]]
            fitness2 = fitness[selections[best]]

            if fitness1 < fitness2:
                best = j

        parents.append(selections[best])

    return parents

def crossover(parents, paths):
    random.shuffle(parents)
    newPaths = []

    for i in range(0, len(parents) - 1, 2):
        parentA = parents[i]
        parentB = parents[i + 1]

        pathA = paths[parentA]
        pathB = paths[parentB]

        interval = random.sample(range(1, len(paths[parentA])-1), 2)
        interval.sort()
        i0 = interval[0]
        i1 = interval[1] + 1
        subParentA = pathA[i0:i1]
        subParentB = [i0 for i0 in pathB if i0 not in subParentA]

        newPath = []

        for j in range(0, i0):
            newPath.append(subParentB.pop())

        for j in range(0, len(subParentA)):
            newPath.append(subParentA[j])

        for j in range(0, len(subParentB)):
            newPath.append(subParentB.pop())

        newPaths.append(newPath)

    return newPaths


def mutate(paths, probability):
    for i in range(len(paths)):

        p = random.random()
        if p > probability:
            continue

        path = paths[i]

        indices = random.sample(range(1, len(path) - 1), 2)
        indices.sort()

        subPath = path[indices[0]:(indices[1] + 1)]
        subPath.reverse()

        path[indices[0]: indices[1] + 1] = subPath

        paths[i] = path

    return paths


def replacement(paths, newPaths, fitness, data):

    worstIndices = []
    for i in range(len(newPaths)):
        worstIndex = 0
        for j in range(len(paths)):
            if fitness[j] > fitness[worstIndex]:
                if j not in worstIndices:
                    worstIndex = j
        if worstIndex not in worstIndices:
            worstIndices.append(worstIndex)

    for i in range(len(worstIndices)):
        paths[worstIndices[i]] = newPaths[i]
        fitness[worstIndices[i]] = calculate_fitness(paths[worstIndices[i]], data)


def find_best_fitness(fitness):
    bestIndex = 0
    for i in range(len(fitness)):
        if fitness[i] < fitness[bestIndex]:
            bestIndex = i
    return [fitness[bestIndex], bestIndex]


def plot(paths, bestIndex, data, fitnessPerGeneration, generationIndices):
    plt.subplot(1, 2, 1)
    plt.plot(fitnessPerGeneration, 'r-')
    plt.title("Fitness Performance")
    plt.xlabel("Number of generations")
    plt.ylabel("Fitness")

    path = paths[bestIndex]
    bestFitness = calculate_fitness(path, data)
    x = []
    y = []
    for i in range(0, len(path)):
        x.append(data[path[i]][1])
        y.append(data[path[i]][2])
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'o-')
    pathTitle = "Path where Fitness=" + str(round(bestFitness, 2))
    plt.title(pathTitle)
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.pause(0.01)
    plt.draw()
    plt.show()


# Main
numPopulation = 500
numParents = 60
tournamentSize = 8
mutationProbability = 0.4
targetFitness = 8000

data = load_data()
[paths, fitness] = initialize_population(numPopulation, data)

newBest = numpy.Inf
bestIndex = 0

fitnessPerGeneration = []
generationIndices = []

generationIndex = 1
while newBest > targetFitness:
    parents = selection(fitness, tournamentSize, numParents)
    newPaths = crossover(parents, paths)
    newPaths = mutate(newPaths, mutationProbability)
    replacement(paths, newPaths, fitness, data)

    [currentBest, currentBestIndex] = find_best_fitness(fitness)
    if currentBest < newBest:
        newBest = currentBest
        bestIndex = currentBestIndex
        print("New best fitness:", newBest, "Generation:", generationIndex)

    fitnessPerGeneration.append(currentBest)
    generationIndices.append(generationIndex)
    generationIndex = generationIndex + 1

plot(paths, bestIndex, data, fitnessPerGeneration, generationIndices)

