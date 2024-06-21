import numpy
import pandas
import numpy as np

def calculate_entropy(data):
    sortedData = data.tolist()
    sortedData.sort()

    # Calculate the proportion of each decision
    decisionProportionMap = {}
    for i in range(len(sortedData)):
        decisionProportionMap[sortedData[i]] = 0

    for i in range(len(sortedData)):
        decisionProportionMap[sortedData[i]] = decisionProportionMap[sortedData[i]] + 1

    decisionProportions = list(decisionProportionMap.values())
    for i in range(len(decisionProportions)):
        decisionProportions[i] = decisionProportions[i] / len(sortedData)

    # sum to make up the entropy
    entropy = 0
    for i in range(len(decisionProportions)):
        entropy += -decisionProportions[i]*numpy.log2(decisionProportions[i])

    return entropy



def calculate_information_gain(parentEntropy, parentNumTrainingExamples, childrenNumTrainingExamplesArray, childrenEntropyArray):
    sum = 0
    for i in range(len(childrenEntropyArray)):
        sum += (childrenNumTrainingExamplesArray[i]/parentNumTrainingExamples) * childrenEntropyArray[i]

    return parentEntropy - sum

labels = ['Weekend', 'Weather', 'Parents', 'Money', 'Decision']

data = pandas.read_csv('data.txt', sep=' ', names=labels)

# attributes
attributes = data.drop(labels=['Weekend', 'Decision'], axis=1)

# decisions (classes)
decisions = data.loc[:, 'Decision'].values

#
calculate_entropy(decisions)