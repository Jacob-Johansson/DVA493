import numpy
import pandas
import math
import os
import sys
import matplotlib.pyplot
from sklearn import datasets

feature_set = []

# import data and parse it to training samples
def parse_file():
    feature_set, labels = datasets.make_moons(100, noise=0.10)

    labels = labels.reshape(100, 1)

    with open(os.path.join(sys.path[0], "Diabetic.txt"), "r") as f:
        patients = f.read().splitlines()

    # remove the first unusable lines
    for i in range(0, 24):
        patients.pop(0)

    patient_attributes = []

    # Every patient has 19 attributes, split them by ","
    for i, patient in enumerate(patients):
        patient_attributes.append(patient.split(','))

    # Take the first 18 attributes in training input
    training_input = [attribute[0:19] for attribute in patient_attributes]
    training_input = numpy.array(training_input, dtype=float)

    # Take the last attribute as training output
    training_target = [[attribute[-1] for attribute in patient_attributes]][0]
    training_target = numpy.array(training_target, dtype=float)

    return training_input, training_target
###########################################################################

# Sigmoid
def sigmoid(x):
    return 1/(1 + numpy.exp(-x))

def sigmoid_derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

# The function takes a weights, inputs and biases connected to the net to sum up
# the linear combination which is used for the output function of the net
def sum_net(weights, inputs, biases):
    if len(weights) != len(inputs) != len(biases):
        return 0
    fWeights = numpy.array(weights, dtype=float)
    fInputs = numpy.array(inputs, dtype=float)

    output = numpy.dot(fWeights, fInputs)
    for i in range(len(biases)):
        output += biases[i]

    return output

# Error function of the network that takes an array of the target output and an array of the predicted output,
# and return an error of the prediction of all training samples
def error(target, predicted):
    error = 0
    for i in range(len(target)):
        error += math.pow(target[i] - predicted[i], 2)

    error *= 0.5
    return error
# Calculated the error term for a net, in our case just at the output layer.
def calculate_error_term(target:float, predicted:float):
    return (target - predicted) * predicted * (1 - predicted)

# Backpropogation algorithm per output node
def backpropogation(inputs, weights, target, predicted, learningRate:float):

    errorTerm = calculate_error_term(target, predicted)

    for i in range(len(weights)):
        weights[i] += learningRate * errorTerm * inputs[i]

    return weights
# Predicts an output from the inputs, weights and biases
def predict(inputs, weights, biases):

    if len(inputs) != len(weights) != len(biases):
        return 0

    sum = sum_net(weights, inputs, biases)
    return sigmoid(sum)
####################################################

# main loop
training_input, training_target = parse_file()

zeroBiases = []
zeroWeights = []
oneBiases = []
oneWeights = []

training_set = training_input[0:864]
training_target_set = training_target[0:864]
training_test_set = training_input[979:1151]

# initializes weights & biases
for i in range(len(training_input[0])):
    zeroBiases.append(0.1)
    zeroWeights.append(0.1)
    oneBiases.append(0)
    oneWeights.append(0.1)

# normalizes the training inputs
for sample in range(len(training_set)):
    length = len(training_set[sample])
    for attribute in range(length):
        training_set[sample][attribute] /= length

#print(training_set[sample])

# Training
zeroPredictedArray = []
onePredictedArray = []
sampleNumberArray = []

for epoch in range(100):
    for sample in range(len(training_set)):
        zeroPredicted = predict(training_set[sample], zeroWeights, zeroBiases)
        onePredicted = predict(training_set[sample], oneWeights, oneBiases)

        zeroPredictedArray.append(zeroPredicted)
        onePredictedArray.append(onePredicted)
        sampleNumberArray.append(sample)

        zeroWeights = backpropogation(training_set[sample], zeroWeights, training_target_set[sample], zeroPredicted, 0.2)
        oneWeights = backpropogation(training_set[sample], oneWeights, training_target_set[sample], onePredicted, 0.2)

testPredicted = []
testTarget = []

# Test
for sample in range(len(training_test_set)):
    zeroPredicted = predict(training_test_set[sample], zeroWeights, zeroBiases)
    onePredicted = predict(training_test_set[sample], oneWeights, oneBiases)

    if onePredicted > 0.5:
        testPredicted.append(1)
    else:
        testPredicted.append((0))

    testTarget.append(training_target_set[sample])

numCorrect = 0
for i in range(len(testPredicted)):
    if testPredicted[i] == testTarget[i]:
        numCorrect += 1

print(numCorrect, len(testPredicted))