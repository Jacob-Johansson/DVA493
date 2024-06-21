import random
import numpy
import pandas
import math
import os
import sys
import matplotlib.pyplot
from sklearn import datasets
import matplotlib.pyplot as plt

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

# Function that sums all the nets in a layer at once.
# The number of rows in weights determine the number of nets of the layer,
# Inputs should be a nx1 matrix where n is the number of previous nets connected to respective net in the layer
# Biases should be a mx1 matrix where m is the number of nets in the layer
def sum_net_v2(weights, inputs, biases):
    output = create_matrix(len(weights), 1)

    for weightRow in range(len(weights)):
        output[weightRow][0] = biases[weightRow][0]
        for inputRow in range(len(inputs)):
            output[weightRow][0] += weights[weightRow][inputRow] * inputs[inputRow][0]

    return output
###############################################################################
# Calculates the error term for a net in the output layer
def calculate_error_term_output_layer(target, predicted):
    return (target - predicted) * predicted * (1 - predicted)

# Calculates the error term for a net in a hidden layer where
# "weights" is an array of the outgoing weights to the next layer of nets
# "errorTerms" is an array of the calculated error term of respective net in the next layer
# output is the calculated output of the net in the hidden layer
def calculate_error_term_hidden_layer(weights, errorTerms, output):
    sum = 0
    for i in range(len(weights)):
        sum += weights[i] * errorTerms[i]

    return (1 - output) * output * sum

def calculate_net(weights, biases, inputs):
    output = sum_net_v2(weights, inputs, biases)

    for i in range(len(output)):
        output[i][0] = sigmoid(output[i][0])

    return output

def calculate_error_mean_square(targets, predicted):
    sum = 0
    for i in range(len(targets)):
        sum += numpy.power(targets[i], predicted[i])

    return sum / len(targets)
# Function for dynamically creating a matrix
def create_matrix(n, m):
    val = [0] * n
    for x in range(n):
        val[x] = [0] * m
    return val

def initialize_weights(weights):
    for row in range(len(weights)):
        for column in range(len(weights)):
            weights[row][column] = random.uniform(0, 0.2)
####################################################
# main loop
training_input, training_target = parse_file()

# normalizes the training inputs
for sample in range(len(training_input)):
    length = len(training_input[sample])
    for attribute in range(length):
        training_input[sample][attribute] /= length

training_set = training_input[0:864]
training_target_set = training_target[0:864]
validation_set = training_input[864:979]
validation_target_set = training_target[864:979]
training_test_set = training_input[979:1151]
training_test_target_set = training_target[979:1151]

#######################################

numNodesInHiddenLayer = 2
weightsHidden = create_matrix(numNodesInHiddenLayer, len(training_set[0]))
weightsOutput = create_matrix(1, numNodesInHiddenLayer)
biasesHidden = create_matrix(numNodesInHiddenLayer, 1)
biasesOutput = create_matrix(1, 1)
learningRate = 0.1

biasesOutput[0][0] = 0.5

# Initialize the weights
initialize_weights(weightsHidden)
initialize_weights(weightsOutput)

trainingErrorMeanSquares = []
validationErrorMeanSquares = []
numEpochs = 500

# Training
for epoch in range(numEpochs):
    trainingErrorMeanSquared = 0
    for sample in range(len(training_set)):
        # Create input matrix
        inputs = create_matrix(len(training_set[0]), 1)
        for i in range(len(training_set[sample])):
            inputs[i][0] = training_set[sample][i]

        # Prediction
        netsHidden = calculate_net(weightsHidden, biasesHidden, inputs)
        netsOutput = calculate_net(weightsOutput, biasesOutput, netsHidden)
        # Backwards propagation
        errorTermOutput = calculate_error_term_output_layer(training_target_set[sample], netsOutput[0][0])

        for n in range(len(netsHidden)):
            errorTermHidden = calculate_error_term_hidden_layer([weightsOutput[0][n]], [errorTermOutput], netsHidden[n][0])
            for c in range(len(weightsHidden[n])):
                weightsHidden[n][c] += learningRate * errorTermHidden * inputs[c][0]

        # Update weights of the output after hidden layer due to the nets needing the previous weights from hidden to output nets
        for row in range(len(weightsOutput)):
            for column in range(len(weightsOutput[row])):
                weightsOutput[row][column] += learningRate * errorTermOutput * netsHidden[row][0]

        # Calculate error mean square of the sample
        trainingErrorMeanSquared += numpy.power(training_target_set[sample] - netsOutput[0][0], 2)

    trainingErrorMeanSquares.append(trainingErrorMeanSquared / len(training_set))

    # Validation
    errorMeanSquare = 0
    for validationSample in range(len(validation_set)):
        validation = validation_set[validationSample]

        inputs = create_matrix(len(validation_set[0]), 1)
        for i in range(len(validation_set[validationSample])):
            inputs[i][0] = validation_set[validationSample][i]

        netsHidden = calculate_net(weightsHidden, biasesHidden, inputs)
        netsOutput = calculate_net(weightsOutput, biasesOutput, netsHidden)
        errorMeanSquare += numpy.power(validation_target_set[validationSample] - netsOutput[0][0], 2)
    errorMeanSquare /= len(validation_set)
    validationErrorMeanSquares.append(errorMeanSquare)
accuracy = []

# Testing
for sample in range(len(training_test_set)):
    # Create input matrix
    inputs = create_matrix(len(training_test_set[0]), 1)
    for i in range(len(training_test_set[sample])):
        inputs[i][0] = training_test_set[sample][i]

    # Prediction
    netsHidden = calculate_net(weightsHidden, biasesHidden, inputs)
    netsOutput = calculate_net(weightsOutput, biasesOutput, netsHidden)

    test = numpy.round(netsOutput[0][0])
    if test == training_test_target_set[sample]:
        accuracy.append(1)
    else:
        accuracy.append(0)

numAccurate = 0
for i in range(len(accuracy)):
    if accuracy[i] == 1:
        numAccurate += 1

print('Predictability:', numAccurate / len(accuracy))

plt.figure()
plt.subplot(211)
plt.title("Validation Error")
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.plot(validationErrorMeanSquares, 'b--')

plt.subplot(212)
plt.title("Training Error")
plt.xlabel("Epoch")
plt.ylabel("Probability")
plt.plot(trainingErrorMeanSquares, 'r--')
plt.show()


