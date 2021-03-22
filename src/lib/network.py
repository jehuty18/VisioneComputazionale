
from math import exp
from random import random
import src.lib.enum_utils as eut

#----------------- NETWORK METHODS ---------------------#
# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation
 
def print_network(network):
    for layer in network:
	    print(layer)

################TRAINING FUNCTIONS######################
def transfer(activation,function_name):
	if function_name == eut.TraningFunctions.SIGMOID:
		return transfer_sigmoid(activation)

def transfer_derivative(output,function_name):
	if function_name == eut.TraningFunctions.SIGMOID:
		return transfer_derivative_sigmoid(output)

# Transfer neuron activation: sigmoid
def transfer_sigmoid(activation):
	return 1.0 / (1.0 + exp(-activation))

# Calculate the derivative of an neuron output: sigmoid derivate
def transfer_derivative_sigmoid(output):
	return output * (1.0 - output)
########################################################

# Forward propagate input to a network output
def forward_propagate(network, row, training_function):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation, training_function)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected, training_function):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'], training_function)

# Update network weights with error: online learning
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs, training_function):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row, training_function)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected, training_function)
			update_weights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row, training_function):
	outputs = forward_propagate(network, row, training_function)
	return outputs.index(max(outputs))

def evaluate_net(dataset, network):
	for row in dataset:
		prediction = predict(network, row, eut.TraningFunctions.SIGMOID)
		if row[-1] is None:
			print('Blind evaluation, Got=%d' % (prediction))
		else:	
			print('Expected=%d, Got=%d' % (row[-1], prediction))