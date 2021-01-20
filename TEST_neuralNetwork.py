import os
import time
import numpy as np
from numpy import transpose
from random import seed
from random import random
from math import exp
from scipy.ndimage import gaussian_filter
from scipy import signal
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import reconstruction
from sewar.full_ref import uqi, mse, rmse, uqi, scc, sam, vifp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from statistics import mean, variance

#----------------- UTILITY METHODS ---------------------#
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

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

# Transfer neuron activation: sigmoid
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

# Calculate the derivative of an neuron output: sigmoid derivate
def transfer_derivative(output):
	return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
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
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

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
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

#----------------- IMAGE METHODS ---------------------#
def read_image(image_path):
    image = imread(image_path)
    return image

def gaussian_filtering(image):
    #image = img_as_float(image)
    image = gaussian_filter(image,1)

    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image

    dilated = reconstruction(seed, mask, method='dilation')
    return dilated

def filtered_image(image, filter_name):
	image1 = image

	if filter_name == "gaussian":
		image2 = gaussian_filtering(image)
    
	image3 =  image1 - image2
	#io.imsave("out.png", (color.convert_colorspace(image3, 'HSV', 'RGB')*255).astype(np.uint8))
	return image3

def cross_correlation_norm(img1, img2):
	a = (img1 - np.mean(img1)) / (np.std(img1) * len(img1))
	a_flatten = np.array(a).flatten()
	b = (img2 - np.mean(img2)) / (np.std(img2))
	b_flatten = np.array(b).flatten()
	#c = np.correlate(a_flatten, b_flatten)
	c = signal.correlate(a_flatten, b_flatten, mode='full')[0]
	return c

#TODO: complete this function took from paper
def mean_cosine_similarity(image, f_image):
	simil = cosine_similarity(image.reshape(len(image),-1),f_image.reshape(len(f_image),-1))
	simil_flatten = simil.flatten()
	sum = np.sum(simil_flatten)
	return sum/len(simil_flatten)

def print_feature_matrix(feature_img_matrix):
	print("Feature matrix: [")
	for row in feature_img_matrix:
		print(row)
	print("]")

def feature_extraction_from_dir(dir):
	feature_img_matrix = []

	print("image extraction...")
	#image extraction from file system
	i = 0
	l = 5
	for drct in os.listdir(dir):
		if os.path.isfile(drct) == False:
			dirpath = dir + '/' + drct
			for file in os.listdir(dirpath):
				i=i+1
				print(f'Number of photos analyzed: \r{i}')
				feature_img = []
				if file.lower().endswith('.png'):
					image_path = dir + '/' + drct + '/' + file

					image = img_as_float(read_image(image_path))
					f_image = filtered_image(image, "gaussian")
					
					#BASE FEATURES
					feature_img.append((mean(np.array(image).flatten())+mean(np.array(f_image).flatten()))/2)
					a = variance(np.array(image).flatten())
					b = variance(np.array(f_image).flatten())
					c = [a, b]
					feature_img.append(variance(c))

					#FEATURE EXTRACTION
					feature_img.append(mse(image,f_image))
					feature_img.append(rmse(image,f_image))
					feature_img.append(uqi(image,f_image))
					feature_img.append(scc(image,f_image))
					feature_img.append(sam(image,f_image))
					feature_img.append(vifp(image,f_image))
					feature_img.append(cross_correlation_norm(image, f_image))
					feature_img.append(mean_cosine_similarity(image, f_image))
					
					#LABEL DEFINITION
					if drct == "base":
						feature_img.append(0)
					elif drct == "doct":
						feature_img.append(1)

					feature_img_matrix.append(feature_img)
				#printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
		i += 1

	return feature_img_matrix

dataset = [[1,2,0],
	[1,3,0],
	[2,3,0],
	[7,9,1],
	[8,6,1],
	[4,1,0],
	[2,3,0],
	[6,7,1],
	[7,9,1],
	[5,3,0],
    [5,1,0],
    [8,6,1]]

testset = [[3,1,0],
	[7,6,None],
    [2,4,None],
	[1,1,None],
	[7,7,None]]

n_inputs = len(dataset[0]) - 1
n_hidden = int(len(dataset) / 2) - 1
n_outputs = len(set([row[-1] for row in dataset]))
l_rate = 0.5
epochs = 50

seed(1)
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, l_rate, epochs, n_outputs)
#print_network(network)

for row in testset:
	prediction = predict(network, row)
	if row[-1] is None:
		print('Blind evaluation, Got=%d' % (prediction))
	else:	
		print('Expected=%d, Got=%d' % (row[-1], prediction))


dir = "./src/resources/imgs"

feature_img_matrix = feature_extraction_from_dir(dir)
#print_feature_matrix(feature_img_matrix)

testdir = "./src/resources/100.png"
testdir_t = "./src/resources/100t.png"
testdir_t2 = "./src/resources/100t2.png"
dataset = feature_img_matrix

feature_img = []
feature_img_matrix_test = []
#BASE IMG
image = img_as_float(read_image(testdir))
f_image = filtered_image(image, "gaussian")
#BASE FEATURES
feature_img.append((mean(np.array(image).flatten())+mean(np.array(f_image).flatten()))/2)
a = variance(np.array(image).flatten())
b = variance(np.array(f_image).flatten())
c = [a, b]
feature_img.append(variance(c))
#FEATURE EXTRACTION
feature_img.append(mse(image,f_image))
feature_img.append(rmse(image,f_image))
feature_img.append(uqi(image,f_image))
feature_img.append(scc(image,f_image))
feature_img.append(sam(image,f_image))
feature_img.append(vifp(image,f_image))
feature_img.append(cross_correlation_norm(image, f_image))
feature_img.append(mean_cosine_similarity(image, f_image))
feature_img.append(0)
feature_img_matrix_test.append(feature_img)
#DOCT IMG
feature_img = []
image = img_as_float(read_image(testdir_t))
f_image = filtered_image(image, "gaussian")
#BASE FEATURES
feature_img.append((mean(np.array(image).flatten())+mean(np.array(f_image).flatten()))/2)
a = variance(np.array(image).flatten())
b = variance(np.array(f_image).flatten())
c = [a, b]
feature_img.append(variance(c))
#FEATURE EXTRACTION
feature_img.append(mse(image,f_image))
feature_img.append(rmse(image,f_image))
feature_img.append(uqi(image,f_image))
feature_img.append(scc(image,f_image))
feature_img.append(sam(image,f_image))
feature_img.append(vifp(image,f_image))
feature_img.append(cross_correlation_norm(image, f_image))
feature_img.append(mean_cosine_similarity(image, f_image))
feature_img.append(1)
feature_img_matrix_test.append(feature_img)
#DOCT IMG2
image = img_as_float(read_image(testdir_t2))
f_image = filtered_image(image, "gaussian")
#BASE FEATURES
feature_img.append((mean(np.array(image).flatten())+mean(np.array(f_image).flatten()))/2)
a = variance(np.array(image).flatten())
b = variance(np.array(f_image).flatten())
c = [a, b]
feature_img.append(variance(c))
#FEATURE EXTRACTION
feature_img.append(mse(image,f_image))
feature_img.append(rmse(image,f_image))
feature_img.append(uqi(image,f_image))
feature_img.append(scc(image,f_image))
feature_img.append(sam(image,f_image))
feature_img.append(vifp(image,f_image))
feature_img.append(cross_correlation_norm(image, f_image))
feature_img.append(mean_cosine_similarity(image, f_image))
feature_img.append(1)
feature_img_matrix_test.append(feature_img)
testset = feature_img_matrix_test

print_feature_matrix(dataset)
print_feature_matrix(testset)

#------------------ NETWORK FOR IMGS ---------------#
n_inputs = len(dataset[0]) - 1
n_hidden = int(len(dataset) / 2) - 1
n_outputs = len(set([row[-1] for row in dataset]))
l_rate = 0.1
epochs = 50

seed(1)
network = initialize_network(n_inputs, n_hidden, n_outputs)
train_network(network, dataset, l_rate, epochs, n_outputs)
#print_network(network)

for row in testset:
	prediction = predict(network, row)
	if row[-1] is None:
		print('Blind evaluation, Got=%d' % (prediction))
	else:	
		print('Expected=%d, Got=%d' % (row[-1], prediction))