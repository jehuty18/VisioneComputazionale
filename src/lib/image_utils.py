import os
import sys
import numpy as np
import src.lib.binary_similarity_measures as bsm
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

import src.lib.loading_bar as lbar

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
	return image3

def cross_correlation_norm(img1, img2):
	a = (img1 - np.mean(img1)) / (np.std(img1) * len(img1))
	a_flatten = np.array(a).flatten()
	b = (img2 - np.mean(img2)) / (np.std(img2))
	b_flatten = np.array(b).flatten()
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

const_file_not_exist = "File does not exist. Creatining new one and proceeding with computing operations.."

def feature_extraction_from_dir(dir):
	txt_files_dir = dir + '/' + 'txt'
	feature_img_matrix = []

	print("image extraction...")
	#image extraction from file system
	#l = 5
	print('Number of photos analyzed in: ')
	for drct in os.listdir(dir):
		dirpath = dir + '/' + drct
		if os.path.isfile(dirpath) == False:
			index_current_file = 0
			#dirpath = dir + '/' + drct
			if drct != 'txt':
				print('- ', dirpath)
				n_files = len(os.listdir(dirpath))
				for file in os.listdir(dirpath):
					index_current_file=index_current_file+1
					lbar.counting_bar(index_current_file, n_files)
					feature_img = []
					if file.lower().endswith('.png'):
						file_name = file.split('.')[0]
						txt_files_name_path = txt_files_dir + '/' + drct + '/' + file_name + '.txt'
						try:
							with open(txt_files_name_path) as f:
								#print("File" + file + "exists!")
								file_content = f.readlines()
								for i in range(len(file_content)):
									file_content[i] = float(file_content[i].rstrip("\n"))
								#LABEL DEFINITION
								if drct == "base":
									file_content.append(int(0))
								elif drct == "doct":
									file_content.append(int(1))
								feature_img_matrix.append(file_content)
								#print(file_content)
						except IOError:
							#print(const_file_not_exist)

							image_path = dir + '/' + drct + '/' + file

							#image = img_as_float(read_image(image_path))
							#f_image = filtered_image(image, "gaussian")

							testdir_img = np.array(read_image(image_path))
							rows = len(testdir_img)
							columns = len(testdir_img[0])
							flat = testdir_img.flatten()

							#--------- FUNCTION TO CONVERT INT INTO BYTE ----------#
							byte_array = bsm.extract_bitplane_from_img_array(flat)

							agreements_array = bsm.extract_agreements_array(byte_array, rows, columns)

							sneeth_and_sokai = bsm.sneeth_and_sokai_sm1(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_2 = bsm.sneeth_and_sokai_sm2(agreements_array[0], agreements_array[1], agreements_array[2])
							sneeth_and_sokai_3 = bsm.sneeth_and_sokai_sm3(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_4 = bsm.sneeth_and_sokai_sm4(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_5 = bsm.sneeth_and_sokai_sm5(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							kulczynski_similarity = bsm.kulczynski_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
							ochiai_similarity = bsm.ochiai_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
							lance_and_williams_dissimilarity = bsm.lance_and_williams_dissm(agreements_array[0], agreements_array[1], agreements_array[2])
							pattern_diff = bsm.pattern_difference(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							variance_diss = bsm.variance_dissimilarity_measure(byte_array)

							binary_min_histogram_diff = bsm.binary_min_histogram_difference(byte_array)
							binary_absolute_histogram_diff = bsm.binary_absolute_histogram_difference(byte_array)
							binary_mutual_entr = bsm.binary_mutual_entropy(byte_array)

							#FEATURE EXTRACTION
							"""feature_img.append(mse(image,f_image))
							feature_img.append(rmse(image,f_image))
							feature_img.append(uqi(image,f_image))
							feature_img.append(scc(image,f_image))
							feature_img.append(sam(image,f_image))
							feature_img.append(vifp(image,f_image))
							feature_img.append(cross_correlation_norm(image, f_image))
							feature_img.append(mean_cosine_similarity(image, f_image))"""

							bsm_feature_array = np.array([sneeth_and_sokai, sneeth_and_sokai_2, sneeth_and_sokai_3, sneeth_and_sokai_4, sneeth_and_sokai_5, kulczynski_similarity, ochiai_similarity, lance_and_williams_dissimilarity, pattern_diff, variance_diss, binary_min_histogram_diff, binary_absolute_histogram_diff, binary_mutual_entr])
							
							with open(txt_files_name_path, "a+") as feature_file:
								np.savetxt(feature_file,bsm_feature_array)
							
							bsm_feature_array_list = [sneeth_and_sokai, sneeth_and_sokai_2, sneeth_and_sokai_3, sneeth_and_sokai_4, sneeth_and_sokai_5, kulczynski_similarity, ochiai_similarity, lance_and_williams_dissimilarity, pattern_diff, variance_diss, binary_min_histogram_diff, binary_absolute_histogram_diff, binary_mutual_entr]

							#LABEL DEFINITION
							if drct == "base":
								bsm_feature_array_list.append(int(0))
							elif drct == "doct":
								bsm_feature_array_list.append(int(1))

							feature_img_matrix.append(bsm_feature_array_list)
					#printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
				print()

	return feature_img_matrix

def feature_extraction_from_dir_unsupervised(dir):
	txt_files_dir = dir + '/' + 'txt'
	feature_img_matrix = []

	print("image extraction...")
	#image extraction from file system
	#l = 5
	print('Number of photos analyzed in: ')
	for drct in os.listdir(dir):
		dirpath = dir + '/' + drct
		if os.path.isfile(dirpath) == False:
			index_current_file = 0
			#dirpath = dir + '/' + drct
			if drct != 'txt':
				print('- ', dirpath)
				n_files = len(os.listdir(dirpath))
				for file in os.listdir(dirpath):
					index_current_file=index_current_file+1
					lbar.counting_bar(index_current_file, n_files)
					feature_img = []
					if file.lower().endswith('.png'):
						file_name = file.split('.')[0]
						txt_files_name_path = txt_files_dir + '/' + drct + '/' + file_name + '.txt'
						try:
							with open(txt_files_name_path) as f:
								#print("File" + file + "exists!")
								file_content = f.readlines()
								for i in range(len(file_content)):
									file_content[i] = float(file_content[i].rstrip("\n"))
								#LABEL DEFINITION
								file_content.append(None)
								feature_img_matrix.append(file_content)
								#print(file_content)
						except IOError:
							#print(const_file_not_exist)

							image_path = dir + '/' + drct + '/' + file

							#image = img_as_float(read_image(image_path))
							#f_image = filtered_image(image, "gaussian")

							testdir_img = np.array(read_image(image_path))
							rows = len(testdir_img)
							columns = len(testdir_img[0])
							flat = testdir_img.flatten()

							#--------- FUNCTION TO CONVERT INT INTO BYTE ----------#
							byte_array = bsm.extract_bitplane_from_img_array(flat)

							agreements_array = bsm.extract_agreements_array(byte_array, rows, columns)

							sneeth_and_sokai = bsm.sneeth_and_sokai_sm1(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_2 = bsm.sneeth_and_sokai_sm2(agreements_array[0], agreements_array[1], agreements_array[2])
							sneeth_and_sokai_3 = bsm.sneeth_and_sokai_sm3(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_4 = bsm.sneeth_and_sokai_sm4(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							sneeth_and_sokai_5 = bsm.sneeth_and_sokai_sm5(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							kulczynski_similarity = bsm.kulczynski_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
							ochiai_similarity = bsm.ochiai_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
							lance_and_williams_dissimilarity = bsm.lance_and_williams_dissm(agreements_array[0], agreements_array[1], agreements_array[2])
							pattern_diff = bsm.pattern_difference(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
							variance_diss = bsm.variance_dissimilarity_measure(byte_array)

							binary_min_histogram_diff = bsm.binary_min_histogram_difference(byte_array)
							binary_absolute_histogram_diff = bsm.binary_absolute_histogram_difference(byte_array)
							binary_mutual_entr = bsm.binary_mutual_entropy(byte_array)

							bsm_feature_array = np.array([sneeth_and_sokai, sneeth_and_sokai_2, sneeth_and_sokai_3, sneeth_and_sokai_4, sneeth_and_sokai_5, kulczynski_similarity, ochiai_similarity, lance_and_williams_dissimilarity, pattern_diff, variance_diss, binary_min_histogram_diff, binary_absolute_histogram_diff, binary_mutual_entr])
							
							with open(txt_files_name_path, "a+") as feature_file:
								np.savetxt(feature_file,bsm_feature_array)
							
							bsm_feature_array_list = [sneeth_and_sokai, sneeth_and_sokai_2, sneeth_and_sokai_3, sneeth_and_sokai_4, sneeth_and_sokai_5, kulczynski_similarity, ochiai_similarity, lance_and_williams_dissimilarity, pattern_diff, variance_diss, binary_min_histogram_diff, binary_absolute_histogram_diff, binary_mutual_entr]

							#LABEL DEFINITION
							bsm_feature_array_list.append(None)

							feature_img_matrix.append(bsm_feature_array_list)
					#printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
				print()

	return feature_img_matrix