import os
import time
import numpy as np
from numpy import transpose
from random import seed
from random import random
from math import exp
from scipy.ndimage import gaussian_filter
from scipy import signal
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread
from skimage.morphology import reconstruction
from sewar.full_ref import uqi, mse, rmse, uqi, scc, sam, vifp
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import skew
from statistics import mean, variance
from sympy import DiracDelta


#----------------- PDF 'IMAGE MANIPULATION DETECTION' FUNCTIONS ---------------------#
def extract_bitplane_from_img_array(image_array):
    b = []
    x = bin(int(image_array[0]))[2:].zfill(8)

    for i in range(len(image_array)):
        b.append(bin(int(image_array[i]))[2:].zfill(8))
    
    return b

#-------- LA FUNZIONE DIRAC DELTA GENERA UN IMPULSO PER VALORI UGUALI AD UNA Xo, DA TUTTE LE ALTRE PARTI E' UGUALE A ZERO
def dirac_delta_selector(indicator, dirac_delta_index):
    if indicator == dirac_delta_index:
        return 1
    else:
        return 0

def compute_alpha_agreement(bitplane_array, j_index):
    dirac_delta_sum = 0
    for bitplane in bitplane_array:
        dirac_delta_sum += dirac_delta_selector(bitplane,j_index)
    
    return dirac_delta_sum

def extract_agreements_array(image_array,rows,columns):
    #-------- provare ad ottimizzare la funziona spostando questi calcoli nella la creazione dell'array b -------#
    agreement_a = 0
    agreement_b = 0
    agreement_c = 0
    agreement_d = 0
    #-------- AGREEMENTS -------#
    for elem in image_array:
        #SOMMATORIA per k=1..K con K=4 sono i 4 K-neighborhood definiti tramite 4 coppie di bitplanes (3-4, 4-5, 5-6, 6-7)
        #plan_3 = elem[2]
        plan_4 = elem[3]
        plan_5 = elem[4]
        plan_6 = elem[5]
        plan_7 = elem[6]

        plan_i = elem[7]

        #QUI le 4 coppie di bitplanes
        bitplane_4 = indicator_function(plan_i, plan_4)
        bitplane_3 = indicator_function(plan_i, plan_5)
        bitplane_2 = indicator_function(plan_i, plan_6)
        bitplane_1 = indicator_function(plan_i, plan_7)

        bitplane_array_var = np.array([bitplane_1,bitplane_2,bitplane_3,bitplane_4])

        #QUI i risultati parziali derivati dall'applicazione del dirac delta selector per ogni bitplane ("alpha con i alla j")
        #sum_a = dirac_delta_selector(bitplane_1,1) + dirac_delta_selector(bitplane_2,1) + dirac_delta_selector(bitplane_3,1) + dirac_delta_selector(bitplane_4,1)
        sum_a = compute_alpha_agreement(bitplane_array_var,1)
        sum_b = compute_alpha_agreement(bitplane_array_var,2)
        sum_c = compute_alpha_agreement(bitplane_array_var,3)
        sum_d = compute_alpha_agreement(bitplane_array_var,4)
        
        #QUI le somme aggregate
        agreement_a += sum_a
        agreement_b += sum_b
        agreement_c += sum_c
        agreement_d += sum_d

    #QUI le variabili di agreement per il calcolo delle BSM
    agreement_a /= (rows*columns)
    agreement_b /= (rows*columns)
    agreement_c /= (rows*columns)
    agreement_d /= (rows*columns)

    return np.array([agreement_a, agreement_b, agreement_c, agreement_d])

def extract_local_agreement(image_array, num_local_bitplanes, j_index):    
    agreement_alpha = 0
    agreement_inner = 0
    bitplane_array_var = []
    
    for elem in image_array:
        sum_inner = 0
        bitplane_array_var_temp = []
        plane_i = elem[7]
        #QUESTA operazione è a rischio indexOutOfBoundsException 
        for i in range(num_local_bitplanes):
            bitplane_array_var_temp.append( indicator_function(plane_i, elem[i+3]) )
        
        bitplane_array_var = np.array(bitplane_array_var_temp)
        #bitplane_array_var = np.array([ indicator_function(elem[3], elem[4]) ]) 

        sum_alpha = compute_alpha_agreement(bitplane_array_var,j_index)
        agreement_alpha += sum_alpha

        for j in range(j_index):
            sum_inner_temp = compute_alpha_agreement(bitplane_array_var,j+1)
            sum_inner += sum_inner_temp
        
        #sum_inner += compute_alpha_agreement(bitplane_array_var,1)
        
        agreement_inner += sum_inner

    return agreement_alpha / agreement_inner

def indicator_function(r,s):
    if r == "0" and s == "0":
        return 1
    if r == "0" and s == "1":
        return 2
    if r == "1" and s == "0":
        return 3
    if r == "1" and s == "1":
        return 4
    return None

def bitplane_extraction(byte_array, bitplane_index):
    return byte_array[bitplane_index]

################################################
def sneeth_and_sokai_sm1(a,b,c,d):
    num = 2*(a+d)
    denum = 2*(a+d)+b+c
    return num/denum

def sneeth_and_sokai_sm2(a,b,c):
    num = a
    denum = a+2*(b+c)
    return num/denum

def sneeth_and_sokai_sm3(a,b,c,d):
    num = (a+d)
    denum = (b+c)
    return num/denum

def sneeth_and_sokai_sm4(a,b,c,d):
    first_member = a/(a+b)
    second_member = a/(a+c)
    third_member = d/(b+d)
    fourth_member = d/(c+d)

    num = first_member + second_member + third_member + fourth_member
    denum = 4
    return num/denum

def sneeth_and_sokai_sm5(a,b,c,d):
    denum_elem = (a+b)*(a+c)*(b+d)*(c+d)

    num = a*d
    denum = denum_elem ** (1/2)
    return num/denum

def kulczynski_sm1(a,b,c):
    num = a
    denum = (b+c)
    return num/denum

def ochiai_sm1(a,b,c):
    first_member = (a/(a+b))
    second_member = (a/(a+c))

    return (first_member * second_member) ** (1/2)

def lance_and_williams_dissm(a,b,c):
    num = b+c
    denum = 2*a + b + c

    return num/denum

def pattern_difference(a,b,c,d):
    num = b*c

    denum_elem = a+b+c+d
    denum = denum_elem ** 2

    return num/denum

def variance_dissimilarity_measure(image_array):
    measure = 0
    for n in range(4):
        p1 = extract_local_agreement(image_array,1,n+1)
        p2 = extract_local_agreement(image_array,2,n+1)
        if p1 < p2:
            measure += p1
        else:
            measure += p2
    return measure

################################################



#----------------- IMAGE METHODS ---------------------#
def read_image(image_path):
    image = imread(image_path)
    print('reading image')
    #print(image)
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


dir = "./src/resources/imgs"

#feature_img_matrix = feature_extraction_from_dir(dir)
#print_feature_matrix(feature_img_matrix)
testdir = "./src/resources/100.png"
testdir_t = "./src/resources/100t.png"
testdir_t2 = "./src/resources/100t2.tiff"
#dataset = feature_img_matrix

feature_img = []
feature_img_matrix_test = []
#BASE IMG

#--------- TEST FUNCTIONS WITH PLANES ----------#
testdir_img = np.array(read_image(testdir))
r = len(testdir_img)
c = len(testdir_img[0])
flat = testdir_img.flatten()

#--------- FUNCTION TO CONVERT INT INTO BYTE ----------#
b = extract_bitplane_from_img_array(flat)

#Tutto quello che segue è corretto
agreements_array = extract_agreements_array(b, r, c)
print(agreements_array)

sneeth_and_sokai = sneeth_and_sokai_sm1(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
sneeth_and_sokai_2 = sneeth_and_sokai_sm2(agreements_array[0], agreements_array[1], agreements_array[2])
sneeth_and_sokai_3 = sneeth_and_sokai_sm3(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
sneeth_and_sokai_4 = sneeth_and_sokai_sm4(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
sneeth_and_sokai_5 = sneeth_and_sokai_sm5(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
kulczynski_similarity = kulczynski_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
ochiai_similarity = ochiai_sm1(agreements_array[0], agreements_array[1], agreements_array[2])
lance_and_williams_dissimilarity = lance_and_williams_dissm(agreements_array[0], agreements_array[1], agreements_array[2])
pattern_diff = pattern_difference(agreements_array[0], agreements_array[1], agreements_array[2], agreements_array[3])
print("sneeth_and_sokai")
print(sneeth_and_sokai)
print("sneeth_and_sokai 2")
print(sneeth_and_sokai_2)
print("sneeth_and_sokai 3")
print(sneeth_and_sokai_3)
print("sneeth_and_sokai 4")
print(sneeth_and_sokai_4)
print("sneeth_and_sokai 5")
print(sneeth_and_sokai_5)
print("kulczynski_similarity")
print(kulczynski_similarity)
print("ochiai_similarity")
print(ochiai_similarity)
print("lance_and_williams_dissimilarity")
print(lance_and_williams_dissimilarity)
print("pattern_diff")
print(pattern_diff)

variance_diss = variance_dissimilarity_measure(b)
print("variance_diss")
print(variance_diss)


#--------- --------------------------------- ----------#

#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#
print()
print()
#_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-#

testdir_img_t = np.array(read_image(testdir_t))
r_t = len(testdir_img_t)
c_t = len(testdir_img_t[0])
flat_t = testdir_img_t.flatten()

b_t = extract_bitplane_from_img_array(flat_t)

agreements_array_t = extract_agreements_array(b_t, r_t, c_t)
print(agreements_array_t)

sneeth_and_sokai_t = sneeth_and_sokai_sm1(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2], agreements_array_t[3])
sneeth_and_sokai2_t = sneeth_and_sokai_sm2(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2])
sneeth_and_sokai3_t = sneeth_and_sokai_sm3(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2], agreements_array_t[3])
sneeth_and_sokai4_t = sneeth_and_sokai_sm4(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2], agreements_array_t[3])
sneeth_and_sokai5_t = sneeth_and_sokai_sm5(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2], agreements_array_t[3])
kulczynski_similarity_t = kulczynski_sm1(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2])
ochiai_similarity_t = ochiai_sm1(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2])
lance_and_williams_dissimilarity_t = lance_and_williams_dissm(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2])
pattern_diff_t = pattern_difference(agreements_array_t[0], agreements_array_t[1], agreements_array_t[2], agreements_array_t[3])
print("sneeth_and_sokai_t")
print(sneeth_and_sokai_t)
print("sneeth_and_sokai2_t")
print(sneeth_and_sokai2_t)
print("sneeth_and_sokai3_t")
print(sneeth_and_sokai3_t)
print("sneeth_and_sokai4_t")
print(sneeth_and_sokai4_t)
print("sneeth_and_sokai5_t")
print(sneeth_and_sokai5_t)
print("kulczynski_similarity_t")
print(kulczynski_similarity_t)
print("ochiai_similarity_t")
print(ochiai_similarity_t)
print("lance_and_williams_dissimilarity_t")
print(lance_and_williams_dissimilarity_t)
print("pattern_diff_t")
print(pattern_diff_t)

"""p1_t = extract_local_agreement(b_t,1,4)
print("p1_t")
print(p1_t)
p2_t = extract_local_agreement(b_t,2,4)
print("p2_t")
print(p2_t)"""

variance_diss_t = variance_dissimilarity_measure(b_t)
print("variance_diss_t")
print(variance_diss_t)



"""
image = img_as_float(read_image(testdir))
print(image.flatten())

f_image = filtered_image(image, "gaussian")
print(f_image.flatten())
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

#print_feature_matrix(dataset)
#print_feature_matrix(testset)"""

