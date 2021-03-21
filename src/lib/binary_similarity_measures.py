import numpy as np

#----------------- PDF 'IMAGE MANIPULATION DETECTION' FUNCTIONS ---------------------#
def extract_bitplane_from_img_array(image_array):
    b = []
    x = bin(int(image_array[0]))[2:].zfill(8)

    for i in range(len(image_array)):
        b.append(bin(int(image_array[i]))[2:].zfill(8))
    
    return b

#-------- DIRACT DELTA FUNCTION GENERATES AN IMPULSE FOR VALUES UQUALS TO x0, EVERYWHERE ELSE IS ZERO
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
    agreement_a = 0
    agreement_b = 0
    agreement_c = 0
    agreement_d = 0
    #-------- AGREEMENTS -------#
    for elem in image_array:
        plan_4 = elem[3]
        plan_5 = elem[4]
        plan_6 = elem[5]
        plan_7 = elem[6]

        plan_i = elem[7]

        #BITPLANES COUPLES
        bitplane_4 = indicator_function(plan_i, plan_4)
        bitplane_3 = indicator_function(plan_i, plan_5)
        bitplane_2 = indicator_function(plan_i, plan_6)
        bitplane_1 = indicator_function(plan_i, plan_7)

        bitplane_array_var = np.array([bitplane_1,bitplane_2,bitplane_3,bitplane_4])

        #PARTIAL RESULTS FOR DELTA DIRACT SELECTOR APPLY
        sum_a = compute_alpha_agreement(bitplane_array_var,1)
        sum_b = compute_alpha_agreement(bitplane_array_var,2)
        sum_c = compute_alpha_agreement(bitplane_array_var,3)
        sum_d = compute_alpha_agreement(bitplane_array_var,4)
        
        #AGGREGATE SUMS
        agreement_a += sum_a
        agreement_b += sum_b
        agreement_c += sum_c
        agreement_d += sum_d

    #AGREEMENTS VARIABLES FOR BSM OPERATIONS
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
        
        for i in range(num_local_bitplanes):
            bitplane_array_var_temp.append( indicator_function(plane_i, elem[i+3]) )
        
        bitplane_array_var = np.array(bitplane_array_var_temp)

        sum_alpha = compute_alpha_agreement(bitplane_array_var,j_index)
        agreement_alpha += sum_alpha

        for j in range(j_index):
            sum_inner_temp = compute_alpha_agreement(bitplane_array_var,j+1)
            sum_inner += sum_inner_temp
        
        agreement_inner += sum_inner

    return agreement_alpha / agreement_inner

#I can't get how to compute ojala from pdf
def extract_ojala_agreements(image_array):
    #-------- AGREEMENTS -------#
    for elem in image_array:
        ojala_weight = compute_ojala_dir_weight(elem)
        

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
            temp = p1
        else:
            temp = p2
        measure += temp

    return measure

def binary_min_histogram_difference(image_array):
    measure = 0
    for n in range(4):
        p1 = extract_local_agreement(image_array,1,n+1)
        p2 = extract_local_agreement(image_array,2,n+1)
        temp = abs(p1-p2)
        measure += temp

    return measure

def binary_absolute_histogram_difference(image_array):
    measure = 0
    for n in range(4):
        p1 = extract_local_agreement(image_array,1,n+1)
        p2 = extract_local_agreement(image_array,2,n+1)
        lg_p2 = np.log(p2)
        temp = p1*lg_p2
        measure += temp

    return measure*(-1)

def binary_mutual_entropy(image_array):
    measure = 0
    for n in range(4):
        p1 = extract_local_agreement(image_array,1,n+1)
        p2 = extract_local_agreement(image_array,2,n+1)
        lg_p1p2 = np.log(p1/p2)
        temp = p1*lg_p1p2
        measure += temp

    return measure*(-1)

################################################