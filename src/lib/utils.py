import numpy as np

def save_matrix_bin(matrix, ds_name):
	with open(ds_name, 'wb') as f:
		np.save(f, matrix)