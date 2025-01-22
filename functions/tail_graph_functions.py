import numpy
import pandas as pd
from skimage import graph
from sklearn.decomposition import PCA

def calculate_save_adjacency_matrix(label_image_filename,data_dir,sample):

	"""

	This function calculates and saves the normalized adjacency matrix for the graph of a particular tail sample,
	Defined as:
	Ahat_ij is an nxn matrix, where n is the number of cells
	Ahat = D^(-1/2) A D^(-1/2), D_ii = 1 + deg(i), 0 otherwise; A_ij (nxn) is the adjacency matrix of the graph
	
	Inputs
	------
	label_image_filename (str): path to an array, with dimensions of the image, with integer entries corresponding to each labeled cell region
	data_dir (str): path to save output
	sample (str): sample name
	
	Returns
	------
	adjacency matrix normed (numpy array, n_cells x n_cells): normalized adjacency matrix (note that the matrix is also saved in .npy format)

	"""

	label_image = numpy.load(label_image_filename).astype('int')

	g = graph.RAG(label_image)

	###Remove node 0 and all its edges (this is the background, not a cell)

	g.remove_node(0)

	labels = g.nodes()
	num_labels = len(labels)

	adjacency_matrix = numpy.zeros((num_labels,num_labels),dtype='int')

	for u,v in g.edges:

		##note that the nodes are 1-indexed, not 0-indexed (hence the -1)

		adjacency_matrix[u-1,v-1] = 1
		adjacency_matrix[v-1,u-1] = 1

	D_mat_inv = numpy.diag( 1/numpy.sqrt(1 + adjacency_matrix.sum(axis=1)) )

	adjacency_matrix_normed = numpy.matmul( numpy.matmul(D_mat_inv,adjacency_matrix), D_mat_inv )

	numpy.save( data_dir + '/' + sample + '/' + sample + '-graph_adj_mat_normed.npy',adjacency_matrix_normed )

	return adjacency_matrix_normed

def calculate_save_eigenvectors2(adj_mat,data_dir,sample):

	"""
	This function calculates and saves the eigenvectors of the normalized adjacency matrix,
	via numpy.linalg.svd

	Inputs
	------
	adj_mat (numpy array, n_cellsxn_cells): normalized adjacency matrix
	sample (str): sample name

	Returns
	-------
	None (Saves numpy array of eigenvectors in .npy format)
	"""

	U,S,VT = numpy.linalg.svd(adj_mat)

	numpy.save(data_dir + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy',VT)

def lst_sq_B(data,spatial_eigenmodes,num_modes):

	"""Decomposition of data onto eigenmodes via pseudo inverse;
		also corresponds to fitting coefficients to best predict the data from the eigenbasis, in a least-squares sense. 

	Inputs
	------
	data (numpy array, n_cells x n_genes): matrix of observations
	spatial_eigenmodes (numpy array, n_cells x n_cells): matrix of eigenvectors
	num_modes (int,1-n_cells): number of modes for fitting

	Returns
	-------
	B_R (array, n_modesx1): Best-fit coefficients for modes 1-n_modes
	X.T (array, n_cells x n_modes)

	""" 

	X = spatial_eigenmodes[:num_modes,:]
	Xinv = numpy.linalg.pinv(X.T)

	B_R = numpy.matmul(Xinv,data)

	return B_R, X.T