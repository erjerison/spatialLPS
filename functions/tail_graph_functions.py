import numpy as np
import pandas as pd
from skimage import graph
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay
import scipy.sparse as sp
import scipy.linalg as la
import gpytoolbox as gpy
import math


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

    label_image = np.load(label_image_filename).astype('int')

    g = graph.RAG(label_image)

    ###Remove node 0 and all its edges (this is the background, not a cell)

    g.remove_node(0)

    labels = g.nodes()
    num_labels = len(labels)

    adjacency_matrix = np.zeros((num_labels,num_labels),dtype='int')

    for u,v in g.edges:

        ##note that the nodes are 1-indexed, not 0-indexed (hence the -1)

        adjacency_matrix[u-1,v-1] = 1
        adjacency_matrix[v-1,u-1] = 1

    D_mat_inv = np.diag( 1/np.sqrt(1 + adjacency_matrix.sum(axis=1)) )

    adjacency_matrix_normed = np.matmul( np.matmul(D_mat_inv,adjacency_matrix), D_mat_inv )

    np.save( data_dir + '/' + sample + '/' + sample + '-graph_adj_mat_normed.npy',adjacency_matrix_normed )

    return adjacency_matrix_normed

def calculate_save_eigenvectors2(adj_mat,data_dir,sample):

    """
    This function calculates and saves the eigenvectors of the normalized adjacency matrix,
    via np.linalg.svd

    Inputs
    ------
    adj_mat (numpy array, n_cellsxn_cells): normalized adjacency matrix
    sample (str): sample name

    Returns
    -------
    None (Saves numpy array of eigenvectors in .npy format)
    """

    U,S,VT = np.linalg.svd(adj_mat)

    np.save(data_dir + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy',VT)

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
    Xinv = np.linalg.pinv(X.T)

    B_R = np.matmul(Xinv,data)

    return B_R, X.T


#### alpha-shape and functions for the cotan-Laplacian decomposition

def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set of points.
    see 
    # https://gist.github.com/jclosure/d93f39a6c7b1f24f8b92252800182889

    Inputs
    ------
    points: (numpy array num_points x 3) contains point coordinates
    alpha: alpha value to influence the convexity of the border: alpha is ~1/max radius of curvature of the boundary
    
    Returns
    ------
      triangles
      edge_points
      boundary_vertices: numpy array of vertices at the boundary. 
    """
    #if len(points) < 4:
    #    # When you have a triangle, there is no sense in computing an alpha
    #    # shape.
    #    return geometry.MultiPoint(list(points)).convex_hull

    def add_edge(edges, edge_points, coords, i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

    coords = points #np.array([point for point in points])

    tri = Delaunay(coords)
    edges = set()
    edge_points = []
    boundary_vertices = set()
    # loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    triangles = []
    for ia, ib, ic in tri.simplices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]

        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)

        # Semiperimeter of triangle
        s = (a + b + c)/2.0

        # Area of triangle by Heron's formula
        a2 = s*(s-a)*(s-b)*(s-c)
        if a2 <= 0:
            continue
        area = math.sqrt(a2)
        circum_r = a*b*c/(4.0*area)

        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            triangles.append([int(ia), int(ib), int(ic)])
        else:
            boundary_vertices.add(ia)
            boundary_vertices.add(ib)
            boundary_vertices.add(ic)

    m = edge_points
    return np.array(triangles), np.array(edge_points), np.array(list(boundary_vertices))


def calculate_save_alphashape(centroid_file, data_dir, sample, alpha=0.007):
    """
    Computes the alpha-shape for the given file and saves it

    Inputs
    ------
    centroid_file (str): path to .npy format file with list of cell centroid locations
    data_dir (str): path to save output
    sample (str): sample name
    alpha (float): alpha parameter of the alpha shape. Defaults to 0.007 by experimentations.
    
    Returns
    ------
    triangles: (array num_face x 3) facelist of the mesh
    edge_points: edgelist of the mesh
    boundary_vertices: (array) 
    
    """
    # load centroids
    centroids = np.load(centroid_file)
    ncells,d2 = centroids.shape
    # compute alpha shape
    triangles, edge_points, boundary_vertices = alpha_shape(centroids, alpha)
    # save alpha-shape
    np.save(data_dir + '/' + sample + '/' + sample + '-alpha_facelist.npy',triangles)

    return triangles, edge_points, boundary_vertices

def calculate_save_cotan_laplacian(centroids, faces, data_dir, sample):
    """
    This function calculates and save the cotan Laplacian for the graph connecting centroids with facelist faces.
    
    Inputs
    ------
    centroids (numpy array, n_cellsx3): location of centroids in xy coordinates
    faces (numpy array, n_facesx3): facelist defining the triangulation of the alpha-shape mesh
    data_dir (str): path to save output
    sample (str): sample name

    Returns
    -------
    L (scipy.sparse array,  n_cellsxn_cells): the cotan-Laplacian matrix, weighted by the inverse mass matrix.
    
    """
    L = gpy.cotangent_laplacian(centroids, faces)
    M = gpy.massmatrix(centroids, faces).tocsc()
    m0 = np.mean(M.diagonal())


    if np.any(M.diagonal() == 0):
        print('Average mass ', np.mean(M.diagonal()[M.diagonal()>0])) 
        print('Nans :', np.sum(np.isnan(L.todense().ravel())))
        print('Infs :', np.sum(np.isinf(L.todense().ravel())))
        Lm = L / np.mean(M.diagonal()[M.diagonal()>0])
        w, v  = la.eigh(Lm.todense())
        print(np.sum(np.isinf(w)))
        print(np.sum(np.isinf(v)))
        #Lm.dropna(inplace=True)
        print('Warning: some cells have zero mass; using average mass for normalization')
        print('data_dir/sample: ', data_dir + '/' + sample)
        print('----')

    else:
        Ms = sp.linalg.inv(np.sqrt(M))
        Lm = Ms @ L @ Ms

    np.save(data_dir + '/' + sample + '/' + sample + '-cotan_laplacian.npy',Lm)
    return Lm

def calculate_save_cotan_eigenvectors(Lop,data_dir,sample):
    """
    This function calculates and saves the eigenvectors of the cotan-Laplacian,
    via np.linalg.svd

    Inputs
    ------
    Lop (scipy.sparse array, n_cellsxn_cells): cotan-Laplacian matrix
    sample (str): sample name

    Returns
    -------
    None (Saves numpy array of eigenvectors in .npy format)
    """

    #U,S,VT = np.linalg.svd(Lop.todense())
    #w, v = sp.linalg.eigsh(Lop, k=Lop.shape[0]-10,which='SM')
    w, v = la.eigh(Lop.todense())#,which='SM')

    np.save(data_dir + '/' + sample + '/' + sample + '-cotan_eigenvalues-svd.npy',w)
    np.save(data_dir + '/' + sample + '/' + sample + '-cotan_eigenvectors-svd.npy',v)
      

def lst_sq_cotan(data,spatial_eigenmodes,num_modes):

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
    X (array, n_cells x n_modes)

    """ 

    X = spatial_eigenmodes[:,:num_modes].real
    Xinv = np.linalg.pinv(X)

    B_R = np.matmul(Xinv,data)
    return B_R, X