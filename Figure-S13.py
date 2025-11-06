###Get sample keys, and plot the 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



import numpy
import matplotlib.pylab as pt
import functions.tailmap_plotting_functions as tailmap_plotting_functions
import functions.tail_graph_functions as tail_graph_functions
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.tail_graph_functions import lst_sq_cotan
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

import scipy.linalg as la
import scipy.sparse.linalg as spla
import gpytoolbox as gpy


import scipy
import pandas as pd

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

if __name__ == '__main__':

    sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')

    plt.figure(figsize=(7,8.6))
    count=0
    for sample in sample_list:
        print(sample)

        #sample = sample_list[3]

        eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cotan_eigenvectors-svd.npy'

        eigenvalue_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cotan_eigenvalues-svd.npy'

        segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

        centroid_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'

        if sample in ['d-08052022_LPS20ugml_10hrs_tail2', 'd-08092022_LPS30ugml_4hrs_tail4', 'd-08192022_LPS25ugml_10hrs_tail5', 'd-08092022_LPS30ugml_10hrs_tail4','d-08092022_LPS30ugml_10hrs_tail5']:
            alpha = 0.002
            plot_text=True
        elif sample in ['d-08192022_LPS25ugml_10hrs_tail4']:
            alpha = 0.0018
            plot_text = True
        else:
            alpha = 0.007
            plot_text=False

        triangles, edge_points, boundary_vertices = tail_graph_functions.calculate_save_alphashape(centroid_file, paths_filenames.table_path, sample, alpha=alpha) #0.007

        centroids = 0.12*np.load(centroid_file)

        eigenvector_mat = np.load(eigenvector_file)
        eigenvalues = np.load(eigenvalue_file)

        L = gpy.cotangent_laplacian(centroids,triangles)

        w, v = la.eigh(L.todense())

        plt.subplot(5,4,count+1)
        count+=1
        plt.scatter(centroids[:,0],centroids[:,1],s=0.4,c=eigenvector_mat[:,1],cmap='PuOr')

        plt.scatter(centroids[:,0][boundary_vertices], centroids[:,1][boundary_vertices], 0.4, color=[88/256, 88/256, 88/256])

        for edge in edge_points:
            plt.plot(0.12*edge.T[0], 0.12*edge.T[1], 'k-', linewidth=0.25, zorder=-1)
        plt.gca().set_axis_off()
    plt.tight_layout()
    plt.savefig('Fig_S13_bare.pdf')
