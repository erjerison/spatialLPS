import keys.paths_filenames as paths_filenames
import functions.tail_graph_functions as tail_graph_functions
import functions.file_import_utilities as file_import_utilities
import numpy as np

sample_list = file_import_utilities.import_sample_list( paths_filenames.keys_path +'/Sample_Key.txt')

for sample in sample_list:
	
    print(sample)
	
    segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

    centroid_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'

    print('Calculating alpha shape')


    centroids = np.load(centroid_file)

    if sample in ['d-08052022_LPS20ugml_10hrs_tail2', 'd-08092022_LPS30ugml_4hrs_tail4', 'd-08192022_LPS25ugml_10hrs_tail4',
                   'd-08192022_LPS25ugml_10hrs_tail5', 'd-08092022_LPS30ugml_10hrs_tail5', 'd-08092022_LPS30ugml_10hrs_tail4']:
        alpha = 0.002
    else:
        alpha = 0.007

    triangles, edge_points, boundary_vertices = tail_graph_functions.calculate_save_alphashape(centroid_file, paths_filenames.table_path, sample, alpha=alpha) #0.003 for passing morphology filters.

    print('Calculating cotan Laplacian')
    # I need the vertices (stored in centroid_file), and the face list (output from alpha_shape) to compute Lop

    #print(centroids.shape)
    #print(triangles.shape)
    Lop = tail_graph_functions.calculate_save_cotan_laplacian(centroids,triangles, paths_filenames.table_path,sample)

    #print(Lop)
    print('Calculating eigenvectors')
    # once I have Lop this is easy
    tail_graph_functions.calculate_save_cotan_eigenvectors(Lop, paths_filenames.table_path,sample)