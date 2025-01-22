import keys.paths_filenames as paths_filenames
import functions.tail_graph_functions as tail_graph_functions
import functions.file_import_utilities as file_import_utilities

sample_list = file_import_utilities.import_sample_list( paths_filenames.keys_path +'/Sample_Key.txt')

for sample in sample_list:

	print(sample)

	segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

	print('Calculating normed A')

	Anorm = tail_graph_functions.calculate_save_adjacency_matrix(segmentation_file,paths_filenames.table_path,sample)

	print('Calculating eigenvectors')

	tail_graph_functions.calculate_save_eigenvectors(Anorm,paths_filenames.table_path,sample)