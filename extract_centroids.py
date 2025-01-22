import functions.file_import_utilities as file_import_utilities
import keys.paths_filenames as paths_filenames
import pandas as pd
import numpy

sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key)

for sample in sample_list:

	print(sample)

	centroids_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cell_stats.txt'


	cell_stats = pd.read_csv(centroids_file,sep='\t',header=0)

	centroid_list = []

	for entry in cell_stats['Centroid']:
		coords = entry.split(',')
		centroid_list.append( [float(coords[0]),float(coords[1])] )

	centroid_arr = numpy.array(centroid_list)

	centroid_output_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'

	numpy.save(centroid_output_file,centroid_arr)
