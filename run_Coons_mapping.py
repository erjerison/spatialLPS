import numpy
import matplotlib.pylab as pt
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.Coons_mapping_functions import Coons_mapping

sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation and Pass Morphology Filter')

for sample in sample_list:

	print(sample)

	mask_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-mask.tif'
	centroid_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'

	st_centroids = Coons_mapping(mask_file,centroid_file)

	numpy.save( paths_filenames.table_path + '/' + sample + '/' + sample + '-st_centroids.npy', st_centroids)