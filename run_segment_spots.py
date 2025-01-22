import keys.paths_filenames as paths_filenames
import functions.segmentation_and_spot_counting_functions as segmentation_and_spot_counting_functions
import functions.file_import_utilities as file_import_utilities
from pathlib import Path

sample_list = file_import_utilities.import_sample_list( paths_filenames.keys_path +'/Sample_Key.txt')

bg_dict = file_import_utilities.import_bg_dict(paths_filenames.bg_file)

for sample in sample_list:

	print(sample)

	Path(paths_filenames.table_path + '/' + sample).mkdir(exist_ok=True)

	dapi_file = paths_filenames.image_path + sample + '/' + sample + paths_filenames.dapi_channel

	mask_file = paths_filenames.image_path + sample + '/' + sample + '-mask.tif'

	segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

	segmentation_image_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation-image.pdf'

	segmentation_and_spot_counting_functions.segment_tissue(dapi_file,mask_file,segmentation_file,segmentation_image_file)

	segmentation_and_spot_counting_functions.measure_cell_stats(paths_filenames.image_path,paths_filenames.table_path,sample,bg_dict)