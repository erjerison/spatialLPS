from glob import glob
import register_dapi_cluster_rotation

###Helper functions

def get_sample_list(sample_list_file):
	###This file contains sample names to be processed, 1 key per line.
	file = open(sample_list_file,'r')
	sample_list = []
	for line in file:
		sample_list.append(line.strip())
	return sample_list

def write_output_log(sample_name):
	filename = WORKING_DIR + '/image_processing/'+ str(sample_name) + '/' + str(sample_name) + '_output_log.txt'
	file = open(filename,'w')
	file.write('Saved registered .tif files for sample ' + str(sample_name)+ 'to google drive, path: zfish_LPS_RNAscope/registered/' + str(sample_name))
	file.close()

###Variable definitions

SAMPLES = get_sample_list("sample_list_processing_10302022.txt")
WORKING_DIR = "/oak/stanford/groups/quake/ejerison/lps_phases/in_situ/data_processing"
GDRIVE_DIR = "zfish_LPS_RNAscope/10272022"