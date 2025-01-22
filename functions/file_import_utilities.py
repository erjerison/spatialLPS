import pandas as pd


def import_sample_list(sample_key_file,column='None'):
	
	sample_table = pd.read_csv(sample_key_file,sep='\t',header=0,index_col=0,
		dtype={'Sample Name':str,'Treatment Group':str,'10 hr Timepoint':bool,'Medium to High Activation':bool,'Medium to High Activation and Pass Morphology Filter':bool})

	if column == 'None':
		sample_list = sample_table.index

	else:
		sample_list = sample_table.index[sample_table[column]]

	return sample_list

def import_bg_dict(bg_measurement_file,threshhold='99.5',channel_nums=['0','1','2','4','5','6','8','9','10']):

	bg_df = pd.read_csv(bg_measurement_file,sep='\t',header=0,index_col=0)

	thresh_vals = (bg_df[threshhold].values[:9] + bg_df[threshhold].values[9:])/2
	bg_dict = dict(zip(channel_nums,thresh_vals))

	return bg_dict

def import_color_key(color_key_file):

	color_table = pd.read_csv(color_key_file,sep='\t',header=0,dtype=str)

	color_dict = dict(zip(color_table['Treatment Group'],color_table['Color']))

	return color_dict

def import_conc_key(conc_key_file):


	conc_table = pd.read_csv(conc_key_file,sep='\t',header=0,dtype={'Treatment Group':str,'LPS Concentration':float})

	conc_dict = dict(zip(conc_table['Treatment Group'],conc_table['LPS Concentration']))

	return conc_dict

def import_treatment_dict(sample_key_file):

	sample_table = pd.read_csv(sample_key_file,sep='\t',header=0,
		dtype={'Sample Name':str,'Treatment Group':str,'10 hr Timepoint':bool,'Medium to High Activation':bool,'Medium to High Activation and Pass Morphology Filter':bool})

	treatment_dict = dict(zip(sample_table['Sample Name'],sample_table['Treatment Group']))

	return treatment_dict