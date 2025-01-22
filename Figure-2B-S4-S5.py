import numpy
import matplotlib.pylab as pt
import pandas as pd
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.cell_cell_spatial_correlation_functions import cell_spatial_autocorrelation
import scipy
from matplotlib.lines import Line2D

def plot_autocorrelations(output_name,mode='Linear',data_src='Counts'):

	"""
	Parameters
	----------
	output_name (str, extension specifies file type)--name for output plot
	mode (str, if 'Linear', y axis is plotted on a linear scale; otherwise, y axis is plotted on log scale)
	data (str, 'Counts' or 'Intensities', takes the spot counts as inputs; if 'Intensities', takes the pixel intensities as inputs)

	Returns
	-------
	None (saves plot)
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds = [0,1,2,3,4,5,6,8]
	microns_per_pixel = .12

	fig,axes = pt.subplots(2,4,figsize=(10,6),sharey=True,sharex=True)
	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu$ g/mL 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}

	xbins = numpy.arange(-50,21*100,100)

	bin_centers = (xbins[:-1]+xbins[1:])/2.

	###Make artists for the legend

	sample_cat_list = ['LPS25ugml_10hrs', 'LPS27ugml_10hrs','LPS30ugml_4hrs','LPS30ugml_10hrs']

	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu$ g/mL 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}


	lines=[]
	labels = []
	for sample_cat in sample_cat_list:

		color = color_dict[sample_cat]
		conc = conc_dict[sample_cat]
		label = sample_cat_to_label[sample_cat]
		line=Line2D([1,2,3],[4,5,6],color=color,linewidth=2,alpha=.7)
		lines.append(line)
		labels.append(label)

	line=Line2D([1,2,3],[4,5,6],color='grey',linewidth=1,alpha=.5)
	lines.append(line)
	labels.append('Permuted')

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]
		color = color_dict[sample_cat]

		if data_src == 'Counts':

			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'

		if data_src == 'Intensities':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'

		centroids_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cell_stats.txt'

		data_mat = numpy.load(data_file)

		cell_stats = pd.read_csv(centroids_file,sep='\t',header=0)

		centroid_list = []

		for entry in cell_stats['Centroid']:
			coords = entry.split(',')
			centroid_list.append( [float(coords[0]),float(coords[1])] )

		centroid_arr = numpy.array(centroid_list)

		n_cells, n_genes = data_mat.shape

		ax_counter = 0
		for g in gene_inds:

			data_vec = data_mat[:,g]

			autocorr_est, per_bin_std, per_bin_counts = cell_spatial_autocorrelation(data_vec,centroid_arr,bins=xbins)
			autocorr_est_perm, per_bin_std_perm, per_bin_counts_perm = cell_spatial_autocorrelation(numpy.random.permutation(data_vec),centroid_arr,bins=xbins)

			ax = axes[ax_counter//4,ax_counter%4]

			if mode=='Linear':

				ax.plot( bin_centers*microns_per_pixel, autocorr_est, color=color, marker='o', markersize=1.5, alpha=.7)
				ax.plot( bin_centers*microns_per_pixel, autocorr_est_perm, color='grey', alpha=.7)

			else:

				ax.semilogy( bin_centers*microns_per_pixel, autocorr_est, color=color, marker='o', markersize=1.5, alpha=.7)

			if ax_counter%4 == 0:
				ax.set_ylabel('Autocorrelation')
			if ax_counter//4 > .5:
				ax.set_xlabel(r'Distance ($\mu m$)')
			ax.set_title(paths_filenames.gene_list[g])
			ax.spines[['right', 'top']].set_visible(False)

			ax_counter += 1

	axes[0,0].legend(lines,labels,fontsize=9)

	pt.savefig(output_filename,bbox_inches='tight')


def plot_autocorrelations_example_genes(output_name,mode='Linear',data_src='Intensities'):

	"""
	Parameters
	----------
	output_name (str, extension specifies file type)--name for output plot
	mode (str, if 'Linear', y axis is plotted on a linear scale; otherwise, y axis is plotted on log scale)
	-------
	Returns:
	None (saves plot)
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds = [0,2,4]
	microns_per_pixel = .12

	fig,axes = pt.subplots(1,3,figsize=(8,3),sharey=True,sharex=True)
	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu$ g/mL 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}

	xbins = numpy.arange(-50,21*100,100)

	bin_centers = (xbins[:-1]+xbins[1:])/2.

	###Make artists for the legend

	sample_cat_list = ['LPS25ugml_10hrs', 'LPS27ugml_10hrs','LPS30ugml_4hrs','LPS30ugml_10hrs']

	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu g/mL$ 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}


	lines=[]
	labels = []
	for sample_cat in sample_cat_list:

		color = color_dict[sample_cat]
		conc = conc_dict[sample_cat]
		label = sample_cat_to_label[sample_cat]
		line=Line2D([1,2,3],[4,5,6],color=color,linewidth=2,alpha=.7)
		lines.append(line)
		labels.append(label)

	line=Line2D([1,2,3],[4,5,6],color='grey',linewidth=1,alpha=.5)
	lines.append(line)
	labels.append('Permuted')

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]
		color = color_dict[sample_cat]

		if data_src == 'Counts':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'
		elif data_src == 'Intensities':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'
		centroids_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cell_stats.txt'

		data_mat = numpy.load(data_file)

		cell_stats = pd.read_csv(centroids_file,sep='\t',header=0)

		centroid_list = []

		for entry in cell_stats['Centroid']:
			coords = entry.split(',')
			centroid_list.append( [float(coords[0]),float(coords[1])] )

		centroid_arr = numpy.array(centroid_list)

		n_cells, n_genes = data_mat.shape

		ax_counter = 0
		for g in gene_inds:

			data_vec = data_mat[:,g]

			autocorr_est, per_bin_std, per_bin_counts = cell_spatial_autocorrelation(data_vec,centroid_arr,bins=xbins)
			print(per_bin_std,per_bin_counts)
			autocorr_est_perm, per_bin_std_perm, per_bin_counts_perm = cell_spatial_autocorrelation(numpy.random.permutation(data_vec),centroid_arr,bins=xbins)

			ax = axes[ax_counter]

			if mode=='Linear':

				ax.plot( bin_centers*microns_per_pixel, autocorr_est, color=color, marker='o', markersize=1.5, alpha=.7)
				ax.plot( bin_centers*microns_per_pixel, autocorr_est_perm, color='grey', alpha=.7)

			else:

				ax.semilogy( bin_centers*microns_per_pixel, autocorr_est, color=color, marker='o', markersize=1.5, alpha=.7)

			if ax_counter%3 == 0:
				ax.set_ylabel('Autocorrelation')

			ax.set_xlabel(r'Distance ($\mu m$)')
			ax.set_title(paths_filenames.gene_list[g])
			ax.spines[['right', 'top']].set_visible(False)

			ax_counter += 1
			
	axes[0].legend(lines,labels,fontsize=9)

	pt.savefig(output_filename,bbox_inches='tight')


if __name__=='__main__':

	plot_autocorrelations_example_genes(output_name='Fig2B-intensities.pdf',mode='Linear',data_src='Intensities')
	plot_autocorrelations(output_name='FigS4-autocorr-all-intensities.pdf',mode='Linear',data_src='Intensities')
	plot_autocorrelations(output_name='FigS5-autocorr-all-counts.pdf',mode='Linear',data_src='Counts')




