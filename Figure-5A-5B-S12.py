import numpy
import matplotlib.pylab as pt
import functions.tailmap_plotting_functions as tailmap_plotting_functions
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.tail_graph_functions import lst_sq_B
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import scipy
import pandas as pd


def plot_power_spectra_cdfs_and_summary(output_name,mode='Log',data_src='Counts'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)--output name for the figure panel
	mode (str): if 'Log', data will be log-transformed prior to decomposition. Otherwise, will be left as raw counts.
	data_src (str): 'Counts' or 'Intensities'
	Returns
	-------
	None (plots figure)
	"""

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds_all = [0,1,2,3,4,5,6,8]
	gene_inds_examples = [0,2,4]

	gene_names = [paths_filenames.gene_list[g] for g in gene_inds_all]

	figname_power = paths_filenames.figure_path + '/' + output_name + '-' + mode + '-' + data_src + '.pdf'


	frac_power_fig = pt.figure(figsize=(11,8))

	gs = GridSpec(2,3,figure=frac_power_fig)

	ax_list1 = []
	ax_list2 = []

	for i in range(3):

		ax1 = frac_power_fig.add_subplot(gs[0,i])
		ax_list1.append(ax1)

	axes_power = numpy.array([ax_list1])

	frac_var_ax = frac_power_fig.add_subplot(gs[1,1:])

	ncol = 3

	###Make artists for the legend

	sample_cat_list = ['LPS25ugml_10hrs', 'LPS27ugml_10hrs','LPS30ugml_4hrs','LPS30ugml_10hrs']

	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\frac{\mu g}{mL}$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\frac{\mu g}{mL}$ 10 hrs','LPS30ugml_4hrs':r'$30\,\frac{\mu g}{mL}$ 4 hrs','LPS30ugml_10hrs':r'$30\,\frac{\mu g}{mL}$ 10 hrs' }


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

	frac_vars = []

	sample_counter = 0

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]
		color = color_dict[sample_cat]

		eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy'

		if data_src == 'Counts':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'
		elif data_src == 'Intensities':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'

		areas_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cell_stats.txt'

		###Import data tables

		data_mat = numpy.load(data_file)
		eigenvector_mat = numpy.load(eigenvector_file)


		if mode == 'Log':

			data_mat = numpy.log2(data_mat+1)

		n_cells, n_genes = data_mat.shape

		###Center data

		data_mat_centered = data_mat - numpy.outer(numpy.ones((n_cells,),dtype=float),data_mat.mean(axis=0))

		###Decompose

		B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)

		###Permuted cells

		B_R_perm,XT_perm = lst_sq_B(data_mat_centered[numpy.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)

		###Plot power spectra


		axes_counter = 0

		gene_counter = 0

		frac_vars_sample = []

		for g in gene_inds_all:

			###Calculate power spectrum and length scale via mean absolute deviation

			frac = (B_R**2)[:,g]/numpy.sum((B_R**2)[:,g])

			smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)

			cum_frac = numpy.cumsum((B_R**2)[:,g])/numpy.sum((B_R**2)[:,g])

			frac_perm = (B_R_perm**2)[:,g]/numpy.sum((B_R_perm**2)[:,g])

			cum_frac_perm = numpy.cumsum((B_R_perm**2)[:,g])/numpy.sum((B_R_perm**2)[:,g])

			smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)

			###Measure the amount of power per mode in the null to determine the noise floor

			fnull = numpy.mean(frac_perm)

			###Determine when the spectrum goes below the noise floor

			very_smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=10)

			crossover_mode = numpy.argmax( very_smoothed_frac < fnull ) ###First entry where the data curve drops below the noise floor

			frac_var = cum_frac[crossover_mode]
			total_var_gene = numpy.var(data_mat_centered[:,g])

			frac_var_ax.plot(gene_counter + .1*numpy.random.random()-.05, frac_var, marker='o',alpha=.85,color=color)

			frac_vars_sample.append(frac_var)

			if g in gene_inds_examples:

				ax_cdf = axes_power[0,axes_counter]

				ax_cdf.plot(numpy.arange(1,len(smoothed_frac)+.5),cum_frac,color=color,linewidth=2,alpha=.85)

				ax_cdf.plot(numpy.arange(1,len(smoothed_frac)+.5),cum_frac_perm,color='grey',linewidth=1,alpha=.5)
				
				ax_cdf.set_xlabel('Mode Number',fontsize=14)

				ax_cdf.set_title(paths_filenames.gene_list[g],fontsize=14)

				if axes_counter < .5:
					ax_cdf.set_ylabel(r"Variance fraction",fontsize=14)
					ax_cdf.text(-350,1,'A.',fontsize=24)

				axes_counter += 1

			gene_counter += 1

		frac_vars.append(frac_vars_sample)

		sample_counter += 1


	frac_vars = numpy.array(frac_vars)

	frac_var_ax.plot(numpy.arange(8),numpy.mean(frac_vars,axis=0),"k_",markersize=20,zorder=10)
	frac_var_means = numpy.mean(frac_vars,axis=1)

	frac_var_ax.set_ylabel(r"Variance fraction, long modes",fontsize=14)

	frac_var_ax.legend(lines,labels,fontsize=13,loc=[-.6,.15])

	frac_var_ax.set_xticks(range(8))
	frac_var_ax.set_xlim([-.5,7.5])

	frac_var_ax.text(-6,.8,'B.',fontsize=24)


	frac_var_ax.set_xticklabels(gene_names)


	frac_power_fig.savefig(figname_power,bbox_inches='tight')



if __name__=='__main__':

	plot_power_spectra_cdfs_and_summary(output_name='Fig5-variance-explained',data_src='Intensities',mode='Log')
	plot_power_spectra_cdfs_and_summary(output_name='FigS12-variance-explained',data_src='Counts',mode='Log')
