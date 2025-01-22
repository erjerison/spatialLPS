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

def plot_mode_structure_examples(output_name,example_modes = [0,1,6,21,201],alpha_bg=1,cm='PuOr'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)--name for plot
	example_modes (integers)--0-indexed mode numbers to be plotted; note that mode 0 will be relabeled as mode 1 for the plots (and so forth)
	------
	Returns
	None (produces plot)
	------
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')

	sample = sample_list[0]

	eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy'

	segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

	eigenvector_mat = numpy.load(eigenvector_file)

	tailmap_fig1,axes_tailmap = pt.subplots(1,len(example_modes),figsize=(15,5))

	for i in range(len(example_modes)):

		mode = example_modes[i]
		tailmap_plotting_functions.plot_cell_intensities_diverging_cm(segmentation_file, eigenvector_mat[mode,:]-numpy.min(eigenvector_mat[mode,:]), axes_tailmap[i], sample,cm=cm,alpha_bg=alpha_bg)
		axes_tailmap[i].set_title('Mode '+str(mode+1),fontsize=16)

	pt.savefig(output_filename, bbox_inches = 'tight')

def plot_power_spectra(output_name,data_scale='log',spectrum_mode='cdf',plotting_mode='log',data_src='Counts'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)--output name for the figure panel
	data_scale (str): if 'log', data will be log-transformed prior to decomposition. Otherwise, will be left as raw counts.
	spectrum_mode (str): 'cdf' or 'pdf.' 'pdf'--plots the (normalized) squared coefficients of the projection onto the eigenmodes. 'cdf'--plots the sum of these coefficients, up to and including the current mode number.
	plotting_mode (str): scale for plotting axes. 'log'-- loglog plot; otherwise, linear plot.
	data_src (str): 'Counts' or 'Intensities'
	------
	Returns:
	None (plots figure)
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name + '-data-' + data_scale + '-' + spectrum_mode + '-' + plotting_mode + '-' + data_src + '.pdf'

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	frac_power_fig,axes_power = pt.subplots(2,4,figsize=(16,8),sharey=True)

	ncol = 4

	###Make artists for the legend

	sample_cat_list = ['LPS25ugml_10hrs', 'LPS27ugml_10hrs','LPS30ugml_4hrs','LPS30ugml_10hrs']

	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu$ g/mL 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}

	###Specify genes to loop over (all genes other than macrophage marker)

	gene_inds_all = [0,1,2,3,4,5,6,8]

	gene_names = [paths_filenames.gene_list[g] for g in gene_inds_all]

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

	vars_long = []
	mode50_list = []

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]
		color = color_dict[sample_cat]

		eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy'

		if data_src == 'Counts':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'
		elif data_src == 'Intensities':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'

		###Import data tables

		data_mat = numpy.load(data_file)
		eigenvector_mat = numpy.load(eigenvector_file)

		if data_scale == 'log':

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

		for g in gene_inds_all:

			ax = axes_power[axes_counter//4,axes_counter%4]

			frac = (B_R**2)[:,g]/numpy.sum((B_R**2)[:,g])

			smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
			cum_frac = numpy.cumsum((B_R**2)[:,g])/numpy.sum((B_R**2)[:,g])


			###Permute cells for null

			frac_perm = (B_R_perm**2)[:,g]/numpy.sum((B_R_perm**2)[:,g])

			cum_frac_perm = numpy.cumsum((B_R_perm**2)[:,g])/numpy.sum((B_R_perm**2)[:,g])
			smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)

			if spectrum_mode == 'cdf':

				data_to_plot = cum_frac
				perm_to_plot = cum_frac_perm

			else:

				data_to_plot = smoothed_frac
				perm_to_plot = smoothed_frac_perm

			if plotting_mode == 'log':

				ax.loglog(numpy.arange(1,len(smoothed_frac)+.5),data_to_plot,color=color,linewidth=2,alpha=.7)
				ax.loglog(numpy.arange(1,len(smoothed_frac)+.5),perm_to_plot,color='grey',linewidth=1,alpha=.5)

			else:

				ax.plot(numpy.arange(1,len(smoothed_frac)+.5),data_to_plot,color=color,linewidth=2,alpha=.7)
				ax.plot(numpy.arange(1,len(smoothed_frac)+.5),perm_to_plot,color='grey',linewidth=1,alpha=.5)

			if g > 3.5:
				ax.set_xlabel('Mode Number',fontsize=16)

			ax.set_title(paths_filenames.gene_list[g],fontsize=16)

			ax.spines[['right', 'top']].set_visible(False)

			axes_counter += 1


		axes_power[0,0].set_ylabel('Fraction of Power',fontsize=16)
		axes_power[1,0].set_ylabel('Fraction of Power',fontsize=16)

		axes_power[0,0].legend(lines,labels,loc='upper right',fontsize=10)


	frac_power_fig.savefig(output_filename,bbox_inches='tight')


def plot_power_spectra_example_genes(output_name,mode='Log',data_src='Intensities'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)--output name for the figure panel
	mode (str): if 'Log', data will be log-transformed prior to decomposition. Otherwise, will be left as raw counts.

	Returns
	-------
	None (plots figure)
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds_all = [0,1,2,3,4,5,6,8]
	gene_inds_examples = [0,2,4]

	gene_names = [paths_filenames.gene_list[g] for g in gene_inds_all]

	figname_power = paths_filenames.figure_path + '/' + output_name


	frac_power_fig = pt.figure(figsize=(11,9))

	gs = GridSpec(2,3,figure=frac_power_fig)

	ax_list1 = []

	for i in range(3):

		ax1 = frac_power_fig.add_subplot(gs[0,i])
		ax_list1.append(ax1)

	axes_power = ax_list1

	length_scale_summary_ax = frac_power_fig.add_subplot(gs[1,1:])

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

		cell_areas = pd.read_csv(areas_file,sep='\t',header=0,usecols=['Area (pixels)'],dtype='float')

		mean_area = numpy.mean(cell_areas)

		mean_area_um2 = mean_area*.12**2 ###Scale to convert pixel area to area in um^2

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

			num_modes_fit = crossover_mode

			mode_vec = numpy.arange(n_cells)

			kscale = numpy.sum( frac[:num_modes_fit]*mode_vec[:num_modes_fit]/numpy.sum(frac[:num_modes_fit]) )

			length_scale = numpy.sqrt(n_cells/kscale*mean_area_um2)

			length_scale_summary_ax.plot(gene_counter + .1*numpy.random.random()-.05, length_scale, marker='o',alpha=.85,color=color)

			if g in gene_inds_examples:

				ax_pdf = axes_power[axes_counter]

				ax_pdf.loglog(numpy.arange(1,len(smoothed_frac)+.5),smoothed_frac,color=color,linewidth=2,alpha=.85)

				ax_pdf.loglog(numpy.arange(1,len(smoothed_frac)+.5),smoothed_frac_perm,color='grey',linewidth=1,alpha=.5)

				ax_pdf.set_title(paths_filenames.gene_list[g],fontsize=14)

				ax_pdf.set_ylim([10**-5,.16])

				ax_pdf.set_xlabel('Mode number')

				axes_counter += 1

			gene_counter += 1


	axes_power[0].set_ylabel('Fraction of Power',fontsize=14)
	length_scale_summary_ax.set_ylabel(r"Domain Scale ($\mu m$)",fontsize=14)

	length_scale_summary_ax.legend(lines,labels,fontsize=13,loc=[-.6,.15])

	length_scale_summary_ax.set_xticks(range(8))
	length_scale_summary_ax.set_xlim([-.5,7.5])


	length_scale_summary_ax.set_xticklabels(gene_names)


	frac_power_fig.savefig(figname_power,bbox_inches='tight')

def plot_tailmaps_filtered(output_name,data_mode='log',num_modes=50,data_src='Intensities'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)--name for plot
	data_mode (str)--if 'log', data will be log-scaled; otherwise raw values are used
	num_modes (int)--number of modes to project onto (cut-off)
	Returns
	-------
	None (produces plot)
	"""

	###Plot 'long pass filtered' tail map: visualization of projection onto the long eigenmodes, up to a cutoff

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds = [0,2,4]

	for sample in sample_list[:1]:

		print(sample)

		figname = paths_filenames.figure_path + '/tailmaps/' + output_name + '-' + sample + '.pdf'

		fig,axes = pt.subplots(3,2,figsize=(6,12))

		sample_cat = treatment_dict[sample]
		color = color_dict[sample_cat]

		eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy'

		if data_src == 'Counts':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'
		elif data_src == 'Intensities':
			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'
		segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'

		###Import data tables

		data_mat = numpy.load(data_file)
		eigenvector_mat = numpy.load(eigenvector_file)

		if data_mode == 'log':

			data_mat = numpy.log2(data_mat+1)

		n_cells, n_genes = data_mat.shape

		###Center data

		data_mat_centered = data_mat - numpy.outer(numpy.ones((n_cells,),dtype=float),data_mat.mean(axis=0))

		###Decompose

		B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)

		Y_est_long = numpy.matmul(XT[:,:num_modes],B_R[:num_modes,:])

		axes_counter = 0

		for g in gene_inds:

			ax_data = axes[axes_counter,0]
			ax_smoothed = axes[axes_counter,1]

			tailmap_plotting_functions.plot_cell_intensities_diverging_cm(segmentation_file, Y_est_long[:,g]-numpy.min(Y_est_long[:,g]), ax_smoothed, sample)
			tailmap_plotting_functions.plot_cell_intensities_diverging_cm(segmentation_file, data_mat_centered[:,g]-numpy.min(data_mat_centered[:,g]), ax_data, sample)

			ax_data.set_ylabel(paths_filenames.gene_list[g],fontsize=16,rotation=90)

			if axes_counter == 0:

				ax_data.set_title('Data',fontsize=16)
				ax_smoothed.set_title('50 Modes',fontsize=16)

			axes_counter += 1

		pt.savefig(figname,bbox_inches='tight')
		pt.close()


if __name__=='__main__':

	plot_mode_structure_examples(output_name='Fig4A-mode-structure.pdf')

	plot_power_spectra_example_genes(output_name='Fig4B-power-spectra-examples-intensities.pdf',mode='Log',data_src='Intensities')
	plot_tailmaps_filtered(output_name='Fig3C-intensities.pdf',data_mode='log',num_modes=50,data_src='Intensities')

	plot_power_spectra(output_name='FigS8-power-spectra-',spectrum_mode='pdf',data_src='Intensities')
	plot_power_spectra(output_name='FigS9-power-spectra-',spectrum_mode='pdf',data_src='Counts')

	plot_power_spectra(output_name='FigS11-power-spectra-',plotting_mode='linear',spectrum_mode='cdf',data_src='Intensities')

	plot_tailmaps_filtered(output_name='FigS10-counts.pdf',data_mode='log',num_modes=50,data_src='Counts')




