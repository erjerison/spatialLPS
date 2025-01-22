import numpy
import matplotlib.pylab as pt
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from scipy.interpolate import griddata
from matplotlib.patches import Rectangle

def plot_expression_unfolded(output_name,data_src='Counts'):

	"""
	Inputs
	------
	output_name (str, extension specifies file type)
	data_src (str, 'Counts' or 'Intensities')

	Returns
	-------
	None (plots figure)
	---
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name + '.pdf'

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation and Pass Morphology Filter')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	gene_inds = [0,1,2,3,4,5,6,8]

	#LOOP THROUGH SAMPLES AND CALCULATE MEAN COUNTS PER CELL ACROSS ALL GENES TO DETERMINE FIGURE ORDERING.
	#ALSO COMPUTE OVERALL DATA RANGE TO FIX COLOR SCALE

	mean_counts = []
	all_counts = []

	for sample in sample_list:

		###Import data table and cell spatial locations

		if data_src == 'Counts':

			data_table_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-raw_counts.npy'

		elif data_src == 'Intensities':
			data_table_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-intensities.npy'

		spatial_location_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-st_centroids.npy'

		data_mat = numpy.load(data_table_filename)
		data_mat = numpy.delete(data_mat,[7],axis=1) ###mpeg1.1 (macrophage marker)

		mean_counts.append( numpy.mean(data_mat) )
		all_counts.extend(data_mat.flatten())

	overall_plow = numpy.percentile(all_counts,2)
	overall_phigh = numpy.percentile(all_counts,98)

	print(overall_plow,overall_phigh)

	sample_order = numpy.argsort(mean_counts)

	###Make artists for the treatment legend

	sample_cat_list = ['LPS25ugml_10hrs', 'LPS27ugml_10hrs','LPS30ugml_4hrs','LPS30ugml_10hrs']

	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu g/mL$ 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}


	rects=[]
	labels = []
	for sample_cat in sample_cat_list:

		color = color_dict[sample_cat]
		conc = conc_dict[sample_cat]
		label = sample_cat_to_label[sample_cat]
		rect=Rectangle((0,0),1,1,edgecolor=color,linestyle=':',linewidth=.8,facecolor='White')
		rects.append(rect)
		labels.append(label)

	###Plot

	n_genes = len(gene_inds)
	n_samples = len(sample_list)
	fig,axes = pt.subplots(n_samples+1,n_genes,figsize=(6,6))

	fig.subplots_adjust(left=0.1)
	cbar_ax = fig.add_axes([-0.025, 0.12, 0.017, 0.18])

	###Set up grid for plotting rectangle mapping

	x = numpy.arange(0.,1.001,.02)
	y = numpy.arange(0,1.001,.005)
	Xi,Yi = numpy.meshgrid(x,y)

	###Initialize arrays for computing average expression profile across samples

	gene_exp_mean_arrays = []

	for i in range(len(gene_inds)):

		gene_exp_mean_arrays.append([])

	sample_ind = 0

	for siter in range(n_samples):

		sample_i = sample_order[siter]

		sample = sample_list[sample_i]

		print(sample)

		treatment = treatment_dict[sample]

		color = color_dict[treatment]

		###Import data table and cell spatial locations

		if data_src == 'Counts':

			data_table_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-raw_counts.npy'

		elif data_src == 'Intensities':

			data_table_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-intensities.npy'

		spatial_location_filename = paths_filenames.table_path + '/'+ sample + '/' + sample + '-st_centroids.npy' ##Note that this file contains the coordinates using the Coons patch transformation

		data_mat = numpy.load(data_table_filename)
		loc_mat = numpy.load(spatial_location_filename)

		###Log transform data

		data_mat = numpy.log2(data_mat+1)

		xspan = (max(loc_mat[:,0]) - min(loc_mat[:,0]))
		yspan = (max(loc_mat[:,1]) - min(loc_mat[:,1]))

		axis_counter = 0

		for g in gene_inds:

			grid_img = griddata(loc_mat,data_mat[:,g],(Xi,Yi),method='cubic')

			###Construct average expression profile for this gene across all samples. We are dividiing by standard deviation to weight
			###intermediate and high activation samples more evenly in the average.

			gene_exp_mean_arrays[axis_counter].append(grid_img/numpy.nanstd(grid_img)) ###Note that there are often nan values in the corners of the rectangle

			cimg = axes[sample_ind,axis_counter].pcolormesh(grid_img[::-1,:],cmap='RdBu_r',vmin=numpy.log2(overall_plow+1),vmax=numpy.log2(overall_phigh+1))

			if sample_ind == 0:
				axes[sample_ind,axis_counter].set_title(paths_filenames.gene_list[g],fontsize=9)
				
			axes[sample_ind,axis_counter].set_xticks([])
			axes[sample_ind,axis_counter].set_xticklabels([])
			axes[sample_ind,axis_counter].set_yticks([])
			axes[sample_ind,axis_counter].set_yticklabels([])

			for spine in axes[sample_ind,axis_counter].spines.values():
				spine.set_edgecolor(color)
				spine.set_linewidth(.8)
				spine.set_linestyle(':')

			###Add colorbar to show color scale for expression panels

			if sample_ind == 5 and axis_counter == 0:
				cb = fig.colorbar(cimg, cax=cbar_ax)
				cb.set_label(r'$\log_2{Exp}$',fontsize=6)

				for spine in cbar_ax.spines.values():
					spine.set_visible(False)
				cbar_ax.tick_params(labelsize=6,length=3,width=.7)

			if sample_ind == 11 and axis_counter == 0:

				axes[sample_ind,axis_counter].legend(rects,labels,loc=[-2.3,.2],fontsize=6)

			if sample_ind == 7 and axis_counter == 0:

				axes[sample_ind,axis_counter].set_ylabel('Samples')

			axis_counter += 1

		sample_ind += 1

	###Determine range in mean expression for plotting

	exp_summaries = []

	for i in range(n_genes):

		exp_arrays = numpy.array(gene_exp_mean_arrays[i])
		exp_summary = numpy.nanmean(exp_arrays,axis=0)

		exp_summaries.append(exp_summary[::-1,:])

	exp_all = numpy.array(exp_summaries)

	exp_low = numpy.nanpercentile(exp_all,2)
	exp_high = numpy.nanpercentile(exp_all,98)

	for i in range(n_genes):

		axes[n_samples,i].pcolormesh( exp_summaries[i], cmap='RdBu_r', vmin=exp_low,vmax=exp_high)

		axes[n_samples,i].set_xticks([])
		axes[n_samples,i].set_xticklabels([])
		axes[n_samples,i].set_yticks([])
		axes[n_samples,i].set_yticklabels([])

		if i == 0:

			axes[n_samples,i].set_ylabel('Mean',fontsize=8)

	pt.savefig(output_filename,bbox_inches='tight')

if __name__=='__main__':

	plot_expression_unfolded('FigureS7-rectangle-expression-with-means',data_src='Counts')
	plot_expression_unfolded('Figure3-rectangle-expression-with-means',data_src='Intensities')