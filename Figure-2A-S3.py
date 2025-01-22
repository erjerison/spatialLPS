import numpy
import matplotlib.pylab as pt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from scipy.stats import spearmanr
from scipy.stats import binom
from scipy.stats.sampling import DiscreteAliasUrn

def null_model(total_exp,n_cells,nmin=1,nmax=3):

	"""
	Inputs
	----------
	total_exp (int or float)--total spot count for the gene channel and sample
	n_cells (int or float)--number of segmented cell regions in sample
	nmin (int or float, optional)--lower bound for the number of true cells per segmented cell region in the model
	nmax (int or float, optional)--upper bound for the number of true cells per segmented cell region in the model

	Returns
	-------
	bpmf: probability vector for the estimated null distribution, normalized to sum to n_cells (i.e. null model of number of cell regions with a particular spot count)
	xvec: support for bpmf (i.e. the range of values on which bpmf is defined)
	-------

	The null model represents the number of spots per cell under the assumption that
	the spots are equally likely to appear in any cell region, in proportion to the 
	number of cells in that region. (In the case of perfect segmentation this would always be 1)
	Since the total number of spots is large, we can model this as a binomial distribution integrated against the
	probability distribution of cells/region. As a first conservative estimate, we take this distribution
	to be uniform on [1,3].

	The probability of observing k spots in a region with c cells is Binom(N,p), where N is the total number
	of spots in the sample, and p=1/n_regions*c/<c>
	<c> = \int c P(c) dc, where P(c) is the pdf of the number of cells per region, is a normalization constant fixed by the requirement that n_regions*<k> = N

	The probability of observing k spots is then:

	P(k) = \int dc P(c) P(k|c) = \int dc P(c) Binom(N,1/n_regions*c/<c>)

	We are estimating the integral by a sum

	"""

	mean_n = (nmax+nmin)/2.

	pmin = 1./n_cells*nmin/mean_n
	pmax = 1./n_cells*nmax/mean_n

	xvec = numpy.arange( 0, binom.ppf(.999,total_exp,pmax) ) ###We are estimating the pdf on the range from 0 to the 99.9th percentile of the distribution associated with the highest p 

	nvec_for_sum = numpy.arange( nmin, nmax+.001, (nmax-nmin)/500 )

	###Initialize 

	b_pmf = binom.pmf(xvec, total_exp,pmin)

	###Sum to approximate integral

	for n in nvec_for_sum:

		p_n = 1./n_cells*n/mean_n

		b_pmf += binom.pmf(xvec, total_exp,p_n)

	###Normalize

	b_pmf = b_pmf/numpy.sum(b_pmf)*n_cells

	return b_pmf, xvec

def plot_width_ratio_summary_with_inset(output_name='Figure2A_width_ratios-check.pdf',nmin=.5,nmax=2.5):

	"""
	Plot ratio of standard deviation of data and null distributions, for all genes/samples.
	Inset shows example plot for count distribution and null distribution for a sample from the middle of the range for il1b.
	
	Inputs
	------
	output_name (str)--name for output plot (extension specifies format for saving)
	nmin (int or float, optional)--lower bound for the number of true cells per segmented cell region in the model
	nmax (int or float, optional)--upper bound for the number of true cells per segmented cell region in the model

	Returns
	None returned; saves figure
	------
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds = [0,1,2,3,4,5,6,8]

	gene_names = [paths_filenames.gene_list[g] for g in gene_inds]

	sample_cat_to_ax = {'LPS25ugml_10hrs':0, 'LPS27ugml_10hrs':1,'LPS30ugml_4hrs':2,'LPS30ugml_10hrs': 3}
	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu g/mL$ 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}

	fig1 = pt.figure(figsize=(12,4))
	ax = pt.gca()
	ax_inset = ax.inset_axes(bounds=[0,5,1.5,2.1],transform=ax.transData)
	ax_inset.spines[['right', 'top']].set_visible(False)

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]

		ax_ind = sample_cat_to_ax[sample_cat]

		data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'

		data_mat = numpy.load(data_file)

		n_cells, n_genes = data_mat.shape

		xloc_counter = 0
		for g in gene_inds:

			data_vec = data_mat[:,g]

			data_std = numpy.std(data_vec)

			total_exp = numpy.sum(data_vec)

			null_std_list = []

			b_pmf,xvec = null_model(total_exp,n_cells,nmin,nmax)

			dx = xvec[1] - xvec[0]
			b_pmf_norm = numpy.sum(dx*b_pmf)

			model_var = numpy.sum( dx*xvec**2*b_pmf/b_pmf_norm ) - ( numpy.sum(dx*xvec*b_pmf/b_pmf_norm) )**2

			model_std = numpy.sqrt(model_var)

			###Sanity check: measure standard deviation instead by sampling from b_pmf as a probability vector

			urng = numpy.random.default_rng()
			rng = DiscreteAliasUrn(b_pmf/b_pmf_norm, random_state=urng)
			rvs = rng.rvs(size=1000)


			null_std_list.append(model_std)

			color = color_dict[sample_cat]

			xjitter = xloc_counter+.1*(numpy.random.random()-.5)

			ax.plot(xjitter, data_std/model_std, 'o', color=color, alpha=.7)

			###Plot an example of an expression count distribution and null

			if sample == 'd-08092022_LPS30ugml_4hrs_tail4' and g==0:

				ax.plot(xjitter, data_std/model_std, 'o', color=color, markeredgecolor='k',markeredgewidth='2')

				###Plot a histogram of the counts distribution (10 equally spaced bins up to the 98th percentile)

				bin_low = numpy.percentile(data_vec,0)
				bin_high = numpy.percentile(data_vec,98)
				bins = numpy.arange(bin_low,bin_high+.01,(bin_high-bin_low)/10)

				hist,nbins = numpy.histogram(data_vec,bins=bins)
				xlocs = (bins[:-1]+bins[1:])/2.

				###Note that we are setting the scale on the y-axis such that the maximum value is 1 for each curve and/or null distribution.

				ax_inset.semilogy(xlocs,hist/numpy.max(hist),color='C2')
				ax_inset.bar(xvec,b_pmf/numpy.max(b_pmf),alpha=.3,width=xvec[1]-xvec[0],color='C2')

				ax_inset.set_ylim([.005,1.3])
				ax_inset.set_xlabel('Counts')

				xpoint = xjitter
				ypoint = data_std/model_std


			xloc_counter += 1

	ax.plot([-.5,7.5],[1,1],'k--')

	ax.plot([xpoint,0],[ypoint,5],'k--',linewidth=.7,zorder=0,alpha=.4)
	ax.plot([xpoint,1.5],[ypoint,5],'k--',linewidth=.7,zorder=0,alpha=.4)

	ax.set_xticks(range(8))
	ax.set_xlim([-.5,7.5])


	ax.set_ylabel(r'$\frac{\sigma_{Data}}{\sigma_{Null}}$',fontsize=20,rotation=90)

	ax.set_xticklabels(gene_names,fontsize=14)

	###Make dummy plot to generate artists for the legend
	lines=[]
	labels = []
	for sample_cat in sample_cat_to_ax.keys():

		color = color_dict[sample_cat]
		conc = conc_dict[sample_cat]
		label = sample_cat_to_label[sample_cat]
		marker=Circle(xy=1,color=color,alpha=.7)
		lines.append(marker)
		labels.append(label)

	ax.legend(lines,labels,loc='upper center')

	pt.savefig(output_filename)

def plot_width_ratios(output_name='FigureS3_width_ratios.pdf',nrange_list = [ [.5,1], [.5,1.5],[.5,2],[.5,3],[.5,4],[.5,5] ]):

	"""
	Plot ratio of standard deviation of data and null distribution for different choices of the distribution of cells per segmented cell region
	The distribution of true cells per segmented region is taken to be uniform over [lower bound,upper bound] for the bounds specified in nrange_list

	Inputs:
	----------
	output_name (str)--name for output plot (extension specifies format for saving)
	nrange_list (list, optional)--ordered [lower bound, upper bound] pairs for the number of true cells per segmented cell region in the model

	Returns:
	None returned; saves figure
	----------
	"""

	output_filename = paths_filenames.figure_path + '/' + output_name

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###
	gene_inds = [0,1,2,3,4,5,6,8]

	sample_cat_to_ax = {'LPS25ugml_10hrs':0, 'LPS27ugml_10hrs':1,'LPS30ugml_4hrs':2,'LPS30ugml_10hrs': 3}
	sample_cat_to_label = {'LPS25ugml_10hrs':r"$25\,\mu g/mL$ 10 hrs", 'LPS27ugml_10hrs':r'$27.5\,\mu g/mL$ 10 hrs','LPS30ugml_4hrs':r'$30\,\mu g/mL$ 4 hrs','LPS30ugml_10hrs': r'$30\,\mu g/mL$ 10 hrs'}

	n_models = len(nrange_list)

	fig1,axes = pt.subplots(2,4,figsize=(2*n_models,8),sharex=True)

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]

		data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'

		data_mat = numpy.load(data_file)

		n_cells, n_genes = data_mat.shape

		mean_exp_by_gene = []

		ax_counter = 0
		for g in gene_inds:

			data_vec = data_mat[:,g]

			data_std = numpy.std(data_vec)

			total_exp = numpy.sum(data_vec)

			ratio_list = []

			ax = axes[ax_counter//4,ax_counter%4]

			nranges = []

			for model_ind in range(len(nrange_list)):

				nmin = nrange_list[model_ind][0]
				nmax = nrange_list[model_ind][1]

				b_pmf,xvec = null_model(total_exp,n_cells,nmin,nmax)

				dx = xvec[1] - xvec[0]
				b_pmf_norm = numpy.sum(dx*b_pmf)

				model_var = numpy.sum( dx*xvec**2*b_pmf/b_pmf_norm ) - ( numpy.sum(dx*xvec*b_pmf/b_pmf_norm) )**2

				model_std = numpy.sqrt(model_var)

				ratio_list.append(data_std/model_std)
				nranges.append(nmax-nmin)

				color = color_dict[sample_cat]

			ax.plot( numpy.array(nranges), numpy.array(ratio_list),color=color)
			ax.set_title(paths_filenames.gene_list[g])

			if g > 3.5:
				ax.set_xlabel('Null width',fontsize=14)


			if g==0 or g==4:
				ax.set_ylabel(r'$\frac{\sigma_{Data}}{\sigma_{Null}}$',fontsize=20)
			ax_counter += 1

	###Make dummy plot to generate artists for the legend
	lines=[]
	labels = []
	for sample_cat in sample_cat_to_ax.keys():

		color = color_dict[sample_cat]
		conc = conc_dict[sample_cat]
		label = sample_cat_to_label[sample_cat]
		marker=Circle(xy=1,color=color,alpha=.7)
		lines.append(marker)
		labels.append(label)

	ax = axes[0,0]
	ax.legend(lines,labels,loc='upper center')

	pt.savefig(output_filename)

if __name__=='__main__':

	plot_width_ratio_summary_with_inset()

	plot_width_ratios()
