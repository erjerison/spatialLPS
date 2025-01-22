import numpy
import matplotlib.pylab as pt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from scipy.stats import spearmanr, pearsonr
import random

def plot_fig1(mode='Counts',save=True):

	"""Wrapper for plotting Figure 1B and Figure SI 1B

		Inputs
		----------
		mode (str): 'Counts' or 'Intensities' -- raw data is either spot count or intensity/cell
		save (Bool): if True (default), saves figure

		Returns
		-------
		None; saves figure
	"""

	
	###Set up figure axes

	legend_labels = ['Vehicle Control',r'20 $ng/\mu l$',r'25 $ng/\mu l$',r'27.5 $ng/\mu l$',r'30 $ng/\mu l$']
	gene_order = [0,4,5,6,3,8,2,1]

	dist_summary_fig,axes = pt.subplots(3,3,figsize=(12,12))
	inset_ax_list = []
	ncol = 3

	for g in range(8):
		if g < 1.5:
			my_ax = axes[g//ncol,g%ncol]
		else:
			my_ax = axes[(g+1)//ncol,(g+1)%ncol]
		ax_ins = inset_axes(my_ax,width="60%",height="40%",loc='upper right')
		inset_ax_list.append(ax_ins)

	###Get sample keys

	sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='10 hr Timepoint')
	conc_dict = file_import_utilities.import_conc_key(paths_filenames.conc_key)
	color_dict = file_import_utilities.import_color_key(paths_filenames.color_key)
	treatment_dict = file_import_utilities.import_treatment_dict(paths_filenames.sample_key)

	###Get data and plot

	color_list_leg = []

	###During the loop, record concentration vs. average expression for each gene

	concentrations_by_sample = []
	exp_mean_by_gene_sample = []

	for sample in sample_list:

		print(sample)

		sample_cat = treatment_dict[sample]

		concentrations_by_sample.append( conc_dict[sample_cat] )

		if mode=='Counts':

			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-raw_counts.npy'

		elif mode=='Intensity':

			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'

		else:

			print('error specifying mode. mode should be either Counts or Intensity')
			break

		data_mat = numpy.load(data_file)

		n_cells, n_genes = data_mat.shape

		mean_exp_by_gene = []

		for g in range(8):

			###Plot a histogram of the counts distribution (10 equally spaced bins up to the 98th percentile),
			###Unless the 98th percentile occurs at fewer than 9 counts, in which case the bins are assigned to be integer numbers of counts up to the 98th percentile

			gene_ind = gene_order[g]
			data_vec = data_mat[:,gene_ind]
			bin_low = numpy.percentile(data_vec,0)
			bin_high = numpy.percentile(data_vec,98)

			if bin_high - bin_low > 8.5:

				bins = numpy.arange(bin_low,bin_high+.01,(bin_high-bin_low)/10.)
			else:
				bins = numpy.arange(0,bin_high+.01,1)

			hist,nbins = numpy.histogram(data_vec,bins=bins,density=False)
			xlocs = (bins[:-1]+bins[1:])/2. ###Histogram will be plotted vs. bin center locations

			###Determine confidence intervals based on a bootstrap over cell regions.
			###(Note that we are fixing the bin locations)

			random.seed(a=1)

			hist_bs_list = []

			for bs_iter in range(1000):

				data_vec_bs = random.choices(data_vec,k=len(data_vec)) ##re-sample with replacement

				hist_bs,nbins = numpy.histogram(data_vec_bs,bins=bins,density=False) ##re-calculate histogram

				hist_bs_list.append(hist_bs)

			hist_bs_arr = numpy.array(hist_bs_list)

			lbs,ubs = numpy.percentile(hist_bs_arr,q=[2.5,97.5],axis=0)
			yerr_lower = hist - lbs ###input for matplotlib.errorbar: shape(2, N): Separate - and + values for each bar. First row contains the lower errors, the second row contains the upper errors.
			yerr_upper = ubs - hist

			if g < 1.5:
				my_ax = axes[g//ncol,g%ncol]
			else:
				my_ax = axes[(g+1)//ncol,(g+1)%ncol]

			##Plot the fraction of cell regions that falls into each bin (note that error bars then also are scaled by total # of cell regions in histogram)

			ncells_hist = numpy.sum(hist)

			my_ax.errorbar(xlocs,hist/ncells_hist,yerr=[yerr_lower/ncells_hist,yerr_upper/ncells_hist],marker='o',markersize=3,linewidth=2,elinewidth=1,color=color_dict[sample_cat])

			my_ax.set_yscale('log')
			my_ax.set_ylim([10**-2.5,95])
			my_ax.set_title(paths_filenames.gene_list[gene_ind],fontsize=15)

			my_ax.tick_params(axis='both', which='major', labelsize=12)
			my_ax.tick_params(axis='both', which='minor', labelsize=8)

			inset_ax_list[g].semilogy(conc_dict[sample_cat],numpy.mean(data_vec),'o',color=color_dict[sample_cat],markersize=6,alpha=.8)

			mean_exp_by_gene.append( numpy.mean(data_vec) )

			inset_ax_list[g].set_xlabel('LPS conc.',fontsize=13)
			if mode == 'Counts':
				inset_ax_list[g].set_ylabel('Mean counts',fontsize=13)
			elif mode == 'Intensity':
				inset_ax_list[g].set_ylabel('Mean int.',fontsize=13)
			color_list_leg.append(color_dict[sample_cat])

		exp_mean_by_gene_sample.append(mean_exp_by_gene)


	if mode == 'Counts':
		for i in range(3):
			axes[i,0].set_ylabel('Fraction of cells',fontsize=15)
			axes[-1,i].set_xlabel('Counts',fontsize=15)
	if mode == 'Intensity':
		for i in range(3):
			axes[i,0].set_ylabel('Fraction of cells',fontsize=15)
			axes[-1,i].set_xlabel('Intensity',fontsize=15)



	###Generate artists for the legend
	lines=[]
	labels = []
	for sample_cat in conc_dict:

		if '4hr' not in sample_cat:
			color = color_dict[sample_cat]
			conc = conc_dict[sample_cat]
			label = str(conc) + r" $\mu$g/mL"
			line=Line2D([1,2,3],[4,5,6],color=color,linewidth=2)
			lines.append(line)
			labels.append(label)


	concentrations_by_sample = numpy.array(concentrations_by_sample)
	exp_mean_by_gene_sample = numpy.array(exp_mean_by_gene_sample)

	###For each gene, run a permutation test for significance of the positive correlation between concentration and mean expression
	###Note that we are calculating an approximate p value over niter (5000) random permutations, and reporting p values rounded to the nearest .001 (or p<.001 for smaller values)

	niter = 5000

	random.seed(a=2)

	for g in range(8):

		mean_exps = exp_mean_by_gene_sample[:,g]

		r,p = spearmanr(concentrations_by_sample,mean_exps)

		rstat_perm = []

		for n in range(niter):

			rn,pn = spearmanr(concentrations_by_sample,numpy.random.permutation(mean_exps))
			rstat_perm.append(rn)

		rstat_perm = numpy.array(rstat_perm)

		pperm = numpy.count_nonzero( rstat_perm >= r - 1e-14)/float(niter) ##The offset is introduced in case there are finite floating point precision issues

		rround = round(r,2)
		print(rround)

		if pperm < .001:

			inset_ax_list[g].text(.2,.6,r"$r=$"+str(rround)+"\n"+r"$p<.001$",transform = inset_ax_list[g].transAxes,fontsize=13)
		else:
			inset_ax_list[g].text(.2,.6,r"$r=$"+str(rround)+"\n"+r"$p=$"+str(pperm),transform = inset_ax_list[g].transAxes,fontsize=13)

	
	axes[0,2].text(.33,.84,'LPS Conc.',transform = axes[0,2].transAxes,fontsize=14)

	axes[0,2].legend(lines,labels,fontsize=15,loc='center')

	for sloc in ['right','top','bottom','left']:
		axes[0,2].spines[sloc].set_visible(False)

	axes[0,2].set_xticks([])
	axes[0,2].set_yticks([])

	axes[0,2].set_xticklabels([])
	axes[0,2].set_yticklabels([])

	figure_filename = paths_filenames.figure_path + '/' + 'Figure1B-dosage-' + mode + '.pdf'

	if save:
		pt.savefig(figure_filename,bbox_inches='tight')
	else:
		pt.show()

if __name__=='__main__':

	plot_fig1(mode='Intensity',save=True)
	plot_fig1(mode='Counts',save=True)


