import numpy
import matplotlib.pylab as pt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from scipy.stats import spearmanr
import random

###Wrapper for plotting Figure S2B (counts associated with the macrophage marker)

def plot_figS2(mode='Counts',save=True):

	"""
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
	gene_order = [7]

	fig=pt.figure()
	my_ax = pt.gca()

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

		elif mode=='Intensities':

			data_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-intensities.npy'

		else:

			print('error specifying mode. mode should be either Counts or Intensity')
			break

		data_mat = numpy.load(data_file)

		n_cells, n_genes = data_mat.shape

		gene_ind = 7 ###Index for the mpeg1.1 marker
		
		data_vec = data_mat[:,gene_ind]

		my_ax.semilogy(conc_dict[sample_cat],numpy.mean(data_vec),'o',color=color_dict[sample_cat],markersize=6,alpha=.8)

		mean_exp_by_gene = numpy.mean(data_vec)

		my_ax.set_xlabel('LPS conc.',fontsize=13)
		if mode == 'Counts':
			my_ax.set_ylabel('Mean counts',fontsize=13)
		elif mode == 'Intensities':
			my_ax.set_ylabel('Mean intensity',fontsize=13)
		color_list_leg.append(color_dict[sample_cat])

		exp_mean_by_gene_sample.append(mean_exp_by_gene)


	###Make dummy plot to generate artists for the legend
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

	niter = 1000

	random.seed(a=2)

	

	mean_exps = exp_mean_by_gene_sample

	print(mean_exps.shape)

	r,p = spearmanr(concentrations_by_sample,mean_exps)

	rstat_perm = []

	for n in range(niter):

		rn,pn = spearmanr(concentrations_by_sample,numpy.random.permutation(mean_exps))
		rstat_perm.append(rn)

	rstat_perm = numpy.array(rstat_perm)

	pperm = numpy.sum( rstat_perm > r )/float(niter)

	rround = round(r,2)
	print(rround)

	if pperm < .001:

		my_ax.text(.4,.55,r"$r=$"+str(rround)+"\n"+r"$p<.001$",transform = my_ax.transAxes,fontsize=13)
	else:
		my_ax.text(.4,.55,r"$r=$"+str(rround)+"\n"+r"$p=$"+str(pperm),transform = my_ax.transAxes,fontsize=13)

	my_ax.legend(lines,labels,loc='upper center')

	figure_filename = paths_filenames.figure_path + '/' + 'FigureS2B-dosage-mpeg-' + mode + '.pdf'

	if save:
		pt.savefig(figure_filename,bbox_inches='tight')
	else:
		pt.show()

if __name__=='__main__':


	plot_figS2(mode='Intensities',save=True)


