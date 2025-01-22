import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import matplotlib.pylab as pt
import seaborn as sns
import numpy
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp

data_path1 = '/Users/ejerison/Dropbox/zfish_lps_phases/single_cell/seq_11242021_output/Control/'
input_path_ctrl = data_path1 + 'Control_counts_filtered.h5ad.gz'

data_path2 = '/Users/ejerison/Dropbox/zfish_lps_phases/single_cell/seq_11242021_output/LPS/'
input_path_lps = data_path2 + 'LPS_counts_filtered.h5ad.gz'

data_path = '/Users/ejerison/Dropbox/zfish_lps_phases/single_cell/seq_11242021_output/'

###Perform leiden clustering on the control and LPS treatments jointly; we will subsequently assign corresponding clusters as needed

adata_lps = sc.read_h5ad(input_path_lps)
adata_ctrl = sc.read_h5ad(input_path_ctrl)

####Concatenate adatas

adata = anndata.AnnData.concatenate(adata_ctrl,adata_lps,batch_key='treatment',batch_categories=['Control','LPS'])

####Compute clusters

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
#sc.pl.highly_variable_genes(adata)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)

sc.tl.umap(adata)
sc.tl.leiden(adata,resolution=2)

####

#sc.pl.umap(adata,color='treatment',save='combined_decontx.pdf')
#sc.pl.umap(adata,color='leiden',save='combined_decontx_leiden2.pdf')

####

clusters = list(set(adata.obs['leiden']))

group_dict = {}


####Filter as presumptive false positives all genes that show up as differentially expressed between the red blood cells in the two treatments

group_dict['rbc'] = {}
group_dict['rbc']['Control'] = ['16']
group_dict['rbc']['LPS'] = ['20']

control_cluster_list = group_dict['rbc']['Control']
lps_cluster_list = group_dict['rbc']['LPS']

lps_clust = adata[ numpy.logical_and(adata.obs['leiden'].isin(lps_cluster_list),adata.obs['treatment']=='LPS'),: ]

ctrl_clust = adata[ numpy.logical_and(adata.obs['leiden'].isin(control_cluster_list),adata.obs['treatment']=='Control'),:]

lps_expressed = lps_clust.var_names[ (lps_clust.X.toarray().sum(axis=0) > .01) ]

ctrl_expressed = ctrl_clust.var_names[ (ctrl_clust.X.toarray().sum(axis=0) > .01) ]

all_expressed = list( set(lps_expressed).union(set(ctrl_expressed)) )

lps_df = sc.get.obs_df(lps_clust, [*all_expressed])
ctrl_df = sc.get.obs_df(ctrl_clust, [*all_expressed])

mw_ps = []
gene_blacklist = []
for gene in all_expressed:
	
	res,p = mannwhitneyu(lps_df[gene],ctrl_df[gene])
	if p < .5*10**-4:
		gene_blacklist.append(gene)

gene_blacklist = set(gene_blacklist)

group_dict['basal_epithelial'] = {}
group_dict['basal_epithelial']['Control'] = ['15','24','39']
group_dict['basal_epithelial']['LPS'] = ['15','24','39']

for cluster in clusters:

	group_dict[cluster] = {}
	group_dict[cluster]['Control'] = [cluster]
	group_dict[cluster]['LPS'] = [cluster]

####

for group in ['15','21','24','39','basal_epithelial']:

	filename = 'figures/DE_genes_group_' + group + '.txt'
	file = open(filename,'w')
	file.write( ('\t').join(('gene','90th percentile expression, LPS (cptt)','90th percentile expression, control (cptt)')) + '\n')
	control_cluster_list = group_dict[group]['Control']
	lps_cluster_list = group_dict[group]['LPS']

	lps_clust = adata[ numpy.logical_and(adata.obs['leiden'].isin(lps_cluster_list),adata.obs['treatment']=='LPS'),: ]

	ctrl_clust = adata[ numpy.logical_and(adata.obs['leiden'].isin(control_cluster_list),adata.obs['treatment']=='Control'),:]

	lps_expressed = lps_clust.var_names[ (lps_clust.X.toarray().sum(axis=0) > .01) ]

	ctrl_expressed = ctrl_clust.var_names[ (ctrl_clust.X.toarray().sum(axis=0) > .01) ]

	all_expressed = list( set(lps_expressed).union(set(ctrl_expressed)) )

	lps_df = sc.get.obs_df(lps_clust, [*all_expressed])
	ctrl_df = sc.get.obs_df(ctrl_clust, [*all_expressed])

	if len(lps_df.index) > 10 and len(ctrl_df.index) > 10: ###At least 10 LPS and control cells in this cluster

		
		mean_lps = pd.Series(numpy.percentile(lps_df,90,axis=0),index=lps_df.columns)
		mean_ctrl = pd.Series(numpy.percentile(ctrl_df,90,axis=0),index=lps_df.columns)

		de = mean_lps - mean_ctrl

		mw_ps = []
		gene_list_filtered = []
		de = []
		for gene in all_expressed:
			if gene not in gene_blacklist:
				res,p = mannwhitneyu(lps_df[gene],ctrl_df[gene])
				mw_ps.append(p)
				gene_list_filtered.append(gene)
				de.append(mean_lps[gene]-mean_ctrl[gene])

		mw_ps = numpy.array(mw_ps)
		de = numpy.array(de)
		print(mw_ps)
		pt.figure()
		ax = pt.gca()
		ax.set_yscale('log')
		pt.scatter( de, 1/mw_ps, alpha = .8 )
		pt.title(group)
		ax.set_xlabel('ln-fold change, 90th percentile')
		ax.set_ylabel('Mann-Whitney score')

		de2 = numpy.abs(de)
		de_sort_order = numpy.argsort(de2)[::-1]

		print(de_sort_order)

		for ng in range(50):
			gene = gene_list_filtered[de_sort_order[ng]]
			if p < .5*10**4:
				p = mw_ps[de_sort_order[ng]]
				ax.text( mean_lps[gene] - mean_ctrl[gene], 1/p, gene,fontsize=8)
				file.write(('\t').join((gene,str(mean_lps[gene]), str(mean_ctrl[gene])))+'\n')
		file.close()

		pt.savefig('figures/DE_rbcfilt_cluster_'+str(group)+'.pdf')