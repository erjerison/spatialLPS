import numpy as np
import pandas as pd
import scanpy as sc
import anndata

data_path1 = '/Users/ejerison/Dropbox/Postdoc/zfish_lps_phases/single_cell/seq_11242021_output/Control/'
input_path_ctrl = data_path1 + 'Control_counts_filtered.h5ad.gz'

data_path2 = '/Users/ejerison/Dropbox/Postdoc/zfish_lps_phases/single_cell/seq_11242021_output/LPS/'
input_path_lps = data_path2 + 'LPS_counts_filtered.h5ad.gz'

data_path = '/Users/ejerison/Dropbox/Postdoc/zfish_lps_phases/single_cell/seq_11242021_output/'

results_subset = data_path + 'Combined_clusters_epithelial_decont.h5ad'
results_subset2 = data_path + 'Combined_clusters_periderm_decont.h5ad'


####
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')
sc._settings.ScanpyConfig(cachedir=data_path + 'cache/',figdir=data_path + 'figures/')


####

adata_ctrl = sc.read_h5ad(input_path_ctrl)
adata_lps = sc.read_h5ad(input_path_lps)

####Concatenate adatas

adata = anndata.AnnData.concatenate(adata_ctrl,adata_lps,batch_key='treatment',batch_categories=['Control','LPS'])
print(adata.X)
####

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)

sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=50)

sc.tl.umap(adata)

sc.tl.leiden(adata)

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
#sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(40)
marker_df.to_csv(data_path + 'marker_genes_leiden_clusters_decont.tsv',sep='\t')

adata_sub = adata[adata.obs['leiden'].isin(['6']),:]

adata_sub.write(results_subset)

adata_sub2 = adata[adata.obs['leiden'].isin(['4']),:]

adata_sub2.write(results_subset2)


sc.tl.rank_genes_groups(adata,'leiden',groups=['0','1'],reference='2',method='wilcoxon',key_added='grp012_comparison')
marker_df = pd.DataFrame(adata.uns['grp012_comparison']['names']).head(40)
marker_df.to_csv(data_path + 'marker_genes_clusters01v2.tsv',sep='\t')

sc.tl.rank_genes_groups(adata,'leiden',groups=['0'],reference='3',method='wilcoxon',key_added='grp03_comparison')
marker_df = pd.DataFrame(adata.uns['grp03_comparison']['names']).head(40)
marker_df.to_csv(data_path + 'marker_genes_clusters0v3.tsv',sep='\t')

sc.tl.rank_genes_groups(adata,'leiden',groups=['2'],reference='3',method='wilcoxon',key_added='grp23_comparison')
marker_df = pd.DataFrame(adata.uns['grp23_comparison']['names']).head(40)
marker_df.to_csv(data_path + 'marker_genes_clusters2v3.tsv',sep='\t')

sc.pl.umap(adata, color=['treatment','leiden','il1b','mpeg1.1','mpx','and2','krt4','krt5','tp63','socs3a','prss1','ctrb1','bpifcl','ptgs2b'],save='combined_conditions_umap1_decont.pdf')

