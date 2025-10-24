import numpy as np
import matplotlib.pylab as pt
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.harmonic_mapping_functions import harmonic_mapping
from matplotlib.gridspec import GridSpec

sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation and Pass Morphology Filter')

for sample in sample_list[:1]:

    print(sample)

    mask_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-mask.tif'
    centroid_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'
    fig = pt.figure(figsize=(4,12))
    ax_list1 = []
    gs = GridSpec(2,1,figure=fig)

    for i in range(2):
        ax1 = fig.add_subplot(gs[i,0])
        ax_list1.append(ax1)
    print(ax_list1)
    uv_centroids = harmonic_mapping(mask_file,centroid_file, plot=True, axis_list=ax_list1)
    fig.savefig('harmonic_mapping.pdf',bbox_inches='tight')
    np.save( paths_filenames.table_path + '/' + sample + '/' + sample + '-uv_centroids.npy', uv_centroids)