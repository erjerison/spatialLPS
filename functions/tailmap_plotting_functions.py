import matplotlib.pylab as pt
from skimage.io import imread
import numpy

def plot_cell_intensities_diverging_cm(label_image_filename, component_arr, axis_obj, sample, cm='coolwarm',alpha_bg=.2):

	"""

	Plots heatmap visualization of tail samples

	Inputs
	------
	label_image_filename (str): path to an array with the same dimensions as the image, containing integer entries that correspond to labeled cell regions.
	component_arr (numpy array or list of length n_cells): scalar values on each cell region, to be plotted. The first entry corresponds to cell region 0, and so forth.
	axis_obj (matplotlib axis): the axis to plot on
	sample (str): sample name
	cm (str), optional: colormap
	alpha_bg (float, 0-1), optional: the background (areas of the image without labeled cell regions) will be set to this opacity

	Returns
	-------
	None
	
	"""

	label_image = numpy.load(label_image_filename).astype('int')

	nx,ny = label_image.shape

	intensity_image1 = numpy.zeros((nx,ny))

	alpha_image = numpy.zeros((nx,ny))

	background_image = numpy.zeros((nx,ny))

	for i in numpy.arange(len(component_arr))+1:

		###Note that we are re-indexing because entry 0 in the array of values corresponds to region index 1 in the label image (0 is used to label the background)

		intensity_image1[label_image==i] = component_arr[i-1]

		alpha_image[label_image==i] = 1

	###Fix background color

	background_image[label_image==0] = 1

	axis_obj.imshow(intensity_image1,alpha=alpha_image,cmap=cm,vmin=numpy.percentile(component_arr,1),vmax=numpy.percentile(component_arr,99),zorder=2)
	axis_obj.imshow(background_image,cmap='YlOrBr',alpha=alpha_bg,zorder=1)
	axis_obj.set_xticks([])
	axis_obj.set_yticks([])