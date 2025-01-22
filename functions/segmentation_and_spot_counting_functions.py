from skimage.io import imread
from skimage.segmentation import watershed, expand_labels, find_boundaries, join_segmentations
from skimage.measure import label, regionprops
import matplotlib.pylab as pt
import numpy
from aicsimageio import AICSImage
from skimage.filters import sobel, gaussian
from skimage.feature import canny, blob_log, blob_doh, blob_dog, peak_local_max
from skimage.morphology import disk
from skimage import img_as_float, img_as_int,restoration
from skimage.transform import downscale_local_mean
from matplotlib.patches import Ellipse, Circle
from scipy.ndimage import convolve

def segment_tissue(dapi_input_file,mask_file,output_filename,figure_output_name):

	"""

	Segment based on DAPI image:
	1. Bin dapi image for smoothing and to speed up computation.
	2. convolve DAPI image with a disk, chosen to be slightly smaller than a large nucleus.
	3. Detect peaks in the convolution image
	4. Expand labels from each of these peaks, to create non-overlapping regions. Create a label image for the segmentation.
	5. Save the label image

	Inputs
	------
	dapi_input_file (str): path to .tiff file, stained nuclei
	mask_file (str): path to .tiff file, mask
	output_filename (str): path to save array corresponding to image segmentation
	figure_output_name (str): path to output figure

	Returns
	-------
	None (saves .npy array with labeled cell regions and a figure overlaying segmentation on dapi image)
	"""

	dapi = imread(dapi_input_file,plugin='tifffile')
	mask = imread(mask_file,plugin='tifffile').astype('bool')

	dapi_masked = dapi.copy()
	dapi_masked[~mask] = 0

	dapi_smoothed = gaussian(dapi_masked,5)

	struct_elem = disk(35)

	print('Convolving')
	convolution = convolve(dapi_smoothed,struct_elem)

	print('Finding peaks')
	convolution_peak_image = peak_local_max(convolution,indices=False)
	labeled_peaks = label(convolution_peak_image)

	cell_segmentation = expand_labels(labeled_peaks,distance=100)
	segmentation_boundaries = find_boundaries(cell_segmentation)

	pt.imshow(dapi_smoothed + numpy.max(dapi_smoothed)*segmentation_boundaries,vmax=numpy.percentile(dapi_smoothed,99),cmap='Greys_r')
	pt.savefig(figure_output_name)
	numpy.save(output_filename,cell_segmentation)

def measure_cell_stats(images_path,tables_path,sample,background_dict,channel_list=['0','1','2','4','5','6','8','9','10']):

	"""

	Measure:
	1. Cell centroid location
	2. Summed pixel intensities for pixels above background within the cell
	3. Number of called spots
	4. Summed intensity within called spots

	Inputs
	------
	image_file (.tif format, uint16)
	cell_label_file (.npy binary format)
	background_threshold (integer)

	Returns
	-------
	None (saves)

	Outputs:
	1. .csv file with all measurements listed above
	2. .npy binary with spot counts for each called cell region and each gene (N segmented cells x n genes)
	
	"""

	###First import the label file and get centroids and areas for each cell

	cell_label_file = tables_path + '/' + sample + '/' + sample + '-segmentation.npy'

	labels = numpy.load(cell_label_file)
	region_properties = regionprops(label_image = labels)
	cell_dict = {}

	for cell in region_properties:

		cell_dict[cell.label] = {}

		cell_dict[cell.label]['centroid'] = cell.centroid
		cell_dict[cell.label]['area'] = cell.area

	###For each channel, measure pixel intensity per cell

	for channel in channel_list:

		print(channel)

		image_file = images_path + '/' + sample + '/' + sample + '-channel-' + channel + '-registered.-maxz.tif'

		image = imread(image_file,plugin='tifffile')

		background_threshold = background_dict[channel]

		image[image < background_threshold] = 0

		region_properties = regionprops(label_image = labels, intensity_image = image)

		###Detect spots

		spots = peak_local_max( image, threshold_abs=background_threshold+200,indices=False)
		larger_spots = expand_labels(spots,distance=3)

		spot_intensities = image*larger_spots

		spot_count_properties = regionprops(label_image = labels, intensity_image = spots)
		spot_intensity_properties = regionprops(label_image = labels, intensity_image = spot_intensities)

		for cell in region_properties:

			cell_dict[cell.label][channel + '-intensity'] = cell.mean_intensity

		for cell in spot_count_properties:
			cell_dict[cell.label][channel + '-nspots'] = cell.weighted_moments[0,0]

		for cell in spot_intensity_properties:
			cell_dict[cell.label][channel + '-spot_intensity'] = cell.weighted_moments[0,0]

	###Record results

	csv_output_file = tables_path + '/' + sample + '/' + sample + '-cell_stats.txt'

	file = open(csv_output_file,'w')
	file.write( 'Label'+'\t'+'Centroid' + '\t' + 'Area (pixels)' + '\t' + ('\t').join( [channel + ' pixel intensity' for channel in channel_list] ) + '\t' + ('\t').join( [channel + ' spot number' for channel in channel_list] ) + '\t' + ('\t').join( [channel + ' spot intensity' for channel in channel_list] ) + '\n')

	counts_arr = []
	intensity_arr = []
	centroid_arr = []

	for label in cell_dict:

		cell = cell_dict[label]
		r,c = cell['centroid']
		file.write( (',').join([str(label) + '\t' + str(r),str(c)]) + '\t' + str(cell['area']) + '\t' + ('\t').join( [str(cell[channel +'-intensity']) for channel in channel_list] ) + '\t' +
			('\t').join( [str(cell[channel +'-nspots']) for channel in channel_list] ) + '\t' +
			('\t').join( [str(cell[channel +'-spot_intensity']) for channel in channel_list] ) + '\n' )

		counts_arr.append( [cell[channel +'-nspots'] for channel in channel_list] )

		intensity_arr.append( [cell[channel +'-intensity'] for channel in channel_list] )

		centroid_arr.append( [r,c] )

	file.close()

	counts_arr = numpy.array(counts_arr)

	intensity_arr = numpy.array(intensity_arr)

	centroid_arr = numpy.array(centroid_arr)

	numpy.save(tables_path + '/' + sample + '/' + sample + '-raw_counts.npy',counts_arr)

	numpy.save(tables_path + '/' + sample + '/' + sample + '-intensities.npy',intensity_arr)

	numpy.save(tables_path + '/' + sample + '/' + sample + '-xy_centroids.npy',centroid_arr)
