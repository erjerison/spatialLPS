import matplotlib.pylab as pt
import numpy
import pandas as pd

from aicsimageio import AICSImage
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from skimage.transform import warp, warp_polar, rotate, rescale, downscale_local_mean
from skimage.registration import phase_cross_correlation
from skimage import io, feature, registration

from skimage.util import img_as_int, img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from scipy.fft import fft2, fftshift
import gc

###Registration of DAPI images:
###1. Use the cross-correlation between a polar transformation of the ffts of DAPI channel to compute the rotation angle between images
###2. Compute overall translation shift on the rotated DAPI channel
###3. Run a cross-correlation in 3d to register to the nearest slice (micron) in the z dimension

def rotation_angle(dapi1,dapi2,working_dir,sample_handle):

	"""

	Inputs
	------
	dapi1 (.tiff, 2D): First image -- reference for registration
	dapi2 (.tiff, 2D): Second image -- registered relative to first
	working_dir (str): path for output files
	sample_handle (str): preamble to sample name
	
	Returns
	------
	recovered_angle (float): estimated rotation angle of dapi2 relative to dapi1

	"""


	###Input: 2d dapi images.
	###Output: estimate of the overall rotation angle of dapi2 relative to dapi1, in degrees (about the image center)

	# First, band-pass filter both images
	image = difference_of_gaussians(dapi1, 5, 20)
	rts_image = difference_of_gaussians(dapi2, 5, 20)

	# window images
	wimage = image * window('hann', image.shape)
	rts_wimage = rts_image * window('hann', image.shape)

	# work with shifted FFT magnitudes
	image_fs = numpy.abs(fftshift(fft2(wimage)))
	rts_fs = numpy.abs(fftshift(fft2(rts_wimage)))

	# Create log-polar transformed FFT mag images and register
	shape = image_fs.shape
	radius = shape[0] // 8  # only take lower frequencies
	warped_image_fs = warp_polar(image_fs, radius=radius, output_shape=shape,
	                             scaling='log', order=0)
	warped_rts_fs = warp_polar(rts_fs, radius=radius, output_shape=shape,
	                           scaling='log', order=0)

	warped_image_fs = warped_image_fs[:shape[0] // 2, :]  # only use half of FFT
	warped_rts_fs = warped_rts_fs[:shape[0] // 2, :]
	shifts, error, phasediff = phase_cross_correlation(warped_image_fs,
	                                                   warped_rts_fs)

	# Use translation parameters to calculate rotation (note that here the scale is 1:1; if it were not we could recover it from shiftc)
	shiftr, shiftc = shifts[:2]
	print(shiftr, shiftc)
	recovered_angle = (360 / shape[0]) * shiftr

	fig, axes = pt.subplots(2, 2, figsize=(8, 8))
	ax = axes.ravel()
	ax[0].set_title("Original Image FFT\n(magnitude; zoomed)")
	center = numpy.array(shape) // 2
	ax[0].imshow(image_fs[center[0] - radius:center[0] + radius,
	                      center[1] - radius:center[1] + radius],
	             cmap='magma')
	ax[1].set_title("Modified Image FFT\n(magnitude; zoomed)")
	ax[1].imshow(rts_fs[center[0] - radius:center[0] + radius,
	                    center[1] - radius:center[1] + radius],
	             cmap='magma')
	ax[2].set_title("Log-Polar-Transformed\nOriginal FFT")
	ax[2].imshow(warped_image_fs, cmap='magma')
	ax[3].set_title("Log-Polar-Transformed\nModified FFT")
	ax[3].imshow(warped_rts_fs, cmap='magma')
	fig.suptitle('Working in frequency domain can recover rotation and scaling')

	pt.savefig(working_dir + '/' + sample_handle + '/' + sample_handle + '-' + 'polar_fft_plot.pdf')

	return recovered_angle

def register_dapi(working_dir,reference_stack,stack_list,overall_shifts_output,sample_handle):

	"""

	Inputs
	------
	working_dir (str): path for output files
	reference_stack (str): path to z stack of DAPI channel reference (expects .czi format)
	stack_list (list of str): paths to z stacks of DAPI channel images to be registered relative to reference
	overall_shifts_output (str): filename to record shifts relative to reference
	sample_handle (str): sample name
	
	Returns
	------
	None (Saves file with calculated overall shifts and summary images of registration)

	"""

	###Import the images and extract the DAPI channel.

	dapi_channel = 3

	ref_stack = AICSImage(reference_stack)
	lazy_ref_dapi = ref_stack.get_image_dask_data("ZYX", T=0, C=dapi_channel)  # returns out-of-memory dask array
	ref_dapi = lazy_ref_dapi.compute()  # returns in-memory 3D numpy array

	reg_dapi = [] ###List of numpy arrays containing the DAPI stacks to be registered
	for stack in stack_list:

		s = AICSImage(stack)
		print('importing stack to register')
		lazy_obj = s.get_image_dask_data("ZYX", T=0,C=dapi_channel)
		reg_dapi.append( lazy_obj.compute() )

	###Step 1: Trim images to the same number of pixels in x and y and slices in z if necessary.

	nz1,nx1,ny1 = ref_dapi.shape
	minx,miny,minz = nx1,ny1,nz1
	
	stack_depths = [nz1]
	
	for data in reg_dapi:

		nz,nx,ny = data.shape
		stack_depths.append(nz)

		if nx < minx:
			minx = nx
		if ny < miny:
			miny = ny
		if nz < minz:
			minz = nz

	dimx = minx
	dimy = miny
	dimz = minz

	ref_dapi = ref_dapi[:dimz,:dimx,:dimy]
	
	for i in range(len(reg_dapi)):

		reg_dapi[i] = reg_dapi[i][:dimz,:dimx,:dimy]


	###Step 2: Take a MaxZ projection of the dapi stacks and compute the relative rotation angle and overall shift in x and y

	ref_maxz = numpy.amax(ref_dapi,axis=0)

	reg_maxzs = []
	rotated_maxzs = []

	angles = []
	xy_shifts = []

	for i in range(len(reg_dapi)):

		reg_maxz = numpy.amax(reg_dapi[i],axis=0)

		reg_maxzs.append( reg_maxz )

		recovered_angle = rotation_angle( ref_maxz, reg_maxz, working_dir, sample_handle )
		print(recovered_angle)
		print(ref_maxz.shape,reg_maxz.shape)
		angles.append(recovered_angle)

		dapi2_flt = img_as_float(reg_maxz)
		rotation_corrected = rotate(dapi2_flt,-recovered_angle)

		shifts, error, phasediff = phase_cross_correlation(ref_maxz,
                                                   rotation_corrected,
                                                   upsample_factor=1)
		xy_shifts.append(shifts)

		rotated_maxzs.append( rotation_corrected )

	###Step 3: Return to the full DAPI stack. Rotate each slice according to the recovered angle. Bin images in x and y for faster computation and compute the 3d cross correlation. (Note that we are only using this for the z shift between stacks)

	wbin = 10

	ref_binned = downscale_local_mean( ref_dapi, factors=(1,wbin,wbin))

	zshifts = []

	for i in range(len(reg_dapi)):

		angle = angles[i]

		if dimx%wbin == 0:
			dimx_binned = dimx//wbin
		else:
			dimx_binned = dimx//wbin + 1
		if dimy%wbin == 0:
			dimy_binned = dimy//wbin
		else:
			dimy_binned = dimy//wbin + 1

		new_stack = numpy.zeros((dimz,dimx_binned,dimy_binned))

		for zslice in range(dimz):

			dapi_slice = reg_dapi[i][zslice,:,:]

			dapi_slice_rotated = rotate(dapi_slice,-angle)

			new_stack[zslice,:,:] = downscale_local_mean( dapi_slice, factors=(wbin,wbin))

		shifts, error, phasediff = phase_cross_correlation(ref_binned, new_stack, upsample_factor=1)

		zshifts.append(shifts[0])

	###Step 4: Compute the boundaries of the overlap region between all shifted images in x, y, and z; record, along with the angle.

	shift_file = open(overall_shifts_output,'w')
	shift_file.write( ('\t').join(('stack','rotation angle','zstart','depth','xstart','height','ystart','width')) + '\n' )

	minx_shift = 0
	maxx_shift = 0

	miny_shift = 0
	maxy_shift = 0

	minz_shift = 0
	maxz_shift = 0

	for shift in zshifts:

		if shift < minz_shift:
			minz_shift = int(shift)
		if shift > maxz_shift:
			maxz_shift = int(shift)

	for shift in xy_shifts:

		if shift[0] < minx_shift:
			minx_shift = int(shift[0])
		if shift[0] > maxx_shift:
			maxx_shift = int(shift[0])
		if shift[1] < miny_shift:
			miny_shift = int(shift[1])
		if shift[1] > maxy_shift:
			maxy_shift = int(shift[1])
	
	ds = [- maxz_shift + stack_depths[0]]
	
	for image_ind in range(len(xy_shifts)):

		shiftz = zshifts[image_ind]
		shiftx,shifty = xy_shifts[image_ind]
		
		zstart = maxz_shift - int(shiftz)
		di = - zstart + stack_depths[image_ind+1]
		ds.append(di)
		
	ds = numpy.array(ds,dtype='int')
	d = numpy.min(ds)
		
	h = minx_shift - maxx_shift + dimx - 1
	w = miny_shift - maxy_shift + dimy - 1

	ref_shifted = ref_maxz[maxx_shift:maxx_shift+h,maxy_shift:maxy_shift+w]


	shift_file.write( ('\t').join( (reference_stack,str(0),str(maxz_shift),str(d),str(maxx_shift),str(h),str(maxy_shift),str(w) )) + '\n')

	for image_ind in range(len(xy_shifts)):

		shiftz = zshifts[image_ind]
		shiftx,shifty = xy_shifts[image_ind]
		angle = angles[image_ind]

		xstart = maxx_shift - int(shiftx)
		ystart = maxy_shift - int(shifty)
		zstart = maxz_shift - int(shiftz)

		shift_file.write( ('\t').join( (stack_list[image_ind],str(angle),str(zstart),str(d),str(xstart),str(h),str(ystart),str(w) )) + '\n')

		image_shifted = rotated_maxzs[image_ind][xstart:xstart+h,ystart:ystart+w]

		###Plot an RGB overlay for use in evaluating the registration

		nr, nc = ref_shifted.shape
		nr2, nc2 = ref_maxz.shape

		row_coords, col_coords = numpy.meshgrid(numpy.arange(nr), numpy.arange(nc),
	                                     indexing='ij')

		gamma = 1
		# build an RGB image with the unregistered sequence
		seq_im = numpy.zeros((nr2, nc2, 3))
		seq_im[..., 0] = (rotated_maxzs[image_ind]/numpy.amax(rotated_maxzs[image_ind]))**gamma
		seq_im[..., 1] = (ref_maxz/numpy.amax(ref_maxz))**gamma
		seq_im[..., 2] = (ref_maxz/numpy.amax(ref_maxz))**gamma

		# build an RGB image with the registered sequence
		reg_im = numpy.zeros((nr, nc, 3))
		reg_im[..., 0] = (image_shifted/numpy.amax(image_shifted))**gamma
		reg_im[..., 1] = (ref_shifted/numpy.amax(ref_shifted))**gamma
		reg_im[..., 2] = (ref_shifted/numpy.amax(ref_shifted))**gamma

		# build an RGB image with the registered sequence
		target_im = numpy.zeros((nr, nc, 3))
		target_im[..., 0] = (ref_shifted/numpy.amax(ref_shifted))**gamma
		target_im[..., 1] = (ref_shifted/numpy.amax(ref_shifted))**gamma
		target_im[..., 2] = (ref_shifted/numpy.amax(ref_shifted))**gamma

		# --- Save the result

		fig, (ax0, ax1, ax2) = pt.subplots(1,3, figsize=(10,5))

		ax0.imshow(seq_im,vmax=numpy.percentile(seq_im,90))
		ax0.set_title("Sequence, rotation")
		ax0.set_axis_off()

		ax1.imshow(reg_im,vmax=numpy.percentile(reg_im,90))
		ax1.set_title("Sequence, rotation + translation")
		ax1.set_axis_off()

		ax2.imshow(target_im,vmax=numpy.percentile(target_im,90))
		ax2.set_title("Target")
		ax2.set_axis_off()

		fig.tight_layout()
		pt.savefig(working_dir + '/' + sample_handle + '/' + sample_handle + '_registration_image_'+str(image_ind)+'.pdf')

	shift_file.close()

def save_registered_stacks(working_dir,stack_list,sample_handle,overall_shift_file=None):
	
	"""

	Inputs
	------
	working_dir (str): path for output files
	stack_list (list of str): a list of filenames. These are raw stacks corresponding to 3 rounds of measurements on the same sample. The reference stack (i.e. the one with respect to which displacements were calculated) must be the first in the list.
	overall_shift_file (str): a text file listing the net z, x, and y displacements of subsequent stacks relative to the first stack, calculated using the DAPI channel. Rows in this file should correspond to filenames in stacklist.
	sample_handle (str): sample name
	
	Returns
	------
	None (Saves images in ometiff format, which have been shifted according to the computed registration)
	
	"""
	
	overall_shifts_df = pd.read_csv(overall_shift_file,sep='\t',index_col=0,header=0)

	reference_stack = AICSImage(stack_list[0])
	print('importing reference data')

	###c_counter is a channel counter for all acquired channels
	c_counter = 0
	for channel in range(4):
		lazy_obj = reference_stack.get_image_dask_data("ZYX",T=0,C=channel)
		ref_data = lazy_obj.compute()

		ref_shifts = overall_shifts_df.loc[stack_list[0]].astype('int')
		print(ref_shifts)
		print(ref_shifts['ystart'],ref_shifts['width'])

		ref_data_shift = ref_data[ ref_shifts['zstart']:ref_shifts['zstart']+ref_shifts['depth'],
		                           ref_shifts['xstart']:ref_shifts['xstart']+ref_shifts['height'],
		                           ref_shifts['ystart']:ref_shifts['ystart']+ref_shifts['width']]

		OmeTiffWriter.save(ref_data_shift[:,:,:],uri = working_dir + '/' + sample_handle + '/' + sample_handle +'-channel-'+ str(c_counter) +'-registered.tif',dim_order='ZYX')
		OmeTiffWriter.save(numpy.amax(ref_data_shift,axis=0),uri=working_dir + '/' + sample_handle + '/' +sample_handle+'-channel-'+str(c_counter)+'-registered.-maxz.tif',dim_order='YX')

		c_counter += 1

		###Clear larger objects from memory

		del(ref_data_shift)
		del(ref_data)
		gc.collect()
		
	for stack_ind in range(len(stack_list)-1):

		stack = stack_list[stack_ind+1] ###note that the stack list contains the reference stack also, which has been processed above

		img = AICSImage(stack)
		print('importing stack')

		for channel in range(4):
			img_data_lazy = img.get_image_dask_data("ZYX",T=0,C=channel)
			img_data = img_data_lazy.compute()

			ref_shifts = overall_shifts_df.loc[stack].astype('int')

			img_shift = numpy.zeros((ref_shifts['depth'],ref_shifts['height'],ref_shifts['width']),dtype='uint16')

			z_counter = 0 ##This variable starts from 0 and keeps track of where slices are added to the new stack
			for z in range(ref_shifts['zstart'],ref_shifts['zstart']+ref_shifts['depth']):

				img_flt = img_as_float(img_data[z,:,:])
				slice_rot = rotate(img_flt, -ref_shifts['rotation angle'])
				slice_shift = slice_rot[ref_shifts['xstart']:ref_shifts['xstart']+ref_shifts['height'],
				                    ref_shifts['ystart']:ref_shifts['ystart']+ref_shifts['width']]

				img_shift[z_counter,:,:] = img_as_int(slice_shift) ###Convert rotated and shifted image back to uint16

				z_counter += 1

			OmeTiffWriter.save(img_shift,uri=working_dir + '/' + sample_handle + '/' + sample_handle+'-channel-'+str(c_counter)+'-registered.tif',dim_order='ZYX')
			OmeTiffWriter.save(numpy.amax(img_shift,axis=0),uri=working_dir + '/' + sample_handle + '/' + sample_handle+'-channel-'+str(c_counter)+'-registered.-maxz.tif',dim_order='YX')

			c_counter += 1

			###Clear larger objects from memory

			del(img_shift)
			del(img_data)
			gc.collect()