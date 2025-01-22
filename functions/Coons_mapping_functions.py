import numpy
import matplotlib.pylab as pt
import skimage
import scipy.optimize as scopt
from scipy.interpolate import interp1d
from skimage.morphology import square

def get_boundary_curves_and_corners(mask_file):

	"""
	Helper function for defining Coons patch

	Inputs
	------
	mask file (str): path to a .tiff file with the image mask

	Returns
	-------
	Contour pixel list (numpy array): Nx2 array corresponding to the list of pixels in the mask edge. [[x1,x2.,,,],[y1,y2...]]
	Corner ind list (list, length 4): the indexes in contour pixel list that correspond to the corners of the mask
	"""

	###FIND EDGE PIXELS

	image_arr = skimage.io.imread( mask_file, plugin='tifffile')

	edges = skimage.feature.canny(image_arr)

	contours = skimage.measure.find_contours(image_arr,1)

	contour_pixel_list = numpy.rint(contours[0]).astype('int')

	###FIND CORNER PIXELS

	corner_scores = skimage.feature.corner_harris(image_arr)

	###DETECT 4 BOTTOM CORNERS

	potential_corners = edges*corner_scores

	potential_corner_inds = numpy.argsort(potential_corners,axis=None)[::-1] ###Returns flattened indexes, ordered by corner score

	num_corners_found = 0
	corner_counter = 0

	corner_list = []

	while num_corners_found < 3.5:

		flat_corner_ind = potential_corner_inds[corner_counter]

		cx,cy = numpy.unravel_index(flat_corner_ind,image_arr.shape)

		###Check that the potential corner is within the bottom 500 pixels and not right next to another corner

		corner_dists = []
		for c in corner_list:
			corner_dists.append(numpy.abs(cy-c[1]))

		if image_arr.shape[0] - cx < 500 and (len(corner_dists)<.5 or min(corner_dists)>100):

			corner_list.append( (cx,cy) )

			num_corners_found += 1

		corner_counter += 1

	###SORT CORNERS FROM LEFT TO RIGHT (SHORT AXIS)

	corner_arr = numpy.array(corner_list)
	corner_order = numpy.argsort(corner_arr[:,1])

	###FIND CLOSEST ENTRY TO EACH CORNER FROM CONTOUR LIST

	corner_ind_list = []

	for ind in corner_order:

		corner = corner_list[ind]

		xdiffs = contour_pixel_list[:,0] - corner[0]
		ydiffs = contour_pixel_list[:,1] - corner[1]

		corner_ind = numpy.argmin( xdiffs**2 + ydiffs**2)


		corner_ind_list.append(corner_ind)

	return contour_pixel_list, corner_ind_list

def partial_contours(contour_coords,ind_list):

	"""

	Inputs
	------
	contour_coords (numpy array, Nx2): Nx2 array corresponding to the list of pixels in the mask edge. [[x1,x2.,,,],[y1,y2...]]
	ind_list (list, length 4): the indexes in contour pixel list that correspond to the corners of the mask

	This function takes an ordered list of pixels that identify the boundary of the region,
	together with the indexes of the 4 corner pixels, and returns lists that correspond to
	the pixels along the segments in between each pair of corners.
	Note that the segments are ordered so as to be compatible with the interpolation step
	of the Coons mapping.

	Returns
	-------
	L1,L2,M1,M2 (numpy arrays, nx2): The contour coordinates, divided into portions that correspond to the contour between each pair of corners.

	"""

	ncoords,d2 = contour_coords.shape

	iorder = list(numpy.argsort(ind_list))

	forwards = True

	if iorder in [[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2]]:

		forwards = True

	elif iorder in [[3,2,1,0],[2,1,0,3],[1,0,3,2],[0,3,2,1]]:
		forwards = False

	else:
		print('Something is confusing about the contour list; iorder=',iorder)

	if not forwards:

		contour_coords = contour_coords[::-1]

		for i in range(len(ind_list)):

			ind_list[i] = ncoords - ind_list[i] - 1

	i1,i2,i3,i4 = ind_list

	segments = [[i1,i4],[i2,i3],[i1,i2],[i3,i4]] ###Start, stop indexes for each segment

	def get_backwards_segment(start,stop):

		if stop < start:

			seg = contour_coords[start:stop-1:-1,:].copy()
		else:
			seg1 = contour_coords[start::-1,:].copy()
			seg2 = contour_coords[-1:stop-1:-1,:].copy()
			seg = numpy.concatenate((seg1,seg2),axis=0)

		return seg

	def get_forwards_segment(start,stop):

		if stop > start:

			seg = contour_coords[start:stop+1,:].copy()
		else:
			seg1 = contour_coords[start:,:].copy()
			seg2 = contour_coords[:stop+1,:].copy()
			seg = numpy.concatenate((seg1,seg2),axis=0)

		return seg


	L1 = get_backwards_segment(i1,i4)
	L2 = get_forwards_segment(i2,i3)

	M1 = get_forwards_segment(i1,i2)
	M2 = get_backwards_segment(i4,i3)


	return L1,L2,M1,M2

def spl_from_curve(curve):

	"""
	Cubic spline interpolation of a curve, [[x1,x2.....],[y1,y2....]]

	Inputs
	------
	curve (numpy array, Nx2): here, pixels corresponding to a boundary

	Returns
	-------
	spl (spline object): spline object approximating specified contour
	"""

	clen,d2 = curve.shape

	path_t = numpy.linspace(0,1,clen)

	spl = interp1d(path_t,curve.T,kind='cubic')

	return spl

def Coons_mapping(mask_file,centroid_file,plot=False,axis_list=None):

	"""
	Performs Coons bilinear interpolation

	Inputs
	------
	mask_file (str): path to .tiff file with image mask.
	centroid_file (str): path to .npy format file with list of cell centroid locations
	plot (Bool, optional): option to output a plot with the original and transformed boundary curves and cell centroids
	axis_list (list of matplotlib axes objects, optional): list of axes to plot on

	Returns
	-------
	st_centroids (numpy array): list of centroids in the transformed coordinate system
	"""

	contour_pixel_list, ind_list = get_boundary_curves_and_corners(mask_file)

	L1,L2,M1,M2 = partial_contours(contour_pixel_list,ind_list)

	L1_spl = spl_from_curve(L1)
	L2_spl = spl_from_curve(L2)
	M1_spl = spl_from_curve(M1)
	M2_spl = spl_from_curve(M2)

	###DEFINE COONS TRANSFORMATION

	def L_c(s,t):

		return (1-t)*L1_spl(s) + t*L2_spl(s)

	def L_d(s,t):

		return (1-s)*M1_spl(t) + s*M2_spl(t)

	def B(s,t):

		return L1_spl(0)*(1-s)*(1-t) + L1_spl(1.)*s*(1-t) + L2_spl(0.)*(1-s)*t+L2_spl(1.)*s*t

	def Coons(s,t):

		return L_c(s,t) + L_d(s,t) - B(s,t)

	###CREATE INVERSE MAPPING

	trange = numpy.arange(0.,1.001,.01)
	srange = numpy.arange(0.,1.001,.0025)

	coons_list = []
	st_list = []

	for t in trange[1:-1]:
		coon = numpy.array([Coons(s,t) for s in srange])
		st = numpy.array([[s,t] for s in srange])
		coons_list.append(coon)
		st_list.append(st)

	coons_list = numpy.concatenate(coons_list,axis=0)
	st_list = numpy.concatenate(st_list,axis=0)

	###INVERT VIA GRID MAPPING

	def Coons_inv(xy_pos):

		closest_gridpoint = numpy.argmin( (coons_list[:,0]-xy_pos[0])**2 + (coons_list[:,1]-xy_pos[1])**2 )

		x0 = st_list[closest_gridpoint]

		return x0

	###MAP CELL CENTROID LOCATIONS TO COONS COORDINATES (AND RECORD IF 

	centroids = numpy.load(centroid_file)
	ncells,d2 = centroids.shape
	st_centroids = []
	for i in range(ncells):

		st = Coons_inv(centroids[i,:])
		st_centroids.append(st)

	st_centroids = numpy.array(st_centroids)

	if plot:

		###PLOT CURVES, MAPPING, AND CENTROIDS

		ax = axis_list[0]

		for t in trange[::10]:
			coon = numpy.array([Coons(s,t) for s in srange[::10]])
			ax.plot(coon[:,1],coon[:,0],'k',linewidth=1)

		for s in srange[::10]:
			coon = numpy.array([Coons(s,t) for t in trange[::10]])
			ax.plot(coon[:,1],coon[:,0],'k',linewidth=1)

		ax.plot(L1[:,1],L1[:,0],linewidth=3,color='brown')
		ax.plot(L2[:,1],L2[:,0],linewidth=3,color='teal')
		ax.plot(M1[:,1],M1[:,0],linewidth=3,color='pink')
		ax.plot(M2[:,1],M2[:,0],linewidth=3,color='gold')

		ax.set_aspect('equal')

		ax.yaxis.set_inverted(True)

		ax.scatter(centroids[:,1],centroids[:,0],s=2,c='purple')

		ax.set_xticks([])
		ax.set_xticklabels([])

		ax.set_yticks([])
		ax.set_yticklabels([])

		ax.spines.top.set_visible(False)
		ax.spines.right.set_visible(False)
		ax.spines.bottom.set_visible(False)
		ax.spines.left.set_visible(False)

		ax = axis_list[1]

		for t in trange[::10]:
			rect = numpy.array([(s,t) for s in srange[::10]])
			ax.plot(rect[:,0],rect[:,1],'k',linewidth=1)

		for s in srange[::10]:
			rect = numpy.array([(s,t) for t in trange[::10]])
			ax.plot(rect[:,0],rect[:,1],'k',linewidth=1)

		ax.plot([0,1],[1,1],linewidth=3,color='brown')
		ax.plot([0,1],[0,0],linewidth=3,color='teal')
		ax.plot([0,0],[0,1],linewidth=3,color='pink')
		ax.plot([1,1],[0,1],linewidth=3,color='gold')

		ax.scatter(st_centroids[:,0],st_centroids[:,1],s=2,c='purple')

		ax.set_xticks([])
		ax.set_xticklabels([])

		ax.set_yticks([])
		ax.set_yticklabels([])

		ax.spines.top.set_visible(False)
		ax.spines.right.set_visible(False)
		ax.spines.bottom.set_visible(False)
		ax.spines.left.set_visible(False)

	return st_centroids