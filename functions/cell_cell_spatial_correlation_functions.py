import numpy
from scipy.stats import binned_statistic
from sklearn.neighbors import NearestNeighbors

def cell_spatial_autocorrelation(counts, centroids, bins = numpy.arange(0,21*80,40)):

	"""

	Inputs
	------
	counts (numpy array, n_cells x 1): array of values for each cell region (here, gene expression as either spot counts or intensities)
	centroids (numpy array, n_cells x 2): array of centroid locations for each cell region
	bins (array, optional): distance bins on which to compute autocorrelation, in pixels

	Returns
	-------
	prod_stat (numpy array, n_bins x 1): The autocorrelation value for that bin
	prod_std (numpy array, n_bins x 1): The standard deviation of the value in the bin
	counts (numpy array, n_bins x 1): The number of observations in the bin

	"""


	##Center and normalize counts
	counts = (counts - counts.mean())/counts.std()

	##Calculate all pairwise distances

	ncells = len(counts)

	dist_list = []
	counts_prod_list = []

	for i in range(ncells):

		for j in range(i+1):

			distance = numpy.sqrt((centroids[i,0] - centroids[j,0])**2 + (centroids[i,1] - centroids[j,1])**2)
			count_prod = counts[i]*counts[j]

			dist_list.append(distance)
			counts_prod_list.append(count_prod)

	###Calculate average, count, and standard deviation of the count product in each bin

	counts,bin_edges,binnumber = binned_statistic( dist_list, counts_prod_list, statistic='count', bins=bins)
	prod_stat,bin_edges,binnumber = binned_statistic( dist_list, counts_prod_list, statistic='mean', bins=bins)
	prod_std, bin_edges,binnumber = binned_statistic( dist_list, counts_prod_list, statistic='std', bins=bins)

	return prod_stat, prod_std, counts