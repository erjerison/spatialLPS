import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scopt
import gpytoolbox as gpy
import scipy.sparse as sp
import scipy.linalg as la
import scipy.ndimage
import scipy.sparse.linalg as spla
import pandas as pd
from functions.tail_graph_functions import lst_sq_B
from functions.tail_graph_functions import lst_sq_cotan
import keys.paths_filenames as paths_filenames
import functions.file_import_utilities as file_import_utilities
from functions.tailmap_plotting_functions import plot_cell_intensities_diverging_cm_nan


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams.update({
    'text.latex.preamble': r'\usepackage{amsfonts}'
})

def load_data():
    import pandas as pd
    csv_path = './tables/d-08052022_LPS20ugml_10hrs_tail1/d-08052022_LPS20ugml_10hrs_tail1-CentroidLocations.csv'
    df = pd.read_csv(csv_path, header=None)
    X = df[0].to_numpy()
    Y = df[1].to_numpy()
    vertices = np.column_stack((X, Y))
    return vertices

def diffusion_simulation(vertices, concave_hull, ell=1.0):
    """
    Simulate the diffusion equation on a 2D mesh defined by vertices and concave_hull.
    Parameters
    ----------
    L : (n_vertices, n_vertices) sparse cotangent Laplacian matrix
        The cotangent Laplacian matrix.
    vertices: (n_vertices, 2) array of vertex coordinates
        The coordinates of the vertices in 2D.
    concave_hull: (n_faces, 3) array of indices into vertices
        The facelist defining the triangulation of the alpha-shape mesh.
    Returns
    -------
    u : (n_steps, n_vertices) array
        The solution of the diffusion equation at each time step.
    """

    # parameters
    T = 3e1
    a = 1

    D = ell**2 
    # Laplacian
    # rescale vertices
    vertices_r = vertices #/ np.max(np.abs(vertices.ravel()), axis=0) 
    M = gpy.massmatrix(vertices_r, concave_hull, type='barycentric').tocsc()
    M = M/ np.mean(M.diagonal())
    Ms = sp.linalg.inv(np.sqrt(M))
    #L = -sp.linalg.inv(M) @ gpy.cotangent_laplacian(vertices, concave_hull) 
    L = - Ms @ gpy.cotangent_laplacian(vertices, concave_hull)  @ Ms
    # Bilaplacian
    B = L.dot(L)

    # initial condition
    u0 = np.zeros(vertices.shape[0])
    f1 = np.zeros(vertices.shape[0])
    u0 = 0.01*np.random.normal(size=vertices.shape[0])
    f1 = 1*np.random.normal(size=vertices.shape[0])

    # time-stepping
    dx2 = np.mean(M.diagonal())
    dt = 0.01 *(1/ell**2)*dx2
    n_steps = int(T/dt)
    u = np.zeros((n_steps, vertices.shape[0]))
    u[0] = u0
    for n in range(1, n_steps):
        u[n]= u[n-1] + dt* (D*L.dot(u[n-1]) - a * u[n-1] +f1)#+1e-2*np.random.normal(size=vertices.shape[0])*np.sqrt(dt)
    u = u[::100]
    return u



def swifthohnenberg(vertices, concave_hull, lam=5.0):
    """
    Simulate the Swift-Hohnenberg equation on a 2D mesh defined by vertices and concave_hull.
    Parameters
    ----------
    L : (n_vertices, n_vertices) sparse cotangent Laplacian matrix
        The cotangent Laplacian matrix.
    vertices: (n_vertices, 2) array of vertex coordinates
        The coordinates of the vertices in 2D.
    concave_hull: (n_faces, 3) array of indices into vertices
        The facelist defining the triangulation of the alpha-shape mesh.
    Returns
    -------
    u : (n_steps, n_vertices) array
        The solution of the SH equation at each time step.
    """
    # parameters
    T = 5e2
    k = 2*np.pi/lam
    g0 = 2* k**2 

    a = 0.97
    b= np.sqrt(2*(1-a))
    c = 1
    # Laplacian
    # rescale vertices to (0,1)
    vertices_r = vertices / np.max(np.abs(vertices.ravel()), axis=0) 
    M = gpy.massmatrix(vertices_r, concave_hull, type='barycentric').tocsc()
    M = M/ np.mean(M.diagonal())
    Ms = sp.linalg.inv(np.sqrt(M))
    #L = -sp.linalg.inv(M) @ gpy.cotangent_laplacian(vertices, concave_hull) 
    L = - Ms @ gpy.cotangent_laplacian(vertices, concave_hull)  @ Ms
    L = L / k**2
    # Bilaplacian
    B = L.dot(L)

    # initial condition
    u0 = np.zeros(vertices_r.shape[0])
    u0 = 0.01*np.random.normal(size=vertices_r.shape[0])

    # time-stepping
    dt = 0.01 * 1/lam**2
    n_steps = int(T/dt)
    u = np.zeros((n_steps, vertices_r.shape[0]))
    u[0] = u0
    for n in range(1, n_steps):
        u[n]= u[n-1] + dt* (-2*k**2*L.dot(u[n-1]) -  B.dot(u[n-1]) - a * u[n-1] - b* u[n-1]**2 - c*u[n-1]**3)
    u = u[::100]
    return u

import matplotlib as mpl

def plot_power_spectrum(outname, eigenvector_mat, data, type='cotan'):
    """ Plot the power spectrum of the data on the eigenbasis defined by eigenvectors."""

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    for g in range(n_datapoints):
        frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])
        frac_perm = (B_R_perm**2)[:,g]/np.sum((B_R_perm**2)[:,g])

        smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
        smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)
        
        ax.loglog(np.arange(1,len(smoothed_frac)+.5),smoothed_frac,linewidth=2,alpha=.7)
        ax.loglog(np.arange(1,len(smoothed_frac)+.5),smoothed_frac_perm,color='grey',linewidth=1,alpha=.5)
    ax.set_xlabel('Mode Number',fontsize=16)
    ax.set_box_aspect(1)
    ax.set_ylabel('Fraction of Power',fontsize=16)
    ax.set_title('Power Spectrum',fontsize=16)
    fig.savefig('power_spectrum_'+outname+'.pdf', bbox_inches='tight')

def plot_power_spectrum_eigs(outname, eigenvalues, eigenvector_mat, data, type='cotan'):
    """ Plot the power spectrum of the data on the eigenbasis defined by eigenvectors."""

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    for g in range(n_datapoints):
        frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])
        frac_perm = (B_R_perm**2)[:,g]/np.sum((B_R_perm**2)[:,g])

        smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
        smoothed_eigs = (1/0.12)**2*scipy.ndimage.gaussian_filter1d(eigenvalues,sigma=3)
        smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)
        
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac,linewidth=2,alpha=.7)
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac_perm,color='grey',linewidth=1,alpha=.5)
    ax.set_xlabel('k (1/um)',fontsize=16)
    ax.set_box_aspect(1)
    ax.set_ylabel('Fraction of Power',fontsize=16)
    ax.set_title('Power Spectrum',fontsize=16)
    fig.savefig('power_spectrum_'+outname+'.pdf', bbox_inches='tight')


def plot_power_spectrum_eigs_reps(outname, eigenvalues, eigenvector_mat, data, type='cotan'):
    """ Plot the power spectrum of the data on the eigenbasis defined by eigenvectors."""

    n_cells, n_datapoints, n_reps = data.shape
    data_mat_c = data - np.mean(data, axis=0)
    data_mat_centered = data_mat_c.reshape((n_cells, n_datapoints * n_reps))
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    B_R = B_R.reshape((B_R.shape[0],n_datapoints, n_reps))
    B_R_perm = B_R_perm.reshape((B_R_perm.shape[0],n_datapoints, n_reps))
    for g in range(n_datapoints):

        frac = np.mean((B_R**2)[:,g,:], axis=-1)/np.sum(np.mean((B_R**2)[:,g,:], axis=-1))
        frac_perm = np.mean((B_R_perm**2)[:,g,:],axis=-1)/np.sum(np.mean((B_R_perm**2)[:,g,:],axis=-1))

        smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
        smoothed_eigs = (1/0.12)**2*scipy.ndimage.gaussian_filter1d(eigenvalues,sigma=3)
        smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)
        
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac,linewidth=2, alpha=.7)
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac_perm,color='grey',linewidth=1,alpha=.5)
    ax.set_xlabel('k (1/um)',fontsize=16)
    ax.set_box_aspect(1)
    ax.set_ylabel('Fraction of Power',fontsize=16)
    ax.set_title('Power Spectrum',fontsize=16)
    fig.savefig('power_spectrum_'+outname+'.pdf', bbox_inches='tight')

def plot_power_spectrum_doubleX_reps(outname, eigenvalues, eigenvector_mat, data, type='cotan'):
    """ Plot the power spectrum of the data on the eigenbasis defined by eigenvectors."""

    n_cells, n_datapoints, n_reps = data.shape
    data_mat_c = data - np.mean(data, axis=0)
    data_mat_centered = data_mat_c.reshape((n_cells, n_datapoints * n_reps))
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    B_R = B_R.reshape((B_R.shape[0],n_datapoints, n_reps))
    B_R_perm = B_R_perm.reshape((B_R_perm.shape[0],n_datapoints, n_reps))
    colr = plt.cm.Blues(np.linspace(0, 1, n_datapoints+4))
    for g in range(n_datapoints):

        frac = np.mean((B_R**2)[:,g,:], axis=-1)/np.sum(np.mean((B_R**2)[:,g,:], axis=-1))
        frac_perm = np.mean((B_R_perm**2)[:,g,:],axis=-1)/np.sum(np.mean((B_R_perm**2)[:,g,:],axis=-1))

        smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
        smoothed_eigs = (1/0.12)**2*scipy.ndimage.gaussian_filter1d(eigenvalues,sigma=3)
        smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)
        
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac,linewidth=2, color=colr[g+4], alpha=.7)
        ax.loglog(np.sqrt(smoothed_eigs),smoothed_frac_perm,color='grey',linewidth=1,alpha=.5)
    ax.set_xlabel('k (1/um)',fontsize=16)

    def mode2k(mode):
        return np.interp(mode, np.arange(1,len(smoothed_frac)+.5), np.sqrt(smoothed_eigs))
    def k2mode(k):
        return np.interp(k, np.sqrt(smoothed_eigs), np.arange(1,len(smoothed_frac)+.5))
    secax = ax.secondary_xaxis('top', functions=(k2mode, mode2k))
    secax.set_xlabel('Mode Number')
    ax.set_box_aspect(1)
    ax.set_ylabel('Fraction of Power',fontsize=16)
    ax.set_title('Power Spectrum',fontsize=16)
    fig.savefig('power_spectrum_'+outname+'.pdf', bbox_inches='tight')

def measure_lengthscale(data, areas_file, eigenvector_mat, eigenvalue_file, type='adj'):

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    
    g = -1
    frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])

    smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)

    cum_frac = np.cumsum((B_R**2)[:,g])/np.sum((B_R**2)[:,g])

    frac_perm = (B_R_perm**2)[:,g]/np.sum((B_R_perm**2)[:,g])

    cum_frac_perm = np.cumsum((B_R_perm**2)[:,g])/np.sum((B_R_perm**2)[:,g])

    smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)

    ###Measure the amount of power per mode in the null to determine the noise floor

    fnull = np.mean(frac_perm)

    ###Determine when the spectrum goes below the noise floor

    very_smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=10)

    crossover_mode = np.argmax( very_smoothed_frac < fnull ) ###First entry where the data curve drops below the noise floor

    num_modes_fit = crossover_mode

    if type=='adj':
        mode_vec = np.arange(n_cells)
        cell_areas = pd.read_csv(areas_file,sep='\t',header=0,usecols=['Area (pixels)'],dtype='float')
    
        mean_area = np.mean(cell_areas)
        #print(mean_area)
        mean_area_um2 = mean_area*.12**2 ###Scale to convert pixel area to area in um^2
        kscale = np.sum( frac[:num_modes_fit]*mode_vec[:num_modes_fit]/np.sum(frac[:num_modes_fit]) )
        print(kscale)
        length_scale = np.sqrt(n_cells/kscale*mean_area_um2)
    else:
        pixel_to_um =  0.12 # um/pixel ; 1 pixel = 0.12 um
        eigenvalues =  np.sqrt(np.abs(np.load(eigenvalue_file)))
        kscale = np.sum( frac[:num_modes_fit]*eigenvalues[:num_modes_fit]/np.sum(frac[:num_modes_fit]) )
        print(kscale)
        length_scale = 1/kscale
    return length_scale

def measure_lengthscale_adj(data, areas_file, eigenvector_mat, eigenvalue_file, type='adj'):

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
        ###Permuted cells
        B_R_perm,XT_perm = lst_sq_B(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)
    
    g = -1
    frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])

    smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)

    cum_frac = np.cumsum((B_R**2)[:,g])/np.sum((B_R**2)[:,g])

    frac_perm = (B_R_perm**2)[:,g]/np.sum((B_R_perm**2)[:,g])

    cum_frac_perm = np.cumsum((B_R_perm**2)[:,g])/np.sum((B_R_perm**2)[:,g])

    smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)

    ###Measure the amount of power per mode in the null to determine the noise floor

    fnull = np.mean(frac_perm)

    ###Determine when the spectrum goes below the noise floor

    very_smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=10)

    crossover_mode = np.argmax( very_smoothed_frac < fnull ) ###First entry where the data curve drops below the noise floor

    num_modes_fit = crossover_mode

    mode_vec = np.arange(n_cells)
    cell_areas = pd.read_csv(areas_file,sep='\t',header=0,usecols=['Area (pixels)'],dtype='float')

    mean_area = np.mean(cell_areas)
    #print(mean_area)
    mean_area_um2 = mean_area*.12**2 ###Scale to convert pixel area to area in um^2
    kscale = np.sum( frac[:num_modes_fit]*mode_vec[:num_modes_fit]/np.sum(frac[:num_modes_fit]) )
    print('kscale=',kscale)
    length_scale = np.sqrt(n_cells/kscale*mean_area_um2)
    return length_scale, kscale


def measure_lengthscale_cotan(data, eigenvalues, eigenvector_mat):

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
    ###Permuted cells
    B_R_perm,XT_perm = lst_sq_cotan(data_mat_centered[np.random.permutation(n_cells),:],eigenvector_mat,num_modes=n_cells)

    g = -1
    frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])

    smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)

    cum_frac = np.cumsum((B_R**2)[:,g])/np.sum((B_R**2)[:,g])

    frac_perm = (B_R_perm**2)[:,g]/np.sum((B_R_perm**2)[:,g])

    cum_frac_perm = np.cumsum((B_R_perm**2)[:,g])/np.sum((B_R_perm**2)[:,g])

    smoothed_frac_perm = scipy.ndimage.gaussian_filter1d(frac_perm,sigma=2)

    ###Measure the amount of power per mode in the null to determine the noise floor

    fnull = np.mean(frac_perm)

    ###Determine when the spectrum goes below the noise floor

    very_smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=10)

    crossover_mode = np.argmax( very_smoothed_frac < fnull ) ###First entry where the data curve drops below the noise floor

    num_modes_fit = crossover_mode
    
    kscale = np.sum( frac[:num_modes_fit]*eigenvalues[:num_modes_fit]/np.sum(frac[:num_modes_fit]) )
    print('kscale cotan = ', kscale)
    length_scale = 0.12*np.sqrt(np.abs(1/kscale))
    return length_scale


def get_peak_modenumber(data, eigenvalues, eigenvector_mat, areas_file, type='cotan'):
    # if eigenvalues is 

    n_cells, n_datapoints = data.shape
    data_mat_centered = data - np.mean(data, axis=0)
    if type == 'cotan':
        B_R,XT = lst_sq_cotan(data_mat_centered,eigenvector_mat,num_modes=n_cells)
    else:
        B_R,XT = lst_sq_B(data_mat_centered,eigenvector_mat,num_modes=n_cells)
    g = -1
    frac = (B_R**2)[:,g]/np.sum((B_R**2)[:,g])

    smoothed_frac = scipy.ndimage.gaussian_filter1d(frac,sigma=3)
    idx_max = np.argmax(smoothed_frac) # max mode number
    print(idx_max)
    e_star = eigenvalues[idx_max]
    # convert mode number and e_star to lengthscales

    cell_areas = pd.read_csv(areas_file,sep='\t',header=0,usecols=['Area (pixels)'],dtype='float')

    mean_area = np.mean(cell_areas)
    #print(mean_area)
    mean_area_um2 = mean_area*.12**2 ###Scale to convert pixel area to area in um^2

    length_scale_mode = np.sqrt(n_cells/idx_max*mean_area_um2)
    length_e = 0.12*1/np.sqrt(e_star)

    return length_e, length_scale_mode, idx_max



if __name__ == "__main__":
    #vertices = load_data()
    from functions import tail_graph_functions

    sample_list = file_import_utilities.import_sample_list(paths_filenames.sample_key,column='Medium to High Activation')

    sample = sample_list[0]


    segmentation_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-segmentation.npy'
    centroid_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-xy_centroids.npy'
    eigenvector_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cotan_eigenvectors-svd.npy'
    eigenvalue_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cotan_eigenvalues-svd.npy'
    areas_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-cell_stats.txt'

    graph_eigs_file = paths_filenames.table_path + '/' + sample + '/' + sample + '-graph_eigenvectors-svd.npy'

    vertices = np.load(centroid_file)
    concave_hull, edge_points, boundaries = tail_graph_functions.alpha_shape(vertices, alpha=0.007)
    concave_hull = np.array(concave_hull)
    boundaries = np.array(boundaries)

    eigenvector_mat = np.load(eigenvector_file)
    graph_eigenvector_mat = np.load(graph_eigs_file)

    Anorm = tail_graph_functions.calculate_save_adjacency_matrix(segmentation_file,paths_filenames.table_path,sample)

    # weighted cotan laplacian matrix and eigendecomposition
    M = gpy.massmatrix(vertices,concave_hull, type='barycentric')
    L = gpy.cotangent_laplacian(vertices, concave_hull)
    Ms = sp.linalg.inv(np.sqrt(M))
    Lm = Ms @ L @ Ms
    wm, vm = la.eigh(Lm.toarray())

    u = diffusion_simulation(L, vertices, concave_hull)

    power_spectrum_name = 'diffusion_simulation_cotan'
    print('u shape', u.shape)
    print('u[-1] shape', u[-1].shape)
    plot_power_spectrum(power_spectrum_name, eigenvector_mat, u[10:,:].T, type='cotan')

    plot_power_spectrum_eigs(power_spectrum_name, wm, vm, u[10:,:].T, type='cotan')
    power_spectrum_name = 'diffusion_simulation_adj'
    plot_power_spectrum(power_spectrum_name, graph_eigenvector_mat, u[10:,:].T, type='adjacency')

    plot_power_spectrum_doubleX_reps('diffusion_simulation_cotan_doublex', wm, vm, (u[10:,:].T)[:,:,None], type='cotan')

    fig,axes = plt.subplots(1,1,figsize=(6,12))
    plot_cell_intensities_diverging_cm_nan(segmentation_file, u[-1] - np.min(u[-1]), axes, sample, alpha_bg=0.1, scalecolor='k')
    fig.savefig('diffusion_simulation_cells.pdf', bbox_inches='tight')
    ## power spectrum decomposition

    # using Adjacency matrix

    # using cotan Laplacian


    ### Swift-Hohnenberg
    u = swifthohnenberg(vertices, concave_hull, lam=6.0)

    plt.figure(figsize=(5,5))
    plt.scatter(vertices[:,0]*0.12, vertices[:,1]*0.12, c=u[-1,:], s=10, cmap='viridis')
    plt.colorbar()
    plt.title('SH simulation')
    plt.savefig('SH_simulation.pdf', bbox_inches='tight')
    fig,axes = plt.subplots(1,1,figsize=(6,12))
    plot_cell_intensities_diverging_cm_nan(segmentation_file, u[-1] - np.min(u[-1]), axes, sample, scalecolor='k')
    fig.savefig('SH_simulation_cells.pdf', bbox_inches='tight')

    # using Adjacency matrix

    # using cotan Laplacian
    power_spectrum_name = 'SH_simulation_cotan'
    print('u shape', u.shape)
    print('u[-1] shape', u[-1].shape)
    plot_power_spectrum(power_spectrum_name, eigenvector_mat, u[5000::1000,:].T, type='cotan')
    plot_power_spectrum_eigs(power_spectrum_name, wm, vm, u[5000::1000,:].T, type='cotan')
    power_spectrum_name = 'SH_simulation_cotan_adj'
    plot_power_spectrum(power_spectrum_name, graph_eigenvector_mat, u[5000::1000,:].T, type='adjacency')


    plot_power_spectrum_doubleX_reps('SH_simulation_cotan_doublex', wm, vm, (u[5000::1000,:].T)[:,:,None], type='cotan')

    l_e_cotan, l_mode_cotan = get_peak_modenumber(u[10:,:].T, wm, vm, areas_file)
    l_e_adj, l_mode_adj = get_peak_modenumber(u[10:,:].T, np.arange(u.shape[1]), graph_eigenvector_mat, areas_file)


    ### variable lambda for SH - measure correlation length
    ells = np.linspace(3, 15, 10)
    measured_lengths_adj = np.zeros_like(ells)
    measured_lengths_cotan = np.zeros_like(ells)

    var_lengths_adj = np.zeros_like(ells)
    var_lengths_cotan = np.zeros_like(ells) 


    M0 = np.mean(gpy.massmatrix(vertices, concave_hull, type='barycentric').diagonal())
    scale = np.sqrt(M0)*px_to_um

    # mass-matrix weighted eigenvalues with 
    M = gpy.massmatrix(vertices,concave_hull, type='barycentric')
    L = gpy.cotangent_laplacian(vertices, concave_hull)
    Lm = sp.linalg.inv(M) @ L

    Ms = sp.linalg.inv(np.sqrt(M))

    #Lm = np.mean(M)* sp.linalg.inv(M) @ L #L/ np.mean(M) #sp.linalg.inv(M) @ L

    Lm = Ms @ L @ Ms

    N_reps = 4
    #Lm = 0.5*(Lm + Lm.T)
    wm, vn = la.eigh(Lm.toarray())

    u_res = np.zeros((vertices.shape[0], len(ells), N_reps))


    #print(M.diagonal())
    idx_adj = np.zeros((len(ells), N_reps))
    idx_cotan = np.zeros((len(ells), N_reps))
    wm, vm = la.eigh(Lm.toarray())
    len_scale = 1/np.sqrt(wm[1:])
    for i, ell in enumerate(ells):
        vals_adj = []
        vals_cotan = []
        for n in range(N_reps):
            u = swifthohnenberg(vertices, concave_hull, lam=ell)
            u_res[:,i,n] = u[-1]

            # measure correlation length
            l_e_cotan, _, idx_max_cotan = get_peak_modenumber(u[10:,:].T, wm, vm, areas_file, type='cotan')
            _, l_mode_adj, idx_max_adj = get_peak_modenumber(u[10:,:].T, np.arange(u.shape[1]), graph_eigenvector_mat, areas_file, type='adj')
            idx_cotan[i,n] = idx_max_cotan
            idx_adj[i,n] = idx_max_adj
            vals_adj.append(2*np.pi*l_mode_adj)
            vals_cotan.append(2*np.pi*l_e_cotan)
        measured_lengths_adj[i] = np.mean(np.array(vals_adj))
        measured_lengths_cotan[i] = np.mean(np.array(vals_cotan))
        var_lengths_adj[i] = np.std(np.array(vals_adj))
        var_lengths_cotan[i] = np.std(np.array(vals_cotan))
    plot_power_spectrum_doubleX_reps('variable_ell_reps_SH_doublex', wm, vm, u_res[:,1:8,:], type='cotan')
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plt.plot(scale*ells, scale*ells, color='k', linestyle='--', label=r'$\ell$')
    ax.scatter(scale*ells, measured_lengths_adj/np.pi, color='tab:blue', label='adj')
    ax.errorbar(scale*ells, measured_lengths_adj/np.pi, var_lengths_adj/np.pi, capsize=2,zorder=-1)
    ax.scatter(scale*ells, measured_lengths_cotan, color='tab:red', label='cotan')
    ax.errorbar(scale*ells, measured_lengths_cotan, var_lengths_cotan, color='tab:red',capsize=2,zorder=0)
    plt.gca().set_box_aspect(1)
    #plt.ylim([0, 200])
    plt.xlabel(r'$\ell (\mu m)$')
    plt.ylabel(r'Measured length $(\mu m)$')
    plt.legend(frameon=False)

    from scipy.optimize import curve_fit

    def line_func(x, a):
        return a*x

    popt, pcov = curve_fit(line_func, scale*ells, measured_lengths_adj)
    print(popt)
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plt.plot(scale*ells, scale*ells, color='k', linestyle='--', label=r'$\ell$')
    ax.scatter(scale*ells, measured_lengths_adj/popt[0], color='tab:blue', label='adj')
    ax.errorbar(scale*ells, measured_lengths_adj/popt[0], var_lengths_adj/popt[0], capsize=2,zorder=-1)
    ax.scatter(scale*ells, measured_lengths_cotan, color='tab:red', label='cotan')
    ax.errorbar(scale*ells, measured_lengths_cotan, var_lengths_cotan, color='tab:red',capsize=2,zorder=0)
    plt.gca().set_box_aspect(1)
    #plt.ylim([0, 200])
    plt.xlabel(r'$\ell (\mu m)$')
    plt.ylabel(r'Measured length $(\mu m)$')
    plt.legend(frameon=False)
    plt.savefig('SH_scale.pdf')


    ### variable ell for diffusion - measure correlation length
    px_to_um = 0.12
    ells = np.linspace(0.5, 10, 10)
    measured_lengths_adj = np.zeros_like(ells)
    measured_lengths_cotan = np.zeros_like(ells)

    var_lengths_adj = np.zeros_like(ells)
    var_lengths_cotan = np.zeros_like(ells) 


    M0 = np.mean(gpy.massmatrix(vertices, concave_hull, type='barycentric').diagonal())
    scale = np.sqrt(M0)*px_to_um

    N_reps = 32

    u_res = np.zeros((vertices.shape[0], len(ells), N_reps))


    len_scale = 1/np.sqrt(wm[1:])
    for i, ell in enumerate(ells):
        vals_adj = []
        vals_cotan = []
        for n in range(N_reps):
            u = diffusion_simulation(vertices, concave_hull, ell=ell)
            u_res[:,i,n] = u[-1]

            # measure correlation length
            m_adj, kscale = measure_lengthscale_adj(u.T, areas_file, graph_eigenvector_mat, eigenvalue_file, type='adj')
            m_cotan = measure_lengthscale_cotan(u.T, wm, vm) #np.interp(kscale, np.arange(len(len_scale)), len_scale)
            vals_adj.append(m_adj)
            vals_cotan.append(m_cotan)
        measured_lengths_adj[i] = np.mean(np.array(vals_adj))
        measured_lengths_cotan[i] = np.mean(np.array(vals_cotan))
        var_lengths_adj[i] = np.std(np.array(vals_adj))
        var_lengths_cotan[i] = np.std(np.array(vals_cotan))
    np.savez('data_variable_ell.npz', u_res=u_res, measured_lengths_adj=measured_lengths_adj, measured_lengths_cotan=measured_lengths_cotan,
                                                                    var_lengths_adj=var_lengths_adj, var_lengths_cotan=var_lengths_cotan)
    plot_power_spectrum('variable_ell_diffusion_spectra', eigenvector_mat, u_res[:,:,0], type='cotan')

    plot_power_spectrum_doubleX_reps('variable_ell_reps_diffusion_doublex', wm, vm, u_res, type='cotan')
    popt = [3.16523125]

    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plt.plot(2*scale*ells,2*scale*ells, color='k', linestyle='--', label=r'$2\ell$')
    ax.scatter(2*scale*ells, 2*measured_lengths_adj/popt[0], color='tab:blue', label='adj')
    ax.errorbar(2*scale*ells, 2*measured_lengths_adj/popt[0],var_lengths_adj/popt[0], capsize=2,zorder=-1)
    ax.scatter(2*scale*ells, 2*measured_lengths_cotan, color='tab:red', label='cotan')
    ax.errorbar(2*scale*ells, 2*measured_lengths_cotan, 2*var_lengths_cotan, color='tab:red',capsize=2,zorder=0)
    plt.gca().set_box_aspect(1)
    plt.xlabel(r'$2\ell (\mu m)$')
    plt.ylabel(r'Measured length $(\mu m)$')
    plt.legend(frameon=False)

    
