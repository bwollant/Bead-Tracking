import numpy as np
import multiprocessing as mp
import sys,os
import skimage.filters as filt
from radialcenter import *
from functools import partial
import scipy.interpolate as interp
import scipy.signal as sig
from PIL import Image
from scipy import stats
import json

# necessary to add cwd to path when script run 
# by slurm (since it executes a copy)
sys.path.append(os.getcwd()) 

try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = mp.cpu_count()
print ('num cpus available:', ncpus)

class BigMovie:
    #iterator for large tiff movies.
    def __init__(self,filepath):
        self.image = Image.open(filepath)
        self.current = 0
        self.max = np.inf
        self.filepath = filepath
    def __iter__(self):
        self.current=0
        return self
    def __next__(self):
        try:
            if self.current>self.max:
                raise StopIteration
            self.current += 1
            self.image.seek(self.current-1)
            return np.asarray(self.image)
        except EOFError:
            raise StopIteration

    def next(self):
        return self.__next__()
    def reset(self):
        self.current = 0

def find_center_blurred(image, sigma=11):
    newim = filt.gaussian(image,sigma=sigma)
    return radialcenter(newim)
def find_centers(bead_movie,pool,sigma=2,maxframes=np.inf):
    ''' 
    given a pool, find the centers of each frame in a movie.    
    '''
    fcb = partial(find_center_blurred, sigma=sigma)
    bead_movie.max = maxframes
    centers = pool.map(fcb,bead_movie)
    return np.array(centers)
def max_by_quadratic_fit(x,arr):
    coeffs= np.polyfit(x, arr,2)
    a = coeffs[0]
    b = coeffs[1]
    return -b/2/a
def gaussfilt(arr,sigma=2):
    g = sig.windows.gaussian(20,sigma)
    g = g/np.sum(g)
    return sig.convolve(arr,g,mode='same')


def radial_profile(dataandcenter,min_r=0, max_r=50, verbose=False):
    '''
    https://stackoverflow.com/questions/21242011/
    '''
    data,center = dataandcenter
    y, x = np.indices((data.shape))
    r = 1.5*np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    if verbose:
        print(r)
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    rp = radialprofile[min_r:max_r] 
    rp = rp/np.sum(rp)
    return rp
def padfft(im,power=256):
    impad = np.pad(im,((0,power-im.shape[0]),(0,power-im.shape[1])),mode='mean')
    square_shape = min(im.shape)
    ft = (np.abs(np.fft.fftshift(np.fft.fft2(impad)))**2)
    return ft
def radial_profile_fft(im,width=40,minwidth=0):
    raw_data = padfft(im)
    factor = 1
    center = [128.5,128.5]
    data,cross = raw_data,1
    y, x = np.indices((data.shape))
    r = (np.sqrt((x - center[0])**2 + (y - center[1])**2))
    rp, rs = np.histogram(r[(r<width)*(r>minwidth)],bins=int(factor*width),weights=raw_data[(r<width)*(r>minwidth)])
    #     tbin = np.bincount(r.ravel(), raw_data.ravel())
    #     nr = np.bincount(r.ravel())
    #     radialprofile = tbin / nr
    #     rp = radialprofile[1:50] 
    rp = rp
    return rp
def find_radial_profiles(bead_movie,centers, pool, min_r=0, max_r=50, fft=False):
    '''
    given a pool, a bead movie, and a list of centers, get all the radial profiles
    '''
    if fft:
        mapper = partial(radial_profile_fft)
        rps = pool.map(mapper, bead_movie)
    else:
        movie_and_centers = zip(bead_movie,centers)
        mapper = partial(radial_profile, min_r=min_r, max_r=max_r)
        rps = pool.map(mapper,movie_and_centers)
    return rps

# diffraction: use a LUT to fit the z position
def find_z(lut,rp, corr_curv=False):
    #chi-square distance between radial profiles across the LUT.
    #return an interpolated index for the bead.

    # dists = list(map(lambda p: np.linalg.norm(rp-p), lut))
    dists = list(map(lambda p: 1./(stats.pearsonr(rp,p)[0]+1), lut))
    #rough_z: smallest distance in the lut
    rough_z = np.argmin(list(dists))

    interp_dists = dists[rough_z-2:rough_z+3]
    interp_zs = range(rough_z-2, rough_z-2+len(interp_dists)) #in case the rough_z measure is close to the end of the LUT
    
    #quadratic interpolation to get min z
    coeffs = np.polyfit(interp_zs, interp_dists, 2)

    interp_z = -(coeffs[1]/(2*(coeffs[0])))
    if corr_curv:
        print('Correlation Curve! z=%.2f'%(interp_z))
        return dists
    else:
        return interp_z, dists[rough_z]
def find_zs(rps,lut,pool, corr_curv = False):
    '''
    utilize multiple processes to find all z values in a movie.
    '''
    if corr_curv:
        z_finder = partial(find_z, lut, corr_curv=True)
        dists = list(pool.map(z_finder,list(rps)))[0]
        return dists
    else:    
        z_finder = partial(find_z, lut)
        zs, dists = zip(*list(pool.map(z_finder,list(rps))))
        return zs,dists
def estimate_z_fft(testim,lut, interp=True, width=40):
    rs = []
    prof = radial_profile_fft(testim,40,minwidth=3)
    prof = np.log(prof[1:])
    for i in range(len(lut)):
        rs += [stats.pearsonr(prof[:width],lut[i][:width])[0]]
    rs = np.array(rs)
    rs = gaussfilt(rs,sigma=0.5)
    rs[np.isnan(rs)]=0
    ctr = np.nanargmax(rs)
    zs = np.array(range(max(0,ctr-5),min(len(rs),ctr+5)))
    bestz = max_by_quadratic_fit(zs,rs[zs])
    return bestz


# RICM: fit 4 interference fringes and use them to find the pattern

def analyze_one_bead(bead_movie, lut,sigma=2,min_r=0, max_r=50, fft=False, corr_curv=False):
    '''
    final pipeline to get bead centroids and z positions for a whole single-bead trace
    '''

    # make a pool of processes
    pool = mp.Pool(ncpus)
    #find all the centroids first
    if corr_curv:
        bead_movie.max=2
    centers = find_centers(bead_movie,pool,sigma=sigma)
    print ('found centers')
    if not fft:
        #compute all radial profiles
        rps = find_radial_profiles(bead_movie,centers,pool, min_r=min_r, max_r=max_r)
        print ('found radial profiles, no fft')
    else:
        rps = find_radial_profiles(bead_movie,[],pool, min_r=min_r, max_r=max_r,fft=True)
        print ('found radial profiles, fft')
    #get the zs from a lookup table
    if corr_curv:
        dists = find_zs(rps,lut,pool, corr_curv = True)
        return dists
    else:
        zs,dists = find_zs(rps,lut,pool)
        print ('found z positions', zs)
        #return two arguments: centroids in units of pixels and zs in units of lut index
        return (centers,rps,zs, dists)

def generate_LUT(bead_movie,sigma=2, min_r=0, max_r=50, fft=False):
    '''
    pipeline to get radial profiles for a single-bead zstack.
    '''

    # make a pool of processes
    pool = mp.Pool(ncpus)

    #find all the centroids first
    if fft:
        rps = find_radial_profiles(bead_movie,[],pool, min_r=min_r, max_r=max_r,fft=True)
        print('found fft radial profiles')
    else:
        centers = find_centers(bead_movie,pool,sigma=sigma)
        print ('found centers')
        #compute all radial profiles
        rps = find_radial_profiles(bead_movie,centers,pool, min_r=min_r, max_r=max_r)

        print ('found radial profiles')
    #return two arguments: centroids in units of pixels and zs in units of lut index
    return rps
def return_lut(lut_filename):
    lut_json = json.load(open(lut_filename,'r'))
    z_vals = lut_json['z_vals']
    lut = np.array(lut_json['lut'])
    min_r = (lut_json['min_r'])
    max_r = (lut_json['max_r'])
    return [z_vals, lut, min_r, max_r]

def analyze_one_bead_only_centroids(bead_movie,sigma=2,ncpus=ncpus,maxframes=np.inf):
    # make a pool of processes
    pool = mp.Pool(ncpus)

    #find all the centroids first
    centers = find_centers(bead_movie,pool,sigma=sigma,maxframes=maxframes)

    return centers

