
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
#removed numba
#from numba import jit, njit


def lsradialcenterfit(m,b,w):
    '''
    least squares solution to determine the radial symmetry center

    inputs m, b, w are defined on a grid
    w are the weights for each point
    '''
    wm2p1 = w/(m*m+1);
    sw  = np.sum((wm2p1));
    smmw = np.sum(m*m*wm2p1);
    smw  = np.sum(m*wm2p1);
    smbw = np.sum(m*b*wm2p1);
    sbw  = np.sum(b*wm2p1);
    det = smw*smw - smmw*sw;
    xc = (smbw*sw - smw*sbw)/det;    # relative to image center
    yc = (smbw*smw - smmw*sbw)/det; # relative to image center
    return [xc,yc]

def radialcenter(I):
    '''
    Python translation of radialcenter.m
    from Parthasarathy, Nat Methods. 2012 Jun 10;9(7):724-6. doi: 10.1038/nmeth.2071.

    '''

    # Ny, Nx: shape of the image
    Ny,Nx = I.shape

    #xm,ym = the coordinates of the midpoint-grid on which gradients are computed
    xm,ym = np.meshgrid(np.arange(-(Nx-1)/2.0+0.5,(Nx-1)/2.0-0.5+1,1),
    np.arange(-(Ny-1)/2.0+0.5,(Ny-1)/2.0-0.5+1,1))

    #Compute the gradient over these points, in the direction of 
    #45-degree shifted coordinates u and v
    dIdu = I[:Ny-1,1:Nx]-I[1:Ny,:Nx-1]
    dIdv = I[:Ny-1,:Nx-1]-I[1:Ny,1:Nx]
    #smoothing
    h = np.ones((3,3)) #3x3 average filter
    fdu = sig.convolve2d(dIdu,h,'same')
    fdv = sig.convolve2d(dIdv,h,'same')
    dImag2 = fdu*fdu+ fdv*fdv #squared gradient magnitude

    #slope of gradient
    m= -(fdv+fdu)/(fdu-fdv)  #deleted a minus sign here 8/21/2023 for images with dark particles- now replaced, should not matter

    #deal with special cases: if both gradients are 0, replace with unsmoothed.
    num_nans_in_m = np.sum(np.isnan(m))
    if num_nans_in_m >0:
        unsmoothm = (dIdv + dIdu)/(dIdu - dIdv)
        m[np.isnan(m)] = unsmoothm[np.isnan(m)]
    #if it's still nan, replace with 0
    num_nans_in_m = np.sum(np.isnan(m))
    if num_nans_in_m>0:
        m[np.isnan(m)] = 0
    
    #get rid of infinite terms in m
    num_infs_in_m = np.sum(np.isinf(m))
    if num_infs_in_m>0:
        m[np.isinf(m)] = 10*np.max(m[~np.isinf(m)])

    #b is the y-intercept of the line of slope m that goes through each m
    #midpoint gridpoint
    b = ym - m*xm

    #Weight by square of grad mag and inverse distance to centroid of grad intensity
    sdI2 = np.sum(dImag2)

    xcentroid = np.sum(dImag2*xm)/sdI2
    ycentroid = np.sum(dImag2*ym)/sdI2
    w = dImag2/(np.sqrt((xm-xcentroid)*(xm-xcentroid) + (ym-ycentroid)*(ym-ycentroid)))

    center = np.array(lsradialcenterfit(m,b,w))+np.array([(Nx-1)/2.,(Ny-1)/2.])

    return center
