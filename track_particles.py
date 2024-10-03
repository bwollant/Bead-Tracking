# Benjamin Wollant
# August 22, 2023
# Adapted from Particle Tracking Development Jupyter Notebook
# Functions for tracking particles from a cropped image and returning list of positions for each frame


import numpy as np
import tifffile

from radialcenter import lsradialcenterfit, radialcenter
from bead_analysis import find_center_blurred

def find_particle_center_single_frame(I):
    '''
    Find the center of a single particle in the given image.

    Returns the particle center.
    '''
    # Find the particle center in the current frame
    max_pixel = np.max(I)
    #print(I.dtype)
    image_adjusted = -1.0*I + max_pixel
    particle_center = radialcenter(image_adjusted)

    return particle_center

def find_particle_centers_stack(stack):

    num_frames, height, width = stack.shape

    # List to store the particle centers for each frame
    particle_centers_list = np.zeros((num_frames,2))

    for frame in range(num_frames):
        # Find the particle center in the current frame
        particle_center = find_particle_center_single_frame(stack[frame])

        # Append the particle center to the list
        particle_centers_list[frame] = particle_center

    return particle_centers_list