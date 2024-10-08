# Benjamin Wollant
# August 22, 2023
# Adapted from Particle Tracking Development Jupyter Notebook
# Functions for tracking particles from a cropped image and returning list of positions for each frame


import numpy as np
from PIL import Image


from radialcenter import lsradialcenterfit, radialcenter

def find_particle_center_single_frame(I):
    '''
    Find the center of a single particle in the given image.

    Returns the particle center.
    '''
    # Find the particle center in the current frame
    max_pixel = np.max(I)
    image_adjusted = -1.0*I + max_pixel
    particle_center = radialcenter(image_adjusted)

    return particle_center


def find_particle_centers_stack(stack):

    num_frames, height, width = stack.shape

    # Array to store the particle centers for each frame
    particle_centers = np.zeros((num_frames, 2))

    for frame in range(num_frames):
        # Find the particle center in the current frame
        particle_center = find_particle_center_single_frame(stack[frame])

        # Append the particle center to the list
        particle_centers[frame] = particle_center

    return particle_centers


def find_particle_centers_in_movie(movie):
    """
    Find the particle center in each frame of a multi-frame TIFF movie.

    Parameters:
    - movie (BigMovie): An instance of the BigMovie class for handling multi-frame TIFFs.

    Returns:
    - particle_centers (ndarray): A 2D array of shape (num_frames, 2), where each row
                                  represents (y, x) center of the particle in each frame.
    """
    # List to store particle centers for each frame
    particle_centers = []

    # Iterate through each frame in the movie using BigMovie iterator
    for frame_index, frame in enumerate(movie):
        # Find the particle center in the current frame
        center = find_particle_center_single_frame(frame)
        particle_centers.append(center)

        print(f"Processed frame {frame_index + 1}")

    # Convert the list to a numpy array for further processing
    return np.array(particle_centers)


class BigMovie:
    #iterator for large tiff movies.
    def __init__(self, filepath):
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