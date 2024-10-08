# Benjamin Wollant
# August 22, 2023
# Adapted from Particle Localization Development Jupyter Notebook
# Function for identifying particles from image and returning list of identified particles and various properties

import cv2
from skimage import measure
import numpy as np
from plantcv import plantcv as pcv


def identify_particles(image, min_particle_area=10, max_particle_area=40, k_size=11):
    # Note here image has replaced image_path

    # Correct illumination in the image
    pcv.params.debug = "None"
    corrected_image = pcv.transform.nonuniform_illumination(image, ksize=k_size)  # ksize must be odd

    # Perform Otsu's thresholding to convert the image to binary
    _, binary_image = cv2.threshold(corrected_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to clean up the binary image
    kernel = np.ones((2, 2), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Use scikit-image to label the connected components
    labeled_image = measure.label(cleaned_image)

    # Get the properties of the labeled regions
    properties = measure.regionprops(labeled_image)

    # Filter particles based on area
    identified_particles = [prop for prop in properties if min_particle_area <= prop.area <= max_particle_area]
    # Fields of identified particles
    # area: Area of the labeled region in pixels.
    # centroid: Centroid (center of mass) coordinates (y, x) of the labeled region.
    # bbox: Bounding box coordinates (min_row, min_col, max_row, max_col) of the labeled region.
    # perimeter: Perimeter of the labeled region.
    # eccentricity: Eccentricity of the labeled region.
    # major_axis_length: Length of the major axis of the ellipse that has the same normalized second central
    # moments as the labeled region.
    # minor_axis_length: Length of the minor axis of the ellipse that has the same normalized second central
    # moments as the labeled region.

    return identified_particles
