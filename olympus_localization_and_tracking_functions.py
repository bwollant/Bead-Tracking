import cv2
from skimage import measure
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd

from scipy.optimize import curve_fit, least_squares

from joblib import Parallel, delayed
import time

from radialcenter import *
import re

def filter_close_particles(particles: List[Any], min_distance: float = 20.0) -> List[Any]:
    """
    Filter out particles that are too close to each other based on a given minimum distance.

    Parameters:
    - particles: List of particle objects with a `centroid` attribute.
    - min_distance: Minimum allowable distance between particles.

    Returns:
    - filtered_particles: List of particles with close particles removed.
    """
    if not particles:
        return []

    # Extract the (x, y) coordinates from each particle's centroid
    centroids = np.array([p.centroid for p in particles])

    # Compute the pairwise distances between centroids
    dist_matrix = squareform(pdist(centroids))  # Create a symmetric distance matrix

    # Create a boolean mask to mark particles to keep
    to_keep = np.ones(len(particles), dtype=bool)

    # Iterate through the distance matrix and mark close particles for removal
    for i in range(len(particles)):
        # Mark particles that are too close to the current particle
        close_particles = (dist_matrix[i] < min_distance) & (dist_matrix[i] > 0)
        to_keep[np.where(close_particles)[0]] = False

    # Return the filtered list of particles
    filtered_particles = [p for i, p in enumerate(particles) if to_keep[i]]
    return filtered_particles


def fit_2d_gaussian(image):
    """Fits a 2D Gaussian function to the given image."""

    def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, offset):
        x, y = xy
        g = offset + amplitude * np.exp(-(((x - xo) ** 2) / (2 * sigma_x ** 2) + ((y - yo) ** 2) / (2 * sigma_y ** 2)))
        return g.ravel()

    # Normalize image for numerical stability
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    elif min_val == max_val:
        return None, False # fit is bound to fail
        # image = np.zeros_like(image)

    # Generate x, y indices
    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)

    # Improved initial guess using brightest pixel
    x_peak, y_peak = np.unravel_index(np.argmax(image), image.shape)
    initial_guess = (np.max(image), x_peak, y_peak, 5, 5, np.min(image))

    # Apply bounds to stabilize fitting
    try:
        params, _ = curve_fit(
            twoD_Gaussian, (x, y), image.ravel(),
            p0=initial_guess,
            bounds=([0, 0, 0, 1, 1, 0], [np.inf, image.shape[1], image.shape[0], 20, 20, np.max(image)])
        )

        return params, True
    except RuntimeError as e:
        print(f"Gaussian fit failed: {e}")
        return None, False  # Return None, False if fitting fails

def fit_radial_gaussian_ring(image):
    """
    Fits a radial Gaussian ring model to the given 2D image.
    Returns fitted parameters and success flag:
    (x0, y0, A_core, sigma_core, A_ring, sigma_ring, B), True/False
    """

    def radial_gaussian_ring_model(params, x, y, image):
        x0, y0, A_core, sigma_core, A_ring, sigma_ring, B = params
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        model = (
            A_core * np.exp(-r ** 2 / (2 * sigma_core ** 2)) -
            A_ring * np.exp(-r ** 2 / (2 * sigma_ring ** 2)) +
            B
        )
        return (model - image).ravel()

    # Normalize image for numerical stability
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        image = (image - min_val) / (max_val - min_val)
    elif min_val == max_val:
        return None, False  # Flat image — cannot fit

    # Generate coordinate grid
    h, w = image.shape
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))

    # Initial guesses
    y0_init, x0_init = (w / 2, h / 2)  # np.unravel_index(np.argmax(image), image.shape)
    A_core_init = 0.75 #image.min() - np.median(image)  # image.max() - np.median(image)
    sigma_core_init = 1.5
    A_ring_init = 0.25 #image.max() - np.median(image)  # A_core_init / 2
    sigma_ring_init = 4.0
    B_init = np.median(image)

    initial_params = [x0_init, y0_init, A_core_init, sigma_core_init, A_ring_init, sigma_ring_init, B_init]

    try:
        result = least_squares(
            radial_gaussian_ring_model,
            initial_params,
            args=(x_grid, y_grid, image),
            bounds=(
                [0, 0, -np.inf, 0.5, -np.inf, 4.0, 0],
                [w, h, np.inf, 2.5, np.inf, 15, np.inf]
            )
        )
        return result.x, True

    except Exception as e:
        print(f"Fit failed: {e}")
        return None, False

def fit_particle(image):
    params, fit_ok = fit_2d_gaussian(image)
    if fit_ok:
        _, _, _, sigma_x, sigma_y, _ = params
        # check fit parameters, cutoff is for 2.8 um beads, not fiducials
        fit_ok = (
                    sigma_x < 3 and
                    sigma_y < 3
        )
    return params, fit_ok

def fit_particle_radial(image):
    center = radialcenter(image)

    # 1) check for any NaNs
    if np.isnan(center).any():
        fit_ok = False

    # 2) check for Infs too
    elif not np.all(np.isfinite(center)):
        fit_ok = False

    # if you get here, center[0] and center[1] are both real numbers
    else:
        fit_ok = True

    return center, fit_ok

def fit_fiducial_particle(image):
    # for 5 um fiducials
    params, fit_ok = fit_radial_gaussian_ring(image)
    # if fit_ok:
    #     _, _, _, sigma_x, sigma_y, _ = params
    #     # check fit parameters, cutoff is for 2.8 um beads, not fiducials
    #     fit_ok = (
    #                 sigma_x < 3 and
    #                 sigma_y < 3
    #     )
    return params, fit_ok

# Function to process a single particle
def track_particle(index, particle, images, NS_length_pixel, fit_method):
    """Tracks a single particle by fitting its position in the cropped stack."""
    # Round and extract particle position
    x_cent = np.round(particle["Localized X Position"]).astype(int)
    y_cent = np.round(particle["Localized Y Position"]).astype(int)



    crop_size = np.round(1.5 * NS_length_pixel).astype(int)

    # Crop stack around the particle
    y_start = max(0, y_cent - crop_size)
    y_end = min(images.shape[1], y_cent + crop_size)
    x_start = max(0, x_cent - crop_size)
    x_end = min(images.shape[2], x_cent + crop_size)

    cropped_stack = images[:, y_start:y_end, x_start:x_end]

    # Initialize 1D arrays for tracked centers
    stack_centers_x = np.full(len(cropped_stack), np.nan)
    stack_centers_y = np.full(len(cropped_stack), np.nan)
    fit_quality = np.zeros(len(cropped_stack), dtype=bool)

    if fit_method == "gaussian":
        # Fit Gaussian over frames for regular particle
        for i, frame in enumerate(cropped_stack):
            params, fit_ok = fit_particle(image=frame)

            if fit_ok and params is not None:
                _, x0, y0, _, _, _ = params
                stack_centers_x[i] = x0 + x_cent - crop_size
                stack_centers_y[i] = y0 + y_cent - crop_size
                fit_quality[i] = True
            else:
                # NaNs already set by default
                fit_quality[i] = False

    elif fit_method == "radial":
        # Fit radial symmetry centers over frames for regular particle
        for i, frame in enumerate(cropped_stack):
            center, fit_ok = fit_particle_radial(image=frame)

            if fit_ok and center is not None:
                x0, y0 = center
                stack_centers_x[i] = x0 + x_cent - crop_size
                stack_centers_y[i] = y0 + y_cent - crop_size
                fit_quality[i] = True
            else:
                # NaNs already set by default
                fit_quality[i] = False

    return index, stack_centers_x, stack_centers_y, fit_quality  # Return results

def track_particle_drift_correction(index, particle, images, NS_length_pixel, A, offset_y):
    """Tracks a single particle by fitting its position in the cropped stack."""
    # Round and extract particle position
    x_cent = particle["Localized X Position"]  # np.round(particle["Localized X Position"]).astype(int)
    y_cent = particle["Localized Y Position"]  # np.round(particle["Localized Y Position"]).astype(int)

    P = np.array([[x_cent],
                  [y_cent]])

    x_cent_corrected, y_cent_corrected = apply_rigid_transform(A, P).flatten()

    if particle["Particle Number"] > 0:  # if not a fiducial particle, apply offset for reverse flow
        y_cent_corrected = y_cent_corrected + offset_y

    x_cent_corrected = np.round(x_cent_corrected).astype(int)
    y_cent_corrected = np.round(y_cent_corrected).astype(int)

    crop_size = np.round(1.5 * NS_length_pixel).astype(int)

    # Crop stack around the particle
    y_start = max(0, y_cent_corrected - crop_size)
    y_end = min(images.shape[1], y_cent_corrected + crop_size)
    x_start = max(0, x_cent_corrected - crop_size)
    x_end = min(images.shape[2], x_cent_corrected + crop_size)

    cropped_stack = images[:, y_start:y_end, x_start:x_end]

    # Initialize 1D arrays for tracked centers
    stack_centers_x = np.full(len(cropped_stack), np.nan)
    stack_centers_y = np.full(len(cropped_stack), np.nan)
    fit_quality = np.zeros(len(cropped_stack), dtype=bool)

    # Fit Gaussian over frames for regular particle
    for i, frame in enumerate(cropped_stack):
        params, fit_ok = fit_particle(image=frame)

        if fit_ok and params is not None:
            _, x0, y0, _, _, _ = params
            stack_centers_x[i] = x0 + x_cent_corrected - crop_size
            stack_centers_y[i] = y0 + y_cent_corrected - crop_size
            fit_quality[i] = True
        else:
            # NaNs already set by default
            fit_quality[i] = False


    return index, stack_centers_x, stack_centers_y, fit_quality  # Return results



def track_video(video_num: int, image_path: str, background_image_path: str, particles_df, scope: str = "nikon", magnification: int = 10,
                parallelize: bool = False, subtract_background: bool = False, fit_method: str = "gaussian"):
    start = time.time()

    ##--Specify magnification and pixel size --##
    NS_length_nm = 2300  # nm

    if fit_method != "gaussian" and fit_method != "radial":
        print("improper fit method specified. Must be 'gaussian' or 'radial'.")
        return

    if scope == "olympus":
        if magnification == 10:
            pixel_nm = 650
        elif magnification == 20:
            pixel_nm = 325
        else:
            print("error: magnification not properly specified!")
            return
    elif scope == "nikon":
        if magnification == 10:
            pixel_nm = 1100
        elif magnification == 15:
            pixel_nm = 730
        elif magnification == 20:
            pixel_nm = 550
        else:
            print("error: magnification not properly specified!")
            return
    else:
        print("scope name not recognized")
    NS_length_pixel = np.floor(NS_length_nm / pixel_nm).astype(int)

    # Read video for analysis, invert and substract baseline
    images = tifffile.imread(image_path)

    # Perform flat field correction for non-uniform illumination
    if subtract_background:
        background_image = tifffile.imread(
            background_image_path)  # may need to subset to first image or average if stack

        if len(background_image.shape) == 2:
            images = flat_field_correction(image_stack=images, flat_field=background_image)

        elif len(background_image.shape) == 3:
            background_image_mean = np.mean(background_image, axis=0)
            background_image_mean = np.round(background_image_mean).astype(np.uint16)  # Round & convert to uint16
            images = flat_field_correction(image_stack=images, flat_field=background_image_mean)

    images = ((2 ** 16 - 1) - images)
    images = images - np.min(images) + 1

    # Create new column names for X and Y positions
    x_col_title = f"({video_num}) x positions"
    y_col_title = f"({video_num}) y positions"

    # Initialize empty lists in the DataFrame for results
    particles_df[x_col_title] = [[] for _ in range(len(particles_df))]
    particles_df[y_col_title] = [[] for _ in range(len(particles_df))]

    if parallelize:
        # Parallel execution for all particles
        num_cores = -1  # Use all available CPU cores
        results = Parallel(n_jobs=num_cores)(
            delayed(track_particle)(index, row, images, NS_length_pixel, fit_method) for index, row in particles_df.iterrows()
        )

        # Store results in DataFrame with separate X and Y columns
        for index, x_positions, y_positions, _ in results:
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions

    else:
        for index, row in particles_df.iterrows():
            _, x_positions, y_positions, _ = track_particle(index, row, images, NS_length_pixel, fit_method)
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions
    # End timing
    end = time.time()
    elapsed = end - start

    # Display the elapsed time
    print(f"Total execution time for track_video: {elapsed:.2f} seconds")


def track_video_drift_correction(video_num: int, ref_video_num: int, image_path: str, ref_image_path: str, background_image_path: str,
                particles_df, scope: str = "nikon", magnification: int = 10, parallelize: bool = False, subtract_background: bool = False,
                offset_y: int = 0):
    start = time.time()

    ##--Specify magnification and pixel size --##
    NS_length_nm = 2300  # nm

    if scope == "olympus":
        if magnification == 10:
            pixel_nm = 650
        elif magnification == 20:
            pixel_nm = 325
        else:
            print("error: magnification not properly specified!")
            return
    elif scope == "nikon":
        if magnification == 10:
            pixel_nm = 1100
        elif magnification == 15:
            pixel_nm = 730
        elif magnification == 20:
            pixel_nm = 550
        else:
            print("error: magnification not properly specified!")
            return
    else:
        print("scope name not recognized")
    NS_length_pixel = np.floor(NS_length_nm / pixel_nm).astype(int)

    # Read video for analysis, invert and substract baseline
    images = tifffile.imread(image_path)

    # Perform flat field correction for non-uniform illumination
    if subtract_background:
        background_image = tifffile.imread(
            background_image_path)  # may need to subset to first image or average if stack

        if len(background_image.shape) == 2:
            images = flat_field_correction(image_stack=images, flat_field=background_image)

        elif len(background_image.shape) == 3:
            background_image_mean = np.mean(background_image, axis=0)
            background_image_mean = np.round(background_image_mean).astype(np.uint16)  # Round & convert to uint16
            images = flat_field_correction(image_stack=images, flat_field=background_image_mean)

    images = ((2 ** 16 - 1) - images)
    images = images - np.min(images) + 1

    # Create new column names for X and Y positions
    x_col_title = f"({video_num}) x positions"
    y_col_title = f"({video_num}) y positions"

    # Initialize empty lists in the DataFrame for results
    particles_df[x_col_title] = [[] for _ in range(len(particles_df))]
    particles_df[y_col_title] = [[] for _ in range(len(particles_df))]

    # Create fiducial_particles_df
    fiducial_particles_df = particles_df.copy()
    fiducial_particles_df = fiducial_particles_df[fiducial_particles_df["Particle Number"] < 0]
    print("Number of fiducials used for drift correction: ", len(fiducial_particles_df))

    # Track reference video for fiducials
    track_video(video_num=ref_video_num, image_path=ref_image_path, background_image_path=background_image_path,
                particles_df=fiducial_particles_df, scope=scope, magnification=magnification, parallelize=parallelize,
                subtract_background=subtract_background)
    add_mean_position_cols(df=fiducial_particles_df, vid_num=ref_video_num)

    # Track video for fiducials
    track_video(video_num=video_num, image_path=image_path, background_image_path=background_image_path,
                particles_df=fiducial_particles_df, scope=scope, magnification=magnification, parallelize=parallelize,
                subtract_background=subtract_background)
    add_mean_position_cols(df=fiducial_particles_df, vid_num=video_num)


    # Compute rigid body transformation A to apply to Localized X Position and Localized Y Position to re-center crops
    # extract mean positons of all fiducials in reference video
    P1 = extract_mean_fiducial_positions(fiducial_particles_df, vid_num=ref_video_num)

    # extract the mean positions of all fiducial in the video
    P2 = extract_mean_fiducial_positions(fiducial_particles_df, vid_num=video_num)

    assert P1.shape == P2.shape

    # P1 and P2 should both be 2×n arrays
    mask_finite = np.isfinite(P1).all(axis=0) & np.isfinite(P2).all(axis=0)
    if not mask_finite.all():
        # drop any columns (landmarks) that aren’t fully finite
        P1 = P1[:, mask_finite]
        P2 = P2[:, mask_finite]

    # now make sure you still have enough points
    if P1.shape[1] < 2:
        raise ValueError(f"Not enough valid fiducials after dropping nonfinite data: only {P1.shape[1]} left.")

    # define A for rigid body transformation to correct for drift
    A = rigid_transform_least_squares(P2, P1)

    if parallelize:
        # Parallel execution for all particles
        num_cores = -1  # Use all available CPU cores
        results = Parallel(n_jobs=num_cores)(
            delayed(track_particle_drift_correction)(index, row, images, NS_length_pixel, A, offset_y) for index, row in particles_df.iterrows()
        )

        # Store results in DataFrame with separate X and Y columns
        for index, x_positions, y_positions, _ in results:
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions

    else:
        for index, row in particles_df.iterrows():
            _, x_positions, y_positions, _ = track_particle_drift_correction(index, row, images, NS_length_pixel, A, offset_y)
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions
    # End timing
    end = time.time()
    elapsed = end - start

    # Display the elapsed time
    print(f"Total execution time for track_video_drift_corrected: {elapsed:.2f} seconds")


# Functions to process particle traces
def compute_mean_position(position_list):
    """Computes the mean position from a list of X or Y positions."""
    position_array = np.array(position_list)
    return np.mean(position_array).tolist() if len(position_array) > 0 else np.nan


# Function to add mean position columns for X and Y
def add_mean_position_cols(df, vid_num: int):
    """Adds mean position columns for X and Y."""
    x_col = f"({vid_num}) x positions"
    y_col = f"({vid_num}) y positions"
    df[f"({vid_num}) mean x position"] = df[x_col].apply(compute_mean_position)
    df[f"({vid_num}) mean y position"] = df[y_col].apply(compute_mean_position)


# Updated function to compute standard deviation for separate X and Y columns
def compute_stdev_position(position_list):
    """Computes the standard deviation of a list of X or Y positions."""
    position_array = np.array(position_list)
    return np.std(position_array).tolist() if len(position_array) > 0 else np.nan


def add_stdev_position_cols(df, vid_num: int):
    """Adds standard deviation position columns for X and Y."""
    x_col = f"({vid_num}) x positions"
    y_col = f"({vid_num}) y positions"
    df[f"({vid_num}) stdev x position"] = df[x_col].apply(compute_stdev_position)
    df[f"({vid_num}) stdev y position"] = df[y_col].apply(compute_stdev_position)


# def compute_msd_from_trajectory(xy_data: np.ndarray, time_step: float, pixel_size: float):
#     """
#     Compute Mean Squared Displacement (MSD) from an array of [x, y] positions over time.
#
#     Parameters:
#     - xy_data: A NumPy array of shape (N, 2), where N is the number of time steps.
#     - time_step: Time interval between frames (in seconds).
#     - pixel_size: Size of a pixel (in micrometers).
#
#     Returns:
#     - time_lags: NumPy array of time lags (in seconds).
#     - msd: NumPy array of mean squared displacements (in µm²).
#     """
#     if len(xy_data) < 2:
#         return np.array([]), np.array([])  # Not enough data to compute MSD
#
#     # Convert positions to micrometers
#     x = xy_data[:, 0] * pixel_size
#     y = xy_data[:, 1] * pixel_size
#     N = len(x)
#
#     msd = []
#     time_lags = []
#
#     for lag in range(1, N // 2):  # Use up to half the trajectory to avoid noise
#         displacements = (x[lag:] - x[:-lag]) ** 2 + (y[lag:] - y[:-lag]) ** 2
#         msd.append(np.mean(displacements))
#         time_lags.append(lag * time_step)  # Calculate the time difference for each lag
#
#     return np.array(time_lags), np.array(msd)

# def fit_diffusion_coefficient(time_lags: np.ndarray, msd_values: np.ndarray):
#     """
#     Fit the MSD to extract the diffusion coefficient (D).
#
#     Parameters:
#     - time_lags: NumPy array of time lags (in seconds).
#     - msd_values: NumPy array of mean squared displacements (in µm²).
#
#     Returns:
#     - D: Diffusion coefficient (in µm²/s).
#     """
#     if len(time_lags) < 2:
#         raise ValueError("Insufficient data to fit diffusion coefficient.")
#
#     def linear_fit(tau, D):
#         return 4 * D * tau  # Linear relationship for MSD vs time lag
#
#     popt, _ = curve_fit(linear_fit, time_lags, msd_values)
#     D = popt[0]
#     return D

# def add_diffusion_coefficient_cols(df, time_step, magnification):
#     """
#     Adds diffusion coefficient (D) columns to the DataFrame for each row (particle) based on its (x, y) positions.
#
#     Parameters:
#     - df: Pandas DataFrame containing particle tracking data.
#     - time_step: Time interval between frames (in seconds).
#     - pixel_size: Size of a pixel in micrometers (µm).
#
#     Adds a new column named "(number) D" for each particle.
#     """
#     if magnification == 10:
#         pixel_size = 0.650  # um
#     elif magnification == 20:
#         pixel_size = 0.325  # um
#     else:
#         print("error: magnification not properly specified!")
#         return
#
#     for col in df.columns:
#         match = re.match(r"\((\d+)\) x positions", col)  # Match X position columns
#         if match:
#             number = match.group(1)
#             y_col = f"({number}) y positions"
#             if y_col in df.columns:  # Ensure corresponding Y column exists
#                 D_col_title = f"({number}) D"
#                 df[D_col_title] = np.nan  # Initialize the new column with NaN values
#
#                 for index, row in df.iterrows():
#                     x_positions = row[col]
#                     y_positions = row[y_col]
#
#                     # Ensure x_positions and y_positions are numpy arrays with enough data
#                     if isinstance(x_positions, np.ndarray) and isinstance(y_positions, np.ndarray):
#                         if len(x_positions) > 2 and len(y_positions) > 2:
#                             xy_data = np.column_stack((x_positions, y_positions))  # Combine x and y into a 2D array
#                             time_lags, msd_values = compute_msd_from_trajectory(xy_data, time_step, pixel_size)
#
#                             if len(time_lags) > 1 and len(msd_values) > 1:
#                                 try:
#                                     D = fit_diffusion_coefficient(time_lags, msd_values)
#                                     df.at[index, D_col_title] = D  # Assign the computed D to the new column
#                                 except (RuntimeError, ValueError) as e:
#                                     print(f"Fit failed for particle at index {index}: {e}")

def add_video_average_corrected_mean_position_cols(particles_df, vid_num: int = 2, ref_vid_num: int = 1, defocus: bool = False):
    """
            Adds video average corrected mean position columns
                    """
    x_col_corrected = f"({vid_num}) mean x position va corrected"
    y_col_corrected = f"({vid_num}) mean y position va corrected"

    # Initialize nan cols for all corrected mean positions
    # particles_df.loc[:, x_col_corrected] = np.nan
    # particles_df.loc[:, y_col_corrected] = np.nan

    # extract mean positons of all fiducials in reference video
    P1 = extract_mean_fiducial_positions(particles_df, vid_num=ref_vid_num)

    # extract the mean positions of all fiducial in the video
    P2 = extract_mean_fiducial_positions(particles_df, vid_num=vid_num)

    assert P1.shape == P2.shape

    # extract the position of all beads in the video
    P3 = extract_mean_bead_positions(particles_df, vid_num=vid_num)

    # define A for rigid body transformation/similarity transform to correct for drift, defocus
    if defocus:
        # A = similarity_transform_least_squares(P1, P2)
        A, defocus, center = similarity_transform_least_squares_with_focus(P1, P2)
        P3_corrected = apply_similarity_transform_with_defocus(A, defocus, center, P3)
    else:
        # P1 and P2 should both be 2×n arrays
        mask_finite = np.isfinite(P1).all(axis=0) & np.isfinite(P2).all(axis=0)
        if not mask_finite.all():
            # drop any columns (landmarks) that aren’t fully finite
            P1 = P1[:, mask_finite]
            P2 = P2[:, mask_finite]

        # now make sure you still have enough points
        if P1.shape[1] < 2:
            raise ValueError(f"Not enough valid fiducials after dropping nonfinite data: only {P1.shape[1]} left.")

        A = rigid_transform_least_squares(P1, P2)
        P3_corrected = apply_rigid_transform(A, P3)


    particles_df[x_col_corrected] = P3_corrected[0, :]  # First row → x values
    particles_df[y_col_corrected] = P3_corrected[1, :]  # Second row → y values

    P2_corr = apply_rigid_transform(A, P2)
    print("P1 centroid:", P1.mean(axis=1))
    print("P2 centroid:", P2.mean(axis=1))
    print("P2_corr centroid:", P2_corr.mean(axis=1))
    print("Mean residual:", (P1 - P2_corr).mean(axis=1))

def extract_mean_fiducial_positions(particles_df, vid_num: int=1):
    x_col = f"({vid_num}) mean x position"
    y_col = f"({vid_num}) mean y position"

    # Select only fiducials and sort by their ID
    fid = (
        particles_df
        .loc[particles_df["Particle Number"] < 0,
             ["Particle Number", x_col, y_col]]
        .sort_values("Particle Number")
    )

    # Now the rows are in ascending Particle Number order
    P = fid[[x_col, y_col]].to_numpy().T  # shape 2×n
    return P

def extract_mean_bead_positions(particles_df, vid_num: int = 1):
    """
    Extracts bead positions for all beads rows in particles dataframe
    """
    x_col = f"({vid_num}) mean x position"
    y_col = f"({vid_num}) mean y position"
    x_list, y_list = [], []

    for index, row in particles_df.iterrows():
        x_list.append(row[x_col])
        y_list.append(row[y_col])

    P = np.array([x_list, y_list])  # shape 2×n

    return P

def extract_fiducial_positions(particles_df, x_col, y_col, frame_index = 0):
    """
    Extracts fiducial positions for fiducial bead rows in particles dataframe
    """
    x_list, y_list = [], []

    fiducial_particles_df = particles_df[particles_df["Particle Number"] < 0]
    for index, row in fiducial_particles_df.iterrows():
        if len(row[x_col]) > frame_index: #handle nan's?
            x_list.append(row[x_col][frame_index])
            y_list.append(row[y_col][frame_index])

    P = np.array([x_list, y_list])  # shape 2×n

    return P

def extract_all_bead_positions(particles_df, x_col, y_col, frame_index = 0):
    """
    Extracts positions for all bead rows in particles dataframe
    """
    x_list, y_list = [], []

    for index, row in particles_df.iterrows():
        if len(row[x_col]) > frame_index: #handle nan's?
            x_list.append(row[x_col][frame_index])
            y_list.append(row[y_col][frame_index])

    P = np.array([x_list, y_list])  # shape 2×n

    return P

def rigid_transform_least_squares(P1, P2):
    """
    Estimate rotation (theta) and translation (tx, ty) using least squares
    P1: 2 x n matrix (baseline fiducial positions, frame 1)
    P2: 2 x n matrix (current fiducial positions, frame t)
    Returns 2x3 transformation matrix A
    """

    def residuals(params, P1, P2):
        theta, tx, ty = params
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        transformed = R @ P2 + np.array([[tx], [ty]])
        return (P1 - transformed).ravel()

    # Initial guess: no rotation, no translation
    initial_params = [0.0, 0.0, 0.0]
    result = least_squares(residuals, initial_params, args=(P1, P2)) #, max_nfev=50000, ftol=1e-14, xtol=1e-14, gtol=1e-14)

    theta, tx, ty = result.x
    print('theta: ', theta, 'tx: ', tx, 'ty: ', ty)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    A = np.hstack((R, np.array([[tx], [ty]])))
    return A

def similarity_transform_least_squares(P1, P2):
    """
    Estimate scale (s), rotation (theta), and translation (tx, ty) using least squares.
    Returns 2x3 transformation matrix A.
    """

    def residuals(params, P1, P2):
        s, theta, tx, ty = params
        R = s * np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        transformed = R @ P2 + np.array([[tx], [ty]])
        return (P1 - transformed).ravel()

    # Initial guess: no scale, no rotation, no translation
    initial_params = [1.0, 0.0, 0.0, 0.0]
    result = least_squares(residuals, initial_params, args=(P1, P2))

    s, theta, tx, ty = result.x
    print(f'scale: {s}, theta: {theta}, tx: {tx}, ty: {ty}')

    R = s * np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    A = np.hstack((R, np.array([[tx], [ty]])))
    return A

def apply_rigid_transform(A, P):
    """
    Applies 2x3 rigid transform A to 2xn matrix P.
    Returns transformed 2xn matrix.
    """
    n = P.shape[1]
    P_hom = np.vstack((P, np.ones((1, n))))  # Make P 3×n
    return A @ P_hom  # Result is 2×n

def similarity_transform_least_squares_with_focus(P1, P2, image_shape=(1200,1200)):
    """
    Estimate scale (s), rotation (theta), translation (tx, ty), and defocus radial distortion.
    Returns 2x3 transformation matrix A and defocus scalar.
    """

    if image_shape is not None:
        center = np.array([[image_shape[1] / 2], [image_shape[0] / 2]])
    else:
        center = np.mean(P2, axis=1, keepdims=True)

    def residuals(params, P1, P2):
        theta, tx, ty, defocus = params
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        radial = P2 - center
        distorted = P2 + defocus * radial
        transformed = R @ distorted + np.array([[tx], [ty]])
        return (P1 - transformed).ravel()

    initial_params = [0.0, 0.0, 0.0, 0.0]
    result = least_squares(residuals, initial_params, args=(P1, P2))

    theta, tx, ty, defocus = result.x
    print(f'theta: {theta:.4f}, tx: {tx:.4f}, ty: {ty:.4f}, defocus: {defocus:.6f}')

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    A = np.hstack((R, np.array([[tx], [ty]])))
    return A, defocus, center

def apply_similarity_transform_with_defocus(A, defocus, center, P):
    """
    Applies similarity transform A with defocus to 2xn matrix P.
    Returns transformed 2xn matrix.
    """
    radial = P - center
    distorted = P + defocus * radial
    n = distorted.shape[1]
    P_hom = np.vstack((distorted, np.ones((1, n))))
    return A @ P_hom


def add_mean_position_shift_cols(df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) mean x position"
    y_col = f"({vid_num}) mean y position"
    x_ref_col = f"({ref_vid_num}) mean x position"
    y_ref_col = f"({ref_vid_num}) mean y position"

    x_shift_col = f"({ref_vid_num}) to ({vid_num}) mean x position shift"
    y_shift_col = f"({ref_vid_num}) to ({vid_num}) mean y position shift"

    displacement_col = f"({ref_vid_num}) to ({vid_num}) mean displacement"

    df[x_shift_col] = df[x_col] - df[x_ref_col]
    df[y_shift_col] = df[y_col] - df[y_ref_col]

    df[displacement_col] = np.sqrt(df[x_shift_col]**2 + df[y_shift_col]**2)



def add_va_corrected_mean_position_shift_cols(df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) mean x position va corrected"
    y_col = f"({vid_num}) mean y position va corrected"
    x_ref_col = f"({ref_vid_num}) mean x position va corrected"
    y_ref_col = f"({ref_vid_num}) mean y position va corrected"

    x_shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean x position shift"
    y_shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"

    displacement_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean displacement"

    df[x_shift_col] = df[x_col] - df[x_ref_col]
    df[y_shift_col] = df[y_col] - df[y_ref_col]

    df[displacement_col] = np.sqrt(df[x_shift_col]**2 + df[y_shift_col]**2)





def plot_single_particle_tracking(df, image_path, video_num, frame_number, particle_num, threshold_image: bool = False, threshold: float = 0.5):
    """
    Plots a given particle in a given frame with red X markers at localized position.

    Parameters:
    - df: Pandas DataFrame containing particle tracking data.
    - image_path: str of path to video tiff.
    - video_num: number of video analyzed and stored in df_particles
    - frame_number: Integer index of the frame to plot.
    - particle_number: Integer index of particle plotted
    """

    # import tiff file
    images = tifffile.imread(image_path)

    # Invert, subset first image, and compute image shape
    images = ((2 ** 16 - 1) - images)
    images = images - np.min(images) + 1

    # extract frame from image stack
    frame = images[frame_number, :, :]
    height, width = frame.shape

    # Select particle
    particle = df.loc[df['Particle Number'] == particle_num]

    # Collect x and y tracked positions for given video number
    x_col = f"({video_num}) x positions"
    y_col = f"({video_num}) y positions"

    # if x_col in particle.columns and y_col in particle.columns and len(particle[x_col]) > frame_number and len(particle[y_col])> frame_number:  # Ensure valid frame index
    x_position = particle[x_col].values[0][frame_number]
    y_position = particle[y_col].values[0][frame_number]

    if not np.isnan(x_position) and not np.isnan(y_position):
        # crop frame for display and update tracking position
        x_min = max(np.round(x_position - 10).astype(int), 0)
        x_max = min(np.round(x_position + 10).astype(int), width)
        y_min = max(np.round(y_position - 10).astype(int), 0)
        y_max = min(np.round(y_position + 10).astype(int), height)

        x_position = x_position - x_min
        y_position = y_position - y_min

        print(x_position, y_position)
        cropped = frame[y_min:y_max, x_min:x_max]

        if threshold_image:
            max_val = np.max(cropped)
            min_val = np.min(cropped)
            std_val = np.std(cropped)
            mean_val = np.mean(cropped)
            threshold = threshold*max_val

            cropped[cropped < threshold] = 0
            print(max_val, min_val, std_val, mean_val)




        # Plot the frame with particle positions
        plt.figure(figsize=(5, 5))
        plt.imshow(cropped, cmap="gray")  # Display the frame in grayscale
        plt.scatter(x_position, y_position, color='red', marker='x', s=20, linewidths=1)

        plt.title(f"Video {video_num} Frame {frame_number} with Particle {particle_num} Position")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.axis("off")
        plt.show()


    else:
        print('particle tracking failed')
        x_position = particle["Localized X Position"].values[0]
        y_position = particle["Localized Y Position"].values[0]

        # crop frame for display and update tracking position
        x_min = max(np.round(x_position - 10).astype(int), 0)
        x_max = min(np.round(x_position + 10).astype(int), width)
        y_min = max(np.round(y_position - 10).astype(int), 0)
        y_max = min(np.round(y_position + 10).astype(int), height)


        # Plot the frame with particle positions
        plt.figure(figsize=(5, 5))
        plt.imshow(frame[y_min:y_max, x_min:x_max], cmap="gray")  # Display the frame in grayscale

        plt.title(f"Video {video_num} Frame {frame_number} with Particle {particle_num} Position")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.axis("off")
        plt.show()


def plot_particles_tracking_on_frame(df, image_path, video_num, frame_number):
    """
    Plots a given frame with red X markers at all particle positions.

    Parameters:
    - df: Pandas DataFrame containing particle tracking data.
    - image_path: str of path to video tiff.
    - frame_number: Integer index of the frame to plot.
    """

    # import tiff file
    images = tifffile.imread(image_path)

    # Invert, subset first image, and compute image shape
    images = ((2 ** 16 - 1) - images)
    images = images - np.min(images) + 1

    # extract frame from image stack
    frame = images[frame_number, :, :]

    # Initialize lists for particle positions in the given frame
    x_positions = []
    y_positions = []

    # Collect x and y tracked positions for given video number
    x_col = f"({video_num}) x positions"
    y_col = f"({video_num}) y positions"

    if x_col in df.columns and y_col in df.columns and len(df[x_col][0]) > frame_number and len(
            df[y_col][0]) > frame_number:  # Ensure valid frame index
        x_positions.extend([x[frame_number] for x in df[x_col]])
        y_positions.extend([y[frame_number] for y in df[y_col]])

    # Convert to NumPy arrays for plotting
    x_positions = np.array(x_positions)
    y_positions = np.array(y_positions)

    # Draw particle number just below the box
    for i, x in enumerate(x_positions):
        if not np.isnan(x) and not np.isnan(y_positions[i]):
            pnum = df.iloc[i]["Particle Number"]
            cv2.putText(frame, str(pnum), (int(x), int(y_positions[i]) + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

    # Plot the frame with particle positions
    plt.figure(figsize=(20, 20))
    plt.imshow(frame, cmap="gray")  # Display the frame in grayscale
    plt.scatter(x_positions, y_positions, color='red', marker='x', s=5, linewidths=1)

    plt.title(f"Frame {frame_number} with Particle Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.axis("off")
    plt.show()


def plot_fiducials_across_videos(video_paths, particles_df):
    """
        Plots cropped images of each fiducial identified by localization step for each video used for analysis. The
        purpose of this function is to assist in selected fiducials for robust drift correction.

        Parameters:
        - df: Pandas DataFrame containing particle tracking data.
        - video_paths: list of str paths to video tiffs.
        """

    fiducials_df = particles_df[particles_df["Particle Number"] < 0] # subset df to only fiducials


    m = len(fiducials_df)  # number of beads
    n = len(video_paths)   # number of videos

    # Create stack of first frames of each video
    reference_frame = tifffile.imread(video_paths[0])[0,:,:]
    height, width = reference_frame.shape
    first_frame_stack = np.full((n, height, width), np.nan)

    for j, videopath in enumerate(video_paths):
        first_frame = tifffile.imread(video_paths[j])[0,:,:]
        first_frame_stack[j,:,:] = first_frame

    # Create figure to plot particles
    fig, axes = plt.subplots(m, n, figsize=(2 * n, 2 * m), sharex=True, sharey=True)

    # Handle cases when m or n is 1 (to avoid indexing errors)
    if m == 1:
        axes = np.expand_dims(axes, axis=0)
    if n == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, particle in fiducials_df.iterrows():
        particle_num = particle["Particle Number"]
        x_position = particle["Localized X Position"]
        y_position = particle["Localized Y Position"]

        for j, video_path in enumerate(video_paths):
            frame = first_frame_stack[j, :, :]
            ax = axes[i][j]

            # crop frame for display and update tracking position
            crop_size = 5
            x_min = max(np.round(x_position - crop_size).astype(int), 0)
            x_max = min(np.round(x_position + crop_size).astype(int), width)
            y_min = max(np.round(y_position - crop_size).astype(int), 0)
            y_max = min(np.round(y_position + crop_size).astype(int), height)

            # Plot the frame with particle positions
            ax.imshow(frame[y_min:y_max, x_min:x_max], cmap="gray")  # Display the frame in grayscale
            ax.set_xlabel(f"Video {j}")
            ax.set_ylabel(f"Particle {particle_num}")

    plt.tight_layout()
    plt.show()


def localize_particles(image_path: str, background_image_path: str, scope: str = "olympus", magnification: int = 10,
                       subtract_background: bool = False, visualize_localization: bool = False,
                       thresh_value: int = 1, min_particle_separation: float = 3.0, border: float = 4.0,
                       fid_size: int = 1, min_particle_area_NS:int = 0, max_particle_area_NS: int = 0,
                       min_particle_area_fid: int = 0, max_particle_area_fid: int = 0):
    """
    Identifies, filters, and localizes particles in a movie tiff. Optional Fiducial particle identification and localization.

    Parameters:
    - image_path: str path to image stack to be analyzed.
    - background_image_path: str path to image to be used for background subtraction.
    - magnification: magnification of objective lens used.
    - subtraction_background: whether or not to subtract background image from stack.
    - thresh_value: threshold value used for image binarization for particle identification. Will vary with illumination, scope.
    - min_particle_separation: miniumum separation between particle and neighboring particles for filtering identified particles. In units of NS length.
    - border: miniumum separation between particle and edge of image for filtering identified particles. In units of NS length.
    - fid size: diameter of the fiducial particles. int, in microns.
    """

    ##--Specify magnification and pixel size --##
    NS_length_nm = 2300  # nm

    if scope == "olympus":
        if magnification == 10:
            pixel_nm = 650
            if min_particle_area_NS == 0:
                min_particle_area_NS = 18  # default
            if max_particle_area_NS == 0:
                max_particle_area_NS = 30  # default

            if fid_size == 1:
                if min_particle_area_fid == 0:
                    min_particle_area_fid = 3  # default
                if max_particle_area_fid == 0:
                    max_particle_area_fid = 8  # default


        elif magnification == 20:
            pixel_nm = 325
            min_particle_area_NS = 4*18
            max_particle_area_NS = 4*30

            if fid_size == 1:
                min_particle_area_fid = 4*3
                max_particle_area_fid = 4*8

        else:
            print("error: magnification not properly specified!")
            return



    elif scope == "nikon":
        if magnification == 10:
            pixel_nm = 1100

            if min_particle_area_NS == 0:
                min_particle_area_NS = 7  # default
            if max_particle_area_NS == 0:
                max_particle_area_NS = 13  # default

            if fid_size == 1:
                if min_particle_area_fid == 0:
                    min_particle_area_fid = 1  # default
                if max_particle_area_fid == 0:
                    max_particle_area_fid = 3  # default

        elif magnification == 15:
            pixel_nm = 730
            if min_particle_area_NS == 0:
                min_particle_area_NS = 12  # default
            if max_particle_area_NS == 0:
                max_particle_area_NS = 22  # default

            if fid_size == 1:
                if min_particle_area_fid == 0:
                    min_particle_area_fid = 2  # default
                if max_particle_area_fid == 0:
                    max_particle_area_fid = 10  # default

        elif magnification == 20:
            pixel_nm = 550

            min_particle_area_NS = 18
            max_particle_area_NS = 34

            if fid_size == 1:
                min_particle_area_fid = 4*1  # tbd
                max_particle_area_fid = 4*3  # tbd

        else:
            print("error: magnification not properly specified!")
            return



    NS_length_pixel = np.floor(NS_length_nm / pixel_nm).astype(int)

    ##--Load images, subtract background, and invert--##
    # Load the grayscale image
    images = tifffile.imread(image_path)[0:10, :, :]  # subset to first 10 for speed

    # Perform flat field correction for non-uniform illumination
    if subtract_background:
        background_image = tifffile.imread(
            background_image_path)  # may need to subset to first image or average if stack

        if len(background_image.shape) == 2:
            images = flat_field_correction(image_stack=images, flat_field=background_image)

        elif len(background_image.shape) == 3:
            background_image_mean = np.mean(background_image, axis=0)
            background_image_mean = np.round(background_image_mean).astype(np.uint16)  # Round & convert to uint16
            images = flat_field_correction(image_stack=images, flat_field=background_image_mean)
    # Invert, subset first image, and compute image shape
    # Invert image
    images = ((2 ** 16 - 1) - images)
    images = images - np.min(images) + 1

    image = images[0, :, :]

    height, width = image.shape

    ##--Thresholding for binarization and particle identification--##

    # alternate, analytical methods to determine a threshold based off of pixel values (percentile, N distribution)
    pixel_values = image.ravel()
    mu_pixel = np.mean(pixel_values)
    std_pixel = np.std(pixel_values)

    if thresh_value == 1: # if threshold value is not specified explicity (default taken)
        thresh_value = mu_pixel + 3*std_pixel

    _, binary_image = cv2.threshold(image, thresh_value, 5 * thresh_value, cv2.THRESH_BINARY)

    ##--Label image and identify particles--##
    # Use scikit-image to label the connected components
    labeled_image = measure.label(binary_image)

    # Get the properties of the labeled regions
    # Fields of identified regions (particles)
    # area: Area of the labeled region in pixels.
    # centroid: Centroid (center of mass) coordinates (y, x) of the labeled region.
    # bbox: Bounding box coordinates (min_row, min_col, max_row, max_col) of the labeled region.
    # perimeter: Perimeter of the labeled region.
    # eccentricity: Eccentricity of the labeled region.
    # major_axis_length: Length of the major axis of the ellipse that has the same normalized second central moments as the labeled region.
    # minor_axis_length: Length of the minor axis of the ellipse that has the same normalized second central moments as the labeled region.
    particles = measure.regionprops(labeled_image)

    ##--Filter Particles--##
    # Filter out all particles that are too close to one another
    minimum_separation = min_particle_separation * NS_length_pixel
    identified_particles_spacing = filter_close_particles(particles, min_distance=minimum_separation)
    print("num particles filtered out by spacing: ", len(particles) - len(identified_particles_spacing))

    # Filter NS particles based on area - currently manual, can change with different images
    identified_particles_spacing_area = [p for p in identified_particles_spacing if
                                         min_particle_area_NS <= p.area <= max_particle_area_NS]
    print("num additional particles filtered out by area: ",
          len(identified_particles_spacing) - len(identified_particles_spacing_area))

    # Filter fiducial particles based on area - currently manual, can change with different images
    identified_particles_fiducial = [p for p in identified_particles_spacing if
                                     min_particle_area_fid <= p.area <= max_particle_area_fid]

    # Filter out particles that are too close to the edge of the image
    border_size = border * NS_length_pixel
    identified_particles_NS = [p for p in identified_particles_spacing_area if
                               border < p.centroid[1] < width - border_size and border_size < p.centroid[
                                   0] < height - border_size]

    identified_particles_fiducial = [p for p in identified_particles_fiducial if
                                     border < p.centroid[1] < width - border_size and border_size < p.centroid[
                                         0] < height - border_size]

    print("num additional particles filtered out by edge: ",
          len(identified_particles_spacing_area) - len(identified_particles_NS))
    print("num particles remaining for analysis: ", len(identified_particles_NS))
    print("num of fiducial particles identified: ", len(identified_particles_fiducial))

    if visualize_localization:
        plot_localization(image, NS_length_pixel, identified_particles_NS, identified_particles_fiducial)

    localization_parameters = {"localization file": [image_path],
                               "pixel size (nm)": [pixel_nm],
                               "NS length (nm)": [NS_length_nm],
                               "NS length (pixels)": [NS_length_pixel],
                               "thresholding pixel value": [thresh_value],
                               "min particle area (pixels)": [min_particle_area_NS],
                               "max particle area (pixels)": [max_particle_area_NS],
                               "minimum particle separation (pixels)": [minimum_separation],
                               "border (pixels)": [border]}

    df_localization_parameters = pd.DataFrame(localization_parameters)

    return identified_particles_NS, identified_particles_fiducial, df_localization_parameters


def plot_localization(image, NS_length_pixel, identified_particles_NS, identified_particles_fiducial):
    # Draw bounding boxes and numbers around identified particles for original image
    identified_particles_image = image.copy()
    box_size_half = NS_length_pixel * 2

    # Annotate identified_particles_image with NS particles
    for i, particle in enumerate(identified_particles_NS):
        Y, X = np.round(particle.centroid).astype(int)
        cv2.rectangle(identified_particles_image,
                      (np.round(X - box_size_half).astype(int), np.round(Y - box_size_half).astype(int)),
                      ((np.round(X + box_size_half).astype(int)), np.round(Y + box_size_half).astype(int)), (255, 0, 0),
                      2)

        # Draw particle number just below the box
        cv2.putText(identified_particles_image, str(i + 1), (X, Y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

    # Annotate identified_particles_image with fiducial particles
    for i, particle in enumerate(identified_particles_fiducial):
        Y, X = np.round(particle.centroid).astype(int)
        cv2.rectangle(identified_particles_image,
                      (np.round(X - box_size_half).astype(int), np.round(Y - box_size_half).astype(int)),
                      ((np.round(X + box_size_half).astype(int)), np.round(Y + box_size_half).astype(int)),
                      (255, 255, 0), 4)

        # Draw particle number just below the box
        cv2.putText(identified_particles_image, str(i + 1), (X, Y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

    # ## original image
    # plt.figure(figsize=(10, 10))
    # plt.imshow(image, cmap='gray')
    # plt.axis('off')
    # plt.show()
    ## boxes overlay
    plt.figure(figsize=(20, 20))
    plt.imshow(identified_particles_image, cmap='gray')
    plt.axis('off')
    plt.show()
    return


def create_df_particles(identified_particles_NS, identified_particles_fiducial, implement_fiducial: bool = False,
                        fiducial_list: list = [0]):
    # Generate dictionary for particle information
    particle_data = {
        "Particle Number": [],
        "Area": [],
        "Localized X Position": [],
        "Localized Y Position": [],
    }

    # if using fiducial bead correction, append selected fiducial bead to be particle number 0
    if implement_fiducial:
        for i, fid in enumerate(fiducial_list, start=1):
            fiducial_particle = identified_particles_fiducial[fid - 1]

            particle_data["Particle Number"].append(-i)
            particle_data["Area"].append(fiducial_particle.area)
            particle_data["Localized X Position"].append(
                fiducial_particle.centroid[1])  # X coordinate from localization, is position within row
            particle_data["Localized Y Position"].append(
                fiducial_particle.centroid[0])  # Y coordinate from localization, is row

    # Populate the dictionary with identified particle properties
    for i, particle in enumerate(identified_particles_NS, start=1):
        particle_data["Particle Number"].append(i)
        particle_data["Area"].append(particle.area)
        particle_data["Localized X Position"].append(
            particle.centroid[1])  # X coordinate from localization, is position within row
        particle_data["Localized Y Position"].append(particle.centroid[0])  # Y coordinate from localization, is row

    # Convert to a pandas DataFrame
    df_particles = pd.DataFrame(particle_data)

    return df_particles


def analyze_video_motion(df, video_num, nbins: int, range_stdev = (0, 0.2)):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))  # 1 row, 2 columns

    # Plot histograms for each column
    title_fontsize = 14
    axis_fontisze = 12
    tick_fontsize = 10


    axes[0].hist(df[f'({video_num}) stdev x position'], bins=nbins, range=range_stdev, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Video {video_num} Histogram of Standard Deviation in X', fontsize=title_fontsize)
    axes[0].set_xlabel('Pixels', fontsize=axis_fontisze)
    axes[0].set_ylabel('Frequency', fontsize=axis_fontisze)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[0].grid(True)

    axes[1].hist(df[f'({video_num}) stdev y position'], bins=nbins, range=range_stdev, edgecolor='black', alpha=0.7)
    axes[1].set_title(f'Video {video_num} Histogram of Standard Deviation in Y', fontsize=title_fontsize)
    axes[1].set_xlabel('Pixels', fontsize=axis_fontisze)
    axes[1].set_ylabel('Frequency', fontsize=axis_fontisze)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def flat_field_correction(image_stack: np.ndarray, flat_field: np.ndarray):
    """
    Apply flat-field correction to a stack of microscopy images.

    Parameters:
    - image_stack: 3D NumPy array (N, H, W), where N is the number of frames.
    - flat_field: 2D NumPy array (H, W) representing the flat-field correction image.

    Returns:
    - corrected_stack: 3D NumPy array with corrected images.
    """

    # Convert images and flat-field to float32 to prevent precision loss
    image_stack = image_stack.astype(np.float32)
    flat_field = flat_field.astype(np.float32)

    # Prevent division by zero (replace 0s with a small value)
    flat_field[flat_field == 0] = 1e-6

    # Expand flat-field to match the image stack shape
    flat_field_expanded = np.expand_dims(flat_field, axis=0)  # Shape: (1, H, W)
    flat_field_expanded = np.broadcast_to(flat_field_expanded, image_stack.shape)  # Shape: (N, H, W)

    # Perform flat-field correction
    corrected_stack = image_stack / flat_field_expanded

    # Normalize to uint16 range
    corrected_stack = (corrected_stack / np.max(corrected_stack)) * 65535
    corrected_stack = corrected_stack.astype(np.uint16)

    return corrected_stack


def plot_mean_position_shift_histograms(df, correction: str, ref_vid_num: int = 1, vid_num: int=2, bin_range=(-1, 1), fid_bin_range = (-0.5, 0.5), bins=100, fid_bins = 50):
    """
    Plots histograms of mean position shifts and corrected shifts in x and y between videos
    """

    if correction == "va":

        fig, axs = plt.subplots(2, 2, figsize=(18, 6))
        ax1, ax2, ax3, ax4 = axs.flatten()

        df_NS = df[df["Particle Number"] > 0]

        # X shifts
        # ax1.hist(df[f'({ref_vid_num}) to ({vid_num}) mean x position shift'], bins=bins, range=bin_range,
        #          edgecolor='black', alpha=0.6, label="x shift")
        ax1.hist(df_NS[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean x position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected x shift")
        ax1.set_xlabel("x shift (pixels, 650nm/pixel)")
        ax1.set_ylabel("count")
        ax1.set_title("X Position Shift")
        ax1.legend()

        # Y shifts
        # ax2.hist(df[f'({ref_vid_num}) to ({vid_num}) mean y position shift'], bins=bins, range=bin_range,
        #          edgecolor='black', alpha=0.6, label="y shift")
        ax2.hist(df_NS[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean y position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected y shift")
        ax2.set_xlabel("y shift (pixels, 650nm/pixel)")
        ax2.set_title("Y Position Shift")
        ax2.legend()

        # # Displacements
        # ax3.hist(df[f'({ref_vid_num}) to ({vid_num}) mean displacement'], bins=bins, range=bin_range,
        #          edgecolor='black', alpha=0.6, label="displacement")
        # ax3.hist(df[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean displacement'], bins=bins,
        #          range=bin_range,
        #          edgecolor='black', alpha=0.6, label=f"{correction} corrected displacement")
        # ax3.set_xlabel("displacement (pixels, 650nm/pixel)")
        # ax3.set_title("Displacement")
        # ax3.legend()

        # Fiducial particles shifts - check quality of correction
        df_fid = df[df["Particle Number"] < 0]

        # X shifts
        ax3.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean x position shift'], bins=fid_bins, #range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected x shift")
        ax3.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) mean x position shift'], bins=fid_bins,  # range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label="x shift")
        ax3.set_xlabel("x shift (pixels, 650nm/pixel)")
        ax3.set_ylabel("count")
        ax3.set_title("Fiducials X Position Shift")
        ax3.legend()

        # Y shifts
        ax4.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean y position shift'], bins=fid_bins, #range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected y shift")
        ax4.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) mean y position shift'], bins=fid_bins,  # range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label="y shift")
        ax4.set_xlabel("y shift (pixels, 650nm/pixel)")
        ax4.set_title("Fiducials Y Position Shift")
        ax4.legend()

        # # Displacement
        # ax6.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) mean displacement'], bins=fid_bins,  # range=fid_bin_range,
        #          edgecolor='black', alpha=0.6, label="displacement")
        # ax6.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean displacement'], bins=fid_bins,
        #          # range=fid_bin_range,
        #          edgecolor='black', alpha=0.6, label=f"{correction} corrected displacement")
        # ax6.set_xlabel("displacement (pixels, 650nm/pixel)")
        # ax6.set_title("Fiducials Displacement")
        # ax6.legend()


        fig.suptitle(f"Mean Position Shift between ({ref_vid_num}) and ({vid_num}), {correction} correction", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    elif correction == "":
        fig, axs = plt.subplots(1, 2, figsize=(18, 3))
        ax1, ax2 = axs.flatten()

        df_NS = df[df["Particle Number"] > 0]

        # X shifts
        ax1.hist(df_NS[f'({ref_vid_num}) to ({vid_num}) mean x position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label="x shift")
        ax1.set_xlabel("x shift (pixels, 650nm/pixel)")
        ax1.set_ylabel("count")
        ax1.set_title("X Position Shift")
        ax1.legend()

        # Y shifts
        ax2.hist(df_NS[f'({ref_vid_num}) to ({vid_num}) mean y position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label="y shift")
        ax2.set_xlabel("y shift (pixels, 650nm/pixel)")
        ax2.set_title("Y Position Shift")
        ax2.legend()

        # # Displacements
        # ax3.hist(df[f'({ref_vid_num}) to ({vid_num}) mean displacement'], bins=bins, range=bin_range,
        #          edgecolor='black', alpha=0.6, label="displacement")
        # ax3.set_xlabel("displacement (pixels, 650nm/pixel)")
        # ax3.set_title("Displacement")
        # ax3.legend()
        #
        fig.suptitle(f"Mean Position Shift between ({ref_vid_num}) and ({vid_num}), {correction} correction",
                     fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    else:
        print("correction type improperly speficied. Must be 'va' or ''. ")


def build_rigid_transform(theta, tx, ty):
    """
    Build a 2D rigid body transformation matrix (rotation + translation)
    from given parameters.

    Parameters:
    - theta: rotation angle in radians
    - tx: translation in x
    - ty: translation in y

    Returns:
    - A: 2x3 transformation matrix
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    t = np.array([[tx], [ty]])
    A = np.hstack((R, t))  # Combine rotation and translation
    return A


def manually_rotate_mean_position_cols(particles_df, vid_num: int = 1, new_vid_num: int = 2, theta: float = 0.0, tx: float = 0.0, ty: float = 0.0):
    """
            Applies manual 2d rigid body rotation to mean position cols
                    """
    x_col_corrected = f"({new_vid_num}) mean x position"
    y_col_corrected = f"({new_vid_num}) mean y position"

    A = build_rigid_transform(theta=theta, tx=tx, ty=ty)

    # extract the position of all beads in the video
    P1 = extract_mean_bead_positions(particles_df, vid_num=vid_num)

    # perform rigid body transformation to correct for drift
    P1_transformed = apply_rigid_transform(A, P1)

    particles_df[x_col_corrected] = P1_transformed[0, :]  # First row → x values
    particles_df[y_col_corrected] = P1_transformed[1, :]  # Second row → y values


def manually_defocus_mean_position_cols(particles_df, vid_num: int = 1, new_vid_num: int = 2, defocus: float = 0.0, image_shape=(1200,1200)):

    x_col_corrected = f"({new_vid_num}) mean x position"
    y_col_corrected = f"({new_vid_num}) mean y position"

    center = np.array([[image_shape[1] / 2], [image_shape[0] / 2]])

    # extract the position of all beads in the video
    P1 = extract_mean_bead_positions(particles_df, vid_num=vid_num)

    radial = P1 - center
    distorted = P1 + defocus * radial

    particles_df[x_col_corrected] = distorted[0, :]  # First row → x values
    particles_df[y_col_corrected] = distorted[1, :]  # Second row → y values

def compute_mean_displacement_fiducials(particles_df, vid_num: int =2, ref_vid_num: int = 1):
    # subset df to fiducial particles
    fiducial_particles_df = particles_df[particles_df["Particle Number"] < 0]

    # compute the displacment for each fiducial
    x_shift_col = f"({ref_vid_num}) to ({vid_num}) mean x position shift"
    y_shift_col = f"({ref_vid_num}) to ({vid_num}) mean y position shift"

    displacements = np.sqrt(fiducial_particles_df[x_shift_col] ** 2 + fiducial_particles_df[y_shift_col] ** 2).tolist()

    # compute mean displacement and stdev
    mean_radial_displacement = np.mean(displacements)
    stdev_radial_displacement = np.std(displacements)

    return mean_radial_displacement, stdev_radial_displacement


def compute_displacement_individual_fiducials(particles_df, vid_num: int =2, ref_vid_num: int = 1):
    # subset df to fiducial particles
    fiducial_particles_df = particles_df[particles_df["Particle Number"] < 0]

    # compute the displacment for each fiducial
    x_shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean x position shift"
    y_shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"

    displacements = np.sqrt(fiducial_particles_df[x_shift_col] ** 2 + fiducial_particles_df[y_shift_col] ** 2).tolist()

    r = np.sqrt((fiducial_particles_df["Localized X Position"]-600)**2 + (fiducial_particles_df["Localized Y Position"]-600)**2).tolist()

    return r, displacements

def fit_single_gaussian_y_shift(df, vid_num: int = 2, ref_vid_num = 1,  bins=500, plot=True, bin_range=(-0.15, 0.15), sigma_guess: float = 0.02):
    def single_gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    column = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"
    data = df[column].dropna().values

    # Histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=bin_range, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess: peaks at 0 and -0.25
    idx_peak = np.argmin(np.abs(bin_centers-0.00))

    amp_guess = counts[idx_peak]

    mu_guess = bin_centers[idx_peak]

    initial_guess = [amp_guess, mu_guess, sigma_guess]
    bounds = (
        [0, bin_range[0], 1e-6],  # lower bounds
        [np.inf, bin_range[1], 1]  # upper bounds
    )

    try:
        popt, _ = curve_fit(single_gaussian, bin_centers, counts, p0=initial_guess, bounds=bounds)

        amp_fit, mu_fit, sigma_fit = popt
        if plot:



            x_fit = np.linspace(bin_range[0], bin_range[1], 1000)
            y_fit = single_gaussian(x_fit, *popt)

            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=bins, range=bin_range, density=False, alpha=0.5, edgecolor='black', label='Data')
            plt.plot(x_fit, y_fit, 'r-', label='Double Gaussian Fit')

            plt.axvline(x=mu_fit + sigma_fit, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_fit - sigma_fit, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_fit + 2 * sigma_fit, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_fit - 2 * sigma_fit, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_fit + 3 * sigma_fit, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_fit - 3 * sigma_fit, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_fit + 4 * sigma_fit, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_fit - 4 * sigma_fit, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_fit + 5 * sigma_fit, color='red', linestyle='--', linewidth=1, label='5 sigma')
            plt.axvline(x=mu_fit - 5 * sigma_fit, color='red', linestyle='--', linewidth=1, label='5 sigma')


            plt.title(
                f'Double Gaussian Fit for "{column}"\n'
                f'Fit Peak 1: amp={amp_fit:.2f}, μ={mu_fit:.3f}, σ={sigma_fit:.3f} | '
            )


            plt.xlabel(column)
            plt.xlim(bin_range[0], bin_range[1])
            plt.ylabel('Density')
            #plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        return amp_fit, mu_fit, sigma_fit

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None


def fit_double_gaussian_y_shift(df, vid_num: int = 2, ref_vid_num = 1,  bins=500, plot=True, bin_range=(-1, 1), bound_peak_y_mean_shift: float = -0.25, sigma_guess: float = 0.02):
    def double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
        return (
                amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +
                amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
        )

    column = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"
    data = df[column].dropna().values

    # Histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=bin_range, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess: peaks at 0 and -0.25
    idx_unbound = np.argmin(np.abs(bin_centers-0.00))
    idx_bound = np.argmin(np.abs(bin_centers-bound_peak_y_mean_shift))

    amp_guess_unbound = counts[idx_unbound]
    amp_guess_bound = counts[idx_bound]

    mu_guess_unbound = bin_centers[idx_unbound]
    mu_guess_bound = bin_centers[idx_bound]

    initial_guess = [amp_guess_unbound, mu_guess_unbound, sigma_guess, amp_guess_bound, mu_guess_bound, sigma_guess]
    bounds = (
        [0, bin_range[0], 1e-6, 0, bin_range[0], 1e-6],  # lower bounds
        [np.inf, bin_range[1], 1, np.inf, bin_range[1], 1]  # upper bounds
    )

    try:
        popt, _ = curve_fit(double_gaussian, bin_centers, counts, p0=initial_guess, bounds=bounds)

        amp_unbound, mu_unbound, sigma_unbound, amp_bound, mu_bound, sigma_bound = popt
        if plot:



            x_fit = np.linspace(bin_range[0], bin_range[1], 1000)
            y_fit = double_gaussian(x_fit, *popt)

            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=bins, range=bin_range, density=False, alpha=0.5, edgecolor='black', label='Data')
            plt.plot(x_fit, y_fit, 'r-', label='Double Gaussian Fit')

            plt.axvline(x=mu_unbound + sigma_unbound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_unbound - sigma_unbound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_unbound + 2 * sigma_unbound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_unbound - 2 * sigma_unbound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_unbound + 3 * sigma_unbound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_unbound - 3 * sigma_unbound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_unbound + 4 * sigma_unbound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_unbound - 4 * sigma_unbound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_unbound + 5 * sigma_unbound, color='red', linestyle='--', linewidth=1, label='5 sigma')
            plt.axvline(x=mu_unbound - 5 * sigma_unbound, color='red', linestyle='--', linewidth=1, label='5 sigma')

            plt.axvline(x=mu_bound + sigma_bound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_bound - sigma_bound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_bound + 2 * sigma_bound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_bound - 2 * sigma_bound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_bound + 3 * sigma_bound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_bound - 3 * sigma_bound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_bound + 4 * sigma_bound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_bound - 4 * sigma_bound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_bound + 5 * sigma_bound, color='red', linestyle='--', linewidth=1, label='5 sigma')
            plt.axvline(x=mu_bound - 5 * sigma_bound, color='red', linestyle='--', linewidth=1, label='5 sigma')


            plt.title(
                f'Double Gaussian Fit for "{column}"\n'
                f'Unbound Peak 1: amp={amp_unbound:.2f}, μ={mu_unbound:.3f}, σ={sigma_unbound:.3f} | '
                f'Bound Peak 2: amp={amp_bound:.2f}, μ={mu_bound:.3f}, σ={sigma_bound:.3f}'
            )


            plt.xlabel(column)
            plt.xlim(bin_range[0], bin_range[1])
            plt.ylabel('Density')
            #plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        return amp_unbound, mu_unbound, sigma_unbound, amp_bound, mu_bound, sigma_bound

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None


def fit_constrained_double_gaussian_y_shift(df, vid_num: int = 2, ref_vid_num = 1,  bins=500, plot=True, bin_range=(-1, 1), bound_peak_y_mean_shift: float = -0.25, sigma_guess: float = 0.02):
    def constrained_double_gaussian(x, amp1, mu1, sigma1, amp2, delta_mu, sigma2):
        mu2 = mu1 + delta_mu
        return (
                amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2)) +
                amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
        )

    column = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"
    data = df[column].dropna().values

    # Histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=bin_range, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess: peaks at 0 and -0.25
    idx_unbound = np.argmin(np.abs(bin_centers-0.00))
    idx_bound = np.argmin(np.abs(bin_centers-bound_peak_y_mean_shift))

    amp_guess_unbound = counts[idx_unbound]
    amp_guess_bound = counts[idx_bound]

    mu_guess_unbound = bin_centers[idx_unbound]
    mu_guess_bound = bin_centers[idx_bound]

    delta_mu_guess = mu_guess_bound - mu_guess_unbound
    initial_guess = [amp_guess_unbound, mu_guess_unbound, sigma_guess, amp_guess_bound, delta_mu_guess, sigma_guess]

    bounds = (
        [0, bin_range[0], 1e-6, 0, -0.3, 1e-6],  # lower bounds
        [np.inf, bin_range[1], 1, np.inf, -0.23, 1]  # upper bounds
    )

    # initial_guess = [amp_guess_unbound, mu_guess_unbound, sigma_guess, amp_guess_bound, mu_guess_bound, sigma_guess]
    # bounds = (
    #     [0, bin_range[0], 1e-6, 0, bin_range[0], 1e-6],  # lower bounds
    #     [np.inf, bin_range[1], 1, np.inf, bin_range[1], 1]  # upper bounds
    # )

    try:
        popt, _ = curve_fit(constrained_double_gaussian, bin_centers, counts, p0=initial_guess, bounds=bounds)

        amp_unbound, mu_unbound, sigma_unbound, amp_bound, mu_bound, sigma_bound = popt
        if plot:



            x_fit = np.linspace(bin_range[0], bin_range[1], 1000)
            y_fit = constrained_double_gaussian(x_fit, *popt)

            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=bins, range=bin_range, density=False, alpha=0.5, edgecolor='black', label='Data')
            plt.plot(x_fit, y_fit, 'r-', label='Double Gaussian Fit')

            plt.axvline(x=mu_unbound + sigma_unbound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_unbound - sigma_unbound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_unbound + 2 * sigma_unbound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_unbound - 2 * sigma_unbound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_unbound + 3 * sigma_unbound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_unbound - 3 * sigma_unbound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_unbound + 4 * sigma_unbound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_unbound - 4 * sigma_unbound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_unbound + 5 * sigma_unbound, color='red', linestyle='--', linewidth=1, label='5 sigma')
            plt.axvline(x=mu_unbound - 5 * sigma_unbound, color='red', linestyle='--', linewidth=1, label='5 sigma')

            plt.axvline(x=mu_bound + sigma_bound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_bound - sigma_bound, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu_bound + 2 * sigma_bound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_bound - 2 * sigma_bound, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu_bound + 3 * sigma_bound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_bound - 3 * sigma_bound, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu_bound + 4 * sigma_bound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_bound - 4 * sigma_bound, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu_bound + 5 * sigma_bound, color='red', linestyle='--', linewidth=1, label='5 sigma')
            plt.axvline(x=mu_bound - 5 * sigma_bound, color='red', linestyle='--', linewidth=1, label='5 sigma')


            plt.title(
                f'Double Gaussian Fit for "{column}"\n'
                f'Unbound Peak 1: amp={amp_unbound:.2f}, μ={mu_unbound:.3f}, σ={sigma_unbound:.3f} | '
                f'Bound Peak 2: amp={amp_bound:.2f}, μ={mu_bound:.3f}, σ={sigma_bound:.3f}'
            )


            plt.xlabel(column)
            plt.xlim(bin_range[0], bin_range[1])
            plt.ylabel('Density')
            #plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        return amp_unbound, mu_unbound, sigma_unbound, amp_bound, mu_bound, sigma_bound

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None


def fit_single_gaussian_for_stdev(df, vid_num:int = 1, axis: str = "y", bins=500, plot=True, bin_range=(0, 0.25), sigma_guess=0.05):
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    column = f"({vid_num}) stdev {axis} position"
    data = df[column].dropna().values

    # Histogram data
    counts, bin_edges = np.histogram(data, bins=bins, range=bin_range, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Initial guess
    amp_guess = np.max(counts)
    mu_guess = np.mean(data)
    sigma_guess = sigma_guess
    initial_guess = [amp_guess, mu_guess, sigma_guess]

    try:
        popt, _ = curve_fit(gaussian, bin_centers, counts, p0=initial_guess, bounds=([0, 0, 1e-6], [3*amp_guess, 0.25, 0.25]))
        amp, mu, sigma = popt

        if plot:
            x_fit = np.linspace(bin_range[0], bin_range[1], 1000)
            y_fit = gaussian(x_fit, *popt)

            plt.figure(figsize=(8, 4))
            plt.hist(data, bins=bins, range=bin_range, density=False, alpha=0.5, edgecolor='black', label='Data')
            plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')
            plt.axvline(x=mu+sigma, color='blue', linestyle='--', linewidth=1, label='1 sigma')
            plt.axvline(x=mu+2*sigma, color='green', linestyle='--', linewidth=1, label='2 sigma')
            plt.axvline(x=mu+3*sigma, color='yellow', linestyle='--', linewidth=1, label='3 sigma')
            plt.axvline(x=mu+4*sigma, color='orange', linestyle='--', linewidth=1, label='4 sigma')
            plt.axvline(x=mu+5*sigma, color='red', linestyle='--', linewidth=1, label='5 sigma')


            plt.title(
                f'Gaussian Fit for \"{column}\"\n'
                f'amp={amp:.2f}, μ={mu:.3f}, σ={sigma:.3f}'
            )

            plt.xlabel(column)
            plt.xlim(bin_range[0], bin_range[1])
            plt.ylabel('Density')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return amp, mu, sigma

    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return None


def mask_on_stdev_x_manual(particles_df, vid_num:int = 1, window = (0, 0.2)):
    # specify window to create mask from
    std_x_col = f"({vid_num}) stdev x position"
    std_x = particles_df[std_x_col].values
    mask = (std_x > window[0]) & (std_x < window[1])

    return mask

def mask_on_stdev_x_automated(particles_df, vid_num:int = 1, m_sigma: float = 5.0, filter: str = "in", plot: bool = True):
    # specify stdev data to create mask from
    std_x_col = f"({vid_num}) stdev x position"
    std_x = particles_df[std_x_col].values

    # fit gaussian to stdev data
    amp, mu, sigma = fit_single_gaussian_for_stdev(df=particles_df, vid_num=vid_num, axis="x", bins=500, plot=plot,
                                                   bin_range=(0, 0.25), sigma_guess=0.05)

    if filter == "in":
        mask = (std_x > mu - m_sigma * sigma) & (std_x < mu + m_sigma * sigma)
    elif filter == "out":
        mask = (std_x > mu + m_sigma * sigma)

    return mask

def mask_on_stdev_y_manual(particles_df, vid_num:int = 1, window = (0, 0.2)):
    # specify window to create mask from
    std_y_col = f"({vid_num}) stdev y position"
    std_y = particles_df[std_y_col].values
    mask = (std_y > window[0]) & (std_y < window[1])

    return mask

def mask_on_stdev_y_automated(particles_df, vid_num:int = 1, m_sigma: float = 5.0, filter: str = "in", plot: bool = True):
    # specify stdev around mean to create mask from
    std_y_col = f"({vid_num}) stdev y position"
    std_y = particles_df[std_y_col].values

    # fit gaussian to stdev data
    amp, mu, sigma = fit_single_gaussian_for_stdev(df=particles_df, vid_num=vid_num, axis="y", bins=500, plot=plot,
                                                   bin_range=(0, 0.25), sigma_guess=0.05)

    if filter == "in":
        mask = (std_y > mu - m_sigma * sigma) & (std_y < mu + m_sigma * sigma)
    elif filter == "out":
        mask = (std_y > mu + m_sigma * sigma)

    return mask

def mask_on_NS_particles(particles_df):
    # specify stdev around mean to create mask from
    particle_numbers = particles_df["Particle Number"].values
    mask = particle_numbers > 0

    return mask

def mask_on_mean_position_shift(particles_df, ref_vid_num: int = 1, vid_num: int = 2, mean: float = 0, sigma: float = 0.02, n_sigma: float = 3):
    shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"
    shift_values = particles_df[shift_col].values
    mask = (shift_values < mean + n_sigma*sigma) & (shift_values > mean - n_sigma*sigma)

    return mask

def digitize(particles_df, vid_num: int = 2, ref_vid_num: int = 1, sigma_baseline: float = 0.02, delta_bound_peak: float = -0.25, n_sigma: float = 3, m_sigma:float = 3):
    # (1)
    NS_particle_mask = mask_on_NS_particles(particles_df=particles_df)

    # (2)
    # _, mu_unbound, _, _, _, _ = (
    #     fit_double_gaussian_y_shift(df=particles_df, vid_num=vid_num, ref_vid_num=ref_vid_num, bins=500, plot=False, bin_range=(-1, 1),
    #                                 bound_peak_y_mean_shift=delta_bound_peak, sigma_guess=sigma_baseline))
    if vid_num == 1:
        mu_unbound = 0.00
    else:
        _, mu_unbound, _ = fit_single_gaussian_y_shift(df=particles_df, vid_num=vid_num, ref_vid_num=ref_vid_num, bins=100, plot=False,
                                    bin_range=(-0.15, 0.15), sigma_guess=0.02)

    unbound_position_shift_mask = mask_on_mean_position_shift(particles_df=particles_df, vid_num=vid_num,ref_vid_num=ref_vid_num,
                                    mean=mu_unbound, sigma=sigma_baseline, n_sigma=n_sigma)
    bound_position_shift_mask = mask_on_mean_position_shift(particles_df=particles_df, vid_num=vid_num, ref_vid_num=ref_vid_num,
                                    mean=mu_unbound+delta_bound_peak, sigma=sigma_baseline, n_sigma=n_sigma)

    # (3)
    stdev_y_mask = mask_on_stdev_y_automated(particles_df=particles_df, vid_num=vid_num, m_sigma=m_sigma, filter="in", plot=False)

    unbound_mask = NS_particle_mask & unbound_position_shift_mask & stdev_y_mask
    bound_mask = NS_particle_mask & bound_position_shift_mask & stdev_y_mask

    particles_df[f"({vid_num}) state"] = pd.Series([pd.NA] * len(particles_df), dtype='boolean')
    particles_df.loc[bound_mask, f"({vid_num}) state"] = True
    particles_df.loc[unbound_mask, f"({vid_num}) state"] = False

    add_num_bound_instances(particles_df)

    return

def digitize_on_baseline_shift_manual(particles_df, vid_num: int = 2, ref_vid_num: int = 1, unbound_window: tuple = (-0.0, 0.0), bound_window: tuple = (0.0, 0.0), correction: str = "va"):
    state_col = f"({vid_num}) state"

    if correction == "":
        shift_col = f"({ref_vid_num}) to ({vid_num}) mean y position shift"
    elif correction == "va":
        shift_col = f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"


    def map_state(val):
        if unbound_window[0] <= val <= unbound_window[1]:
            return False
        elif bound_window[0] <= val <= bound_window[1]:
            return True
        else:
            return np.nan

    particles_df[state_col] = particles_df[shift_col].apply(map_state)

    add_num_bound_instances(particles_df)


def add_num_bound_instances(particles_df, ref_vid_num = 1):
    state_cols = [col for col in particles_df.columns if col.endswith("state")] # need to digitize with same ref vid num

    for vid_num, _ in enumerate(state_cols, start=ref_vid_num):
        if vid_num == ref_vid_num:
            particles_df[f"({ref_vid_num}) to ({vid_num}) total bound instances"] = (
                particles_df[f"({vid_num}) state"].fillna(False).astype(int))
        else:
            particles_df[f"({ref_vid_num}) to ({vid_num}) total bound instances"] = (
                    particles_df[f"({vid_num}) state"].fillna(False).astype(int) +
                    particles_df[f"({ref_vid_num}) to ({vid_num-1}) total bound instances"])


def create_aggregate_digitized_df(particles_df):
    unbound_list = []
    bound_list = []
    undetermined_list = []
    vid_nums = []

    state_cols = [col for col in particles_df.columns if col.endswith("state")]

    for col in state_cols:

        vid_num = int(re.search(r"\((\d+)\)", col).group(1))
        unbound_count = (particles_df[col] == False).sum()
        bound_count = (particles_df[col] == True).sum()
        undetermined_count = particles_df[col].isna().sum()


        vid_nums.append(vid_num)
        unbound_list.append(unbound_count)
        bound_list.append(bound_count)
        undetermined_list.append(undetermined_count)

    vid_nums = np.asarray(vid_nums)
    unbound_list = np.asarray(unbound_list)
    bound_list = np.asarray(bound_list)
    undetermined_list = np.asarray(undetermined_list)

    identified_list = bound_list+unbound_list
    total_list = bound_list+unbound_list+undetermined_list



    # Stack columns and create DataFrame
    aggregate_df = pd.DataFrame(np.column_stack((vid_nums, total_list, identified_list,
                                                 undetermined_list, undetermined_list/total_list,
                                                 bound_list, bound_list/identified_list, bound_list/total_list,
                                                 unbound_list, unbound_list/identified_list, unbound_list/total_list)),
                                columns=[
        "video number", "total number particles", "number identified particles",
        "number undetermined particles", "fraction total unidentified particles",
        "number particles bound", "fraction identified particles bound", "fraction total particles bond",
        "number particles unbound", "fraction identified particles unbound", "fraction total particles unbound"
        ])

    # Cast specific columns to integers
    int_cols = [
        "video number", "total number particles", "number identified particles",
        "number undetermined particles", "number particles bound", "number particles unbound"
    ]
    aggregate_df[int_cols] = aggregate_df[int_cols].astype(int)

    return aggregate_df


def subset_aggregate_digitized_df(particles_df, aggregate_df, ref_vid_num: int, vid_nums: List[int]):
    subset = particles_df[f"({ref_vid_num}) state"] == False

    subset_cols = [
        f"subset ({ref_vid_num}) total number particles",
        f"subset ({ref_vid_num}) number identified particles",
        f"subset ({ref_vid_num}) number undetermined particles",
        f"subset ({ref_vid_num}) fraction total undetermined particles",
        f"subset ({ref_vid_num}) number particles bound",
        f"subset ({ref_vid_num}) fraction identified particles bound",
        f"subset ({ref_vid_num}) fraction total particles bound",
        f"subset ({ref_vid_num}) number particles unbound",
        f"subset ({ref_vid_num}) fraction identified particles unbound",
        f"subset ({ref_vid_num}) fraction total particles unbound"
    ]

    for col in subset_cols:
        if "number" in col:  # specify as int
            aggregate_df[col] = pd.Series([pd.NA] * len(aggregate_df), dtype="Int64")
        else:  # specify as float
            aggregate_df[col] = np.nan

    vid_list = [ref_vid_num] + vid_nums
    for vid_num in vid_list:
        current_state_col = f"({vid_num}) state"

        unbound_count = (particles_df[subset][current_state_col] == False).sum()
        bound_count = (particles_df[subset][current_state_col] == True).sum()
        undetermined_count = particles_df[subset][current_state_col].isna().sum()
        
        total_count = unbound_count+bound_count+undetermined_count
        identified_count = unbound_count+bound_count

        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[0]] = total_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[1]] = identified_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[2]] = undetermined_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[3]] = undetermined_count/total_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[4]] = bound_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[5]] = bound_count/identified_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[6]] = bound_count/total_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[7]] = unbound_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[8]] = unbound_count / identified_count
        aggregate_df.loc[aggregate_df["video number"] == vid_num, subset_cols[9]] = unbound_count / total_count



def compute_subset_new_binders(particles_df, vid_num: int = 2, ref_vid_num: int = 1):
    start_unbound_total = (particles_df[f"({ref_vid_num}) state"] == False).sum()

    end_bound_total = (
                particles_df[particles_df[f"({ref_vid_num}) state"] == False][f"({vid_num}) state"] == True).sum()

    fraction_new_binders = end_bound_total / start_unbound_total

    return start_unbound_total, end_bound_total, fraction_new_binders


def combine_particle_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Take a list of particle dataframes
    and return a new DataFrame with all their rows appended.
    The index is reset to 0…N_total-1.
    """
    if not dfs:
        return pd.DataFrame()  # or raise ValueError("No dataframes provided")
    return pd.concat(dfs, ignore_index=True)