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
    image = (image - np.min(image)) / (np.max(image) - np.min(image))

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

        # _, _, _, sigma_x, sigma_y, _ = params

        # # Basic quality check thresholds (you can tune these)
        # fit_ok = (
        #         sigma_x < sigma_threshold and
        #         sigma_y < sigma_threshold
        # )

        return params #, fit_ok
    except RuntimeError as e:
        print(f"Gaussian fit failed: {e}")
        return None #, False  # Return None if fitting fails


def fit_particle(image):
    params = fit_2d_gaussian(image)
    _, _, _, sigma_x, sigma_y, _ = params
    fit_ok = ( # really for NS particles
                sigma_x < 3 and
                sigma_y < 3
        )
    return params, fit_ok

# Function to process a single particle
def track_particle(index, particle, images, NS_length_pixel):
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

    # Fit Gaussian over frames
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


    return index, stack_centers_x, stack_centers_y, fit_quality  # Return results


def track_video(video_num: int, image_path: str, background_image_path: str, particles_df, magnification: int = 10,
                parallelize: bool = False, subtract_background: bool = False):
    start = time.time()

    ##--Specify magnification and pixel size --##
    NS_length_nm = 2300  # nm

    if magnification == 10:
        pixel_nm = 650
    elif magnification == 20:
        pixel_nm = 325
    else:
        print("error: magnification not properly specified!")
        return
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
            delayed(track_particle)(index, row, images, NS_length_pixel) for index, row in particles_df.iterrows()
        )

        # Store results in DataFrame with separate X and Y columns
        for index, x_positions, y_positions, _ in results:
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions

    else:
        for index, row in particles_df.iterrows():
            _, x_positions, y_positions, _ = track_particle(index, row, images, NS_length_pixel)
            particles_df.at[index, x_col_title] = x_positions
            particles_df.at[index, y_col_title] = y_positions
    # End timing
    end = time.time()
    elapsed = end - start

    # Display the elapsed time
    print(f"Total execution time: {elapsed:.2f} seconds")


# Functions to process particle traces
def compute_mean_position(position_list):
    """Computes the mean position from a list of X or Y positions."""
    position_array = np.array(position_list)
    return np.mean(position_array).tolist() if len(position_array) > 0 else np.nan


# Function to add mean position columns for X and Y
def add_mean_position_cols(df):
    """Adds mean position columns for X and Y."""
    for col in df.columns:
        match = re.match(r"\((\d+)\) x positions", col)  # Match X position columns
        if match:
            number = match.group(1)
            y_col = f"({number}) y positions"
            if y_col in df.columns:  # Ensure Y column exists
                df[f"({number}) mean x position"] = df[col].apply(compute_mean_position)
                df[f"({number}) mean y position"] = df[y_col].apply(compute_mean_position)

# Function to add mean position columns for X and Y
def add_mean_ff_corrected_position_cols(df):
    """Adds mean position columns for X and Y."""
    for col in df.columns:
        match = re.match(r"\((\d+)\) x positions ff corrected", col)  # Match X position columns
        if match:
            number = match.group(1)
            y_col = f"({number}) y positions ff corrected"
            if y_col in df.columns:  # Ensure Y column exists
                df[f"({number}) mean x position ff corrected"] = df[col].apply(compute_mean_position)
                df[f"({number}) mean y position ff corrected"] = df[y_col].apply(compute_mean_position)


# Updated function to compute standard deviation for separate X and Y columns
def compute_stdev_position(position_list):
    """Computes the standard deviation of a list of X or Y positions."""
    position_array = np.array(position_list)
    return np.std(position_array).tolist() if len(position_array) > 0 else np.nan


def add_stdev_position_cols(df):
    """Adds standard deviation position columns for X and Y."""
    for col in df.columns:
        match = re.match(r"\((\d+)\) x positions", col)  # Match X position columns
        if match:
            number = match.group(1)
            y_col = f"({number}) y positions"
            if y_col in df.columns:  # Ensure Y column exists
                df[f"({number}) stdev x position"] = df[col].apply(compute_stdev_position)
                df[f"({number}) stdev y position"] = df[y_col].apply(compute_stdev_position)


def add_stdev_ff_corrected_position_cols(df):
    """Adds standard deviation position columns for X and Y."""
    for col in df.columns:
        match = re.match(r"\((\d+)\) x positions ff corrected", col)  # Match X position columns
        if match:
            number = match.group(1)
            y_col = f"({number}) y positions ff corrected"
            if y_col in df.columns:  # Ensure Y column exists
                df[f"({number}) stdev x position ff corrected"] = df[col].apply(compute_stdev_position)
                df[f"({number}) stdev y position ff corrected"] = df[y_col].apply(compute_stdev_position)


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

# def add_diffusion_coefficient_corrected_cols(df, time_step, magnification):
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
#         match = re.match(r"\((\d+)\) x positions corrected", col)  # Match X position columns
#         if match:
#             number = match.group(1)
#             y_col = f"({number}) y positions corrected"
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
#

def add_frame_by_frame_corrected_position_cols(particles_df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) x positions"
    y_col = f"({vid_num}) y positions"
    x_col_corrected = f"({vid_num}) x positions ff corrected"
    y_col_corrected = f"({vid_num}) y positions ff corrected"

    # Initialize empty lists for all corrected positions
    corrected_x_all = [[] for _ in range(len(particles_df))]
    corrected_y_all = [[] for _ in range(len(particles_df))]

    particles_df.loc[:, x_col_corrected] = pd.Series([[] for _ in range(len(particles_df))], index=particles_df.index)
    particles_df.loc[:, y_col_corrected] = pd.Series([[] for _ in range(len(particles_df))], index=particles_df.index)

    P1 = extract_mean_fiducial_positions(particles_df, vid_num=ref_vid_num)

    for i in range(len(particles_df[x_col].values[0])): # for every recorded timepoint in the video
        # extract the position of all fiducials in that frame
        P2 = extract_fiducial_positions(particles_df, x_col, y_col, frame_index=i)
        # define A for rigid body transformation to correct for drift
        A = rigid_transform_least_squares(P1, P2)
        # extract the position of all beads in that frame
        P3 = extract_all_bead_positions(particles_df, x_col, y_col, frame_index=i)
        # perform rigid body transformation to correct for drift
        P3_corrected = apply_rigid_transform(A, P3)

        for j in range(len(particles_df)):
            corrected_x_all[j].append(P3_corrected[0, j])
            corrected_y_all[j].append(P3_corrected[1, j])

    # Assign full lists back to the DataFrame at once
    particles_df[x_col_corrected] = corrected_x_all
    particles_df[y_col_corrected] = corrected_y_all

def add_video_average_corrected_mean_position_cols(particles_df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col_corrected = f"({vid_num}) mean x position va corrected"
    y_col_corrected = f"({vid_num}) mean y position va corrected"

    # Initialize nan cols for all corrected mean positions
    # particles_df.loc[:, x_col_corrected] = np.nan
    # particles_df.loc[:, y_col_corrected] = np.nan

    # extract mean positons of all fiducials in reference video
    P1 = extract_mean_fiducial_positions(particles_df, vid_num=ref_vid_num)

    # extract the mean positions of all fiducial in the video
    P2 = extract_mean_fiducial_positions(particles_df, vid_num=vid_num)

    # define A for rigid body transformation to correct for drift
    A = rigid_transform_least_squares(P1, P2)

    # extract the position of all beads in the video
    P3 = extract_mean_bead_positions(particles_df, vid_num=vid_num)

    # perform rigid body transformation to correct for drift
    P3_corrected = apply_rigid_transform(A, P3)

    particles_df[x_col_corrected] = P3_corrected[0, :]  # First row → x values
    particles_df[y_col_corrected] = P3_corrected[1, :]  # Second row → y values


def extract_mean_fiducial_positions(particles_df, vid_num: int = 1):
    """
    Extracts fiducial positions for fiducial bead rows in particles dataframe
    """
    x_col = f"({vid_num}) mean x position"
    y_col = f"({vid_num}) mean y position"
    x_list, y_list = [], []

    fiducial_particles_df = particles_df[particles_df["Particle Number"] < 0]
    for index, row in fiducial_particles_df.iterrows():
        x_list.append(row[x_col])
        y_list.append(row[y_col])

    P = np.array([x_list, y_list])  # shape 2×n

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
    result = least_squares(residuals, initial_params, args=(P1, P2))

    theta, tx, ty = result.x
    #print('theta: ', theta, 'tx: ', tx, 'ty: ', ty)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
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


def add_mean_position_shift_cols(df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) mean x position"
    y_col = f"({vid_num}) mean y position"
    x_ref_col = f"({ref_vid_num}) mean x position"
    y_ref_col = f"({ref_vid_num}) mean y position"
    df[f"({ref_vid_num}) to ({vid_num}) mean x position shift"] = df[x_col] - df[x_ref_col]
    df[f"({ref_vid_num}) to ({vid_num}) mean y position shift"] = df[y_col] - df[y_ref_col]


def add_ff_corrected_mean_position_shift_cols(df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) mean x position ff corrected"
    y_col = f"({vid_num}) mean y position ff corrected"
    x_ref_col = f"({ref_vid_num}) mean x position ff corrected"
    y_ref_col = f"({ref_vid_num}) mean y position ff corrected"
    df[f"({ref_vid_num}) to ({vid_num}) ff corrected mean x position shift"] = df[x_col] - df[x_ref_col]
    df[f"({ref_vid_num}) to ({vid_num}) ff corrected mean y position shift"] = df[y_col] - df[y_ref_col]

def add_va_corrected_mean_position_shift_cols(df, vid_num: int = 2, ref_vid_num: int = 1):
    x_col = f"({vid_num}) mean x position va corrected"
    y_col = f"({vid_num}) mean y position va corrected"
    x_ref_col = f"({ref_vid_num}) mean x position va corrected"
    y_ref_col = f"({ref_vid_num}) mean y position va corrected"
    df[f"({ref_vid_num}) to ({vid_num}) va corrected mean x position shift"] = df[x_col] - df[x_ref_col]
    df[f"({ref_vid_num}) to ({vid_num}) va corrected mean y position shift"] = df[y_col] - df[y_ref_col]

def plot_single_particle_tracking(df, image_path, video_num, frame_number, particle_num):
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

        # Plot the frame with particle positions
        plt.figure(figsize=(5, 5))
        plt.imshow(frame[y_min:y_max, x_min:x_max], cmap="gray")  # Display the frame in grayscale
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


def localize_particles(image_path: str, background_image_path: str, magnification: int = 10,
                       subtract_background: bool = False, visualize_localization: bool = False,
                       thresh_value: int = 1, min_particle_area_fid: int = 3, max_particle_area_fid: int = 8,
                       min_particle_area_NS: int = 15, max_particle_area_NS: float = 30,
                       min_particle_separation: float = 3.0, border: float = 4.0):
    """
    Identifies, filters, and localizes particles in a movie tiff. Optional Fiducial particle identification and localization.

    Parameters:
    - image_path: str path to image stack to be analyzed.
    - background_image_path: str path to image to be used for background subtraction.
    - magnification: magnification of objective lens used.
    - subtraction_background: whether or not to subtract background image from stack.
    - thresh_value: threshold value used for image binarization for particle identification. Will vary with illumination, scope.
    - min_particle_area_fid: minimum particle area for fiducial beads in binarized image
    - max_particle_area_fid: maximum particle area for fiducial beads in binarized image
    - min_particle_area_NS: minimum particle area for NS beads in binarized image
    - max_particle_area_NS: maximum particle area for NS beads in binarized image
    - min_particle_separation: miniumum separation between particle and neighboring particles for filtering identified particles. In units of NS length.
    - border: miniumum separation between particle and edge of image for filtering identified particles. In units of NS length.
    """

    ##--Specify magnification and pixel size --##
    NS_length_nm = 2300  # nm

    if magnification == 10:
        pixel_nm = 650
    elif magnification == 20:
        pixel_nm = 325
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


def analyze_video_motion(df, video_num, free: bool, nbins: int):
    # Set bounds based on free vs flow
    if free:
        range_D = (0, 0.1)
        range_stdev = (0, 1)
    else:
        range_D = (0, 0.002)
        range_stdev = (0, 0.3)

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))  # 1 row, 3 columns

    # Plot histograms for each column
    title_fontsize = 24
    axis_fontisze = 20
    tick_fontsize = 16

    axes[0].hist(df[f'({video_num}) D'], bins=nbins, range=range_D, edgecolor='black', alpha=0.7)
    axes[0].set_title('Histogram of Diffusion Coefficients', fontsize=title_fontsize)
    axes[0].set_xlabel('D (um^2 / s)', fontsize=axis_fontisze)
    axes[0].set_ylabel('Frequency', fontsize=axis_fontisze)
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[0].grid(True)

    axes[1].hist(df[f'({video_num}) stdev x position'], bins=nbins, range=range_stdev, edgecolor='black', alpha=0.7)
    axes[1].set_title('Histogram of Standard Deviation in X', fontsize=title_fontsize)
    axes[1].set_xlabel('Pixels', fontsize=axis_fontisze)
    axes[1].set_ylabel('Frequency', fontsize=axis_fontisze)
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[1].grid(True)

    axes[2].hist(df[f'({video_num}) stdev y position'], bins=nbins, range=range_stdev, edgecolor='black', alpha=0.7)
    axes[2].set_title('Histogram of Standard Deviation in Y', fontsize=title_fontsize)
    axes[2].set_xlabel('Pixels', fontsize=axis_fontisze)
    axes[2].set_ylabel('Frequency', fontsize=axis_fontisze)
    axes[2].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    axes[2].grid(True)

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


def plot_mean_position_shift_histograms(df, correction: str, ref_vid_num: int = 1, vid_num: int=2, bin_range=(-1, 1), fid_bin_range = (-0.5, 0.5), bins=100, fid_bins = 50, ):
    """
    Plots histograms of mean position shifts and corrected shifts in x and y between videos
    """

    if correction == "va" or correction == "ff":

        fig, axs = plt.subplots(2, 2, figsize=(12, 5))
        ax1, ax2, ax3, ax4 = axs.flatten()

        # X shifts
        ax1.hist(df[f'({ref_vid_num}) to ({vid_num}) mean x position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label="x shift")
        ax1.hist(df[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean x position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected x shift")
        ax1.set_xlabel("x shift (pixels, 650nm/pixel)")
        ax1.set_ylabel("count")
        ax1.set_title("X Position Shift")
        ax1.legend()

        # Y shifts
        ax2.hist(df[f'({ref_vid_num}) to ({vid_num}) mean y position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label="y shift")
        ax2.hist(df[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean y position shift'], bins=bins, range=bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected y shift")
        ax2.set_xlabel("y shift (pixels, 650nm/pixel)")
        ax2.set_title("Y Position Shift")
        ax2.legend()

        # Fiducial particles shifts- check quality of correction
        df_fid = df[df["Particle Number"] < 0]

        # X shifts
        ax3.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) mean x position shift'], bins=fid_bins, range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label="x shift")
        ax3.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean x position shift'], bins=fid_bins, range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected x shift")
        ax3.set_xlabel("x shift (pixels, 650nm/pixel)")
        ax3.set_ylabel("count")
        ax3.set_title("Fiducials X Position Shift")
        ax3.legend()

        # Y shifts
        ax4.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) mean y position shift'], bins=fid_bins, range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label="y shift")
        ax4.hist(df_fid[f'({ref_vid_num}) to ({vid_num}) {correction} corrected mean y position shift'], bins=fid_bins, range=fid_bin_range,
                 edgecolor='black', alpha=0.6, label=f"{correction} corrected y shift")
        ax4.set_xlabel("y shift (pixels, 650nm/pixel)")
        ax4.set_title("Fiducials Y Position Shift")
        ax4.legend()




        fig.suptitle(f"Mean Position Shift between ({ref_vid_num}) and ({vid_num}), {correction} correction", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    else:
        print("correction type improperly speficied. Must be 'va' or 'ff'. ")