# Benjamin Wollant
# November 17, 2023
# Adapted from DiffusionDevelopment.ipynb
# Function for calculating diffusion coefficient over time of particle trace

import numpy as np


def calculate_diffusion_coefficient_trace(x_trace, y_trace, window_size=60, dt=0.030):
    def calculate_diffusion_coefficient(x_win, y_win):

        # calculte weight factors
        wf = []
        for m in range(1, 11):
            wf.append((window_size - m + 1) / (m * (2 * m ** 2 + 1)))

        def term_MSD(x, y, i, m):
            return (x[i + m] - x[i]) ** 2 + (y[i + m] - y[i]) ** 2

        def MSD(x, y, m, w):
            MSDw = 0
            for i in range(w - m):
                MSDw += term_MSD(x, y, i, m)

            MSDw = MSDw / (w - m)

            return MSDw

        D = 0

        for m in range(1, 11):
            D += (wf[m - 1] / np.sum(wf)) * (MSD(x_win, y_win, m, window_size) / (4 * m * dt))

        return D

    D_t = np.zeros(len(x_trace))
    spacer = np.zeros(window_size // 2)

    for t in range(len(x_trace) - window_size):
        D_t[t + len(spacer)] = calculate_diffusion_coefficient(1e-3 * x_trace[t:t + window_size],
                                                               1e-3 * y_trace[t:t + window_size])

    return D_t

