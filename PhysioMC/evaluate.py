import numpy as np
import os
import sys
import math
from math import sin


# from scipy.io import loadmat
# import scipy
# from scipy import signal
# from scipy.fftpack import fft, ifft
# from scipy.signal import hilbert, chirp
# from scipy.signal import find_peaks
# from scipy.interpolate import interp1d

# from sklearn.metrics import mean_squared_error

# from feature_extraction import *
# from preprocessing import *

def get_RMSE(RR_label, RR_est):
    RMSE = np.sqrt(mean_squared_error(RR_label, RR_est))
    return RMSE


def get_MAE(RR_label, RR_est):
    MAE_mean = np.abs(RR_label - RR_est).mean()
    MAE_std = np.abs(RR_label - RR_est).std()
    return MAE_mean, MAE_std

def get_MAPE(RR_label, RR_est):
    MAPE_mean = (np.abs(RR_label- RR_est) / RR_label).mean()
    MAPE_std = (np.abs(RR_label- RR_est) / RR_label).std()
    return MAPE_mean, MAPE_std
