import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd
import sys
from datetime import datetime
from pykalman import KalmanFilter

def format_time(dataFrame):
    # Adapted from https://www.programiz.com/python-programming/datetime/strptime.
    return datetime.strptime(dataFrame , "%Y-%m-%d %H:%M:%S.%f")

def to_timestamp(dataFrame):
    return dataFrame.timestamp()

def main():
    # Data Preprocessing:
    filename = sys.argv[1]
    cpu_data = pd.read_csv(filename)
    cpu_data['timestamp'] = cpu_data['timestamp'].apply(format_time)
    cpu_data['new_timestamp'] = cpu_data['timestamp'].apply(to_timestamp)

    # Parameter Tuning:
    lowess = sm.nonparametric.lowess
    loess_smoothed = lowess(cpu_data['temperature'], cpu_data['new_timestamp'], frac = 0.006)

    kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]

    initial_state = kalman_data.iloc[0]
    observation_covariance = np.diag([0.75, 0.45, 0.45, 0.75]) ** 2 # TODO: shouldn't be zero
    transition_covariance = np.diag([0.0075, 0.0075, 0.0075, 0.0075]) ** 2 # TODO: shouldn't be zero
    transition = [[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]] # TODO: shouldn't (all) be zero. Adapted from https://stackoverflow.com/questions/55195011/how-to-define-state-transition-matrix-for-kalman-filters: DT = np.matrix([[1.,0.,dt,0],[0.,1.,0.,dt],[0.,0.,1.,0.],[0.,0.,0.,1.]]).


    kf = KalmanFilter(initial_state_mean = initial_state,
                      observation_covariance = observation_covariance,
                      transition_covariance = transition_covariance,
                      transition_matrices = transition)
    kalman_smoothed, _ = kf.smooth(kalman_data)

    plt.figure(figsize=(12, 4))
    plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5)
    plt.plot(cpu_data['timestamp'], loess_smoothed[:, 1], 'r-')
    plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-')
    # plt.show() # maybe easier for testing
    plt.title ('Task 1: CPU Temperature Noise Reduction')
    plt.xlabel('Time')
    plt.ylabel('temperature')
    plt.legend(['Original Data', 'LOESS Smoothing', 'Kalman Smoothing'])
    plt.savefig('cpu.svg') # for final submission

if __name__ == '__main__':
    main()
    
