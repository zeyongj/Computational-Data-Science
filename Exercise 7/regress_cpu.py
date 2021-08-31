import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression


X_columns = ['temperature', 'cpu_percent', 'fan_rpm', 'sys_load_1', 'cpu_freq']
y_column = 'next_temp'


def get_data(filename):
    """
    Read the given CSV file. Returns sysinfo DataFrame with target (next temperature) column created.
    """
    sysinfo = pd.read_csv(filename, parse_dates=['timestamp'])
    
    # TODO: add the column that we want to predict: the temperatures from the *next* time step.
    # The following codes are adapted from https://stackoverflow.com/questions/20095673/shift-column-in-pandas-dataframe-up-by-one .
    sysinfo[y_column] = sysinfo['temperature'] # should be the temperature value from the next row
    sysinfo[y_column] = sysinfo[y_column].shift(-1)
    sysinfo = sysinfo[sysinfo[y_column].notnull()] # the last row should have y_column null: no next temp known
    return sysinfo


def get_trained_coefficients(X_train, y_train):
    """
    Create and train a model based on the training_data_file data.

    Return the model, and the list of coefficients for the 'X_columns' variables in the regression.
    """
    
    # TODO: create regression model and train.
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html .
    model = LinearRegression(fit_intercept = False, normalize = False, copy_X = True, n_jobs = None, positive = False)
    model = model.fit(X_train, y_train, sample_weight = None)
    coefficients = model.coef_
    
    return model, coefficients


def output_regression(coefficients):
    regress = ' + '.join('%.3g*%s' % (coef, col) for col, coef in zip(X_columns, coefficients))
    print('next_temp = ' + regress)


def plot_errors(model, X_valid, y_valid):
    residuals = y_valid - model.predict(X_valid)
    plt.hist(residuals, bins=100)
    plt.savefig('test_errors.png')
    plt.close()


def smooth_test(coef, sysinfo, outfile):
    X_valid, y_valid = sysinfo[X_columns], sysinfo[y_column]
    
    # feel free to tweak these if you think it helps.
    # I change the values to both 1.0.
    transition_stddev = 1.0
    observation_stddev = 1.0

    dims = X_valid.shape[-1]
    initial = X_valid.iloc[0]
    observation_covariance = np.diag([observation_stddev, 2, 10, 1, 100]) ** 2
    transition_covariance = np.diag([transition_stddev, 80, 100, 10, 1000]) ** 2
    
    # Transition = identity for all variables, except we'll replace the top row
    # to make a better prediction, which was the point of all this.
    transition = np.identity(dims) # identity matrix, except...
    
    # TODO: replace the first row of transition to use the coefficients we just calculated (which were passed into this function as coef.).
    transition[0] = coef
    kf = KalmanFilter(
        initial_state_mean=initial,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition,
    )

    kalman_smoothed, _ = kf.smooth(X_valid)

    plt.figure(figsize=(15, 6))
    plt.plot(sysinfo['timestamp'], sysinfo['temperature'], 'b.', alpha=0.5)
    plt.plot(sysinfo['timestamp'], kalman_smoothed[:, 0], 'g-')
    plt.savefig(outfile)
    plt.close()


def main(training_file, validation_file):
    train = get_data(training_file)
    valid = get_data(validation_file)
    X_train, y_train = train[X_columns], train[y_column]
    X_valid, y_valid = valid[X_columns], valid[y_column]

    model, coefficients = get_trained_coefficients(X_train, y_train)
    output_regression(coefficients)
    smooth_test(coefficients, train, 'train.png')

    print("Training score: %g\nValidation score: %g" % (model.score(X_train, y_train), model.score(X_valid, y_valid)))

    plot_errors(model, X_valid, y_valid)
    smooth_test(coefficients, valid, 'valid.png')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
