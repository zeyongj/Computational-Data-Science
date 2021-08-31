import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys


# representative RGB colours for each label, for nice display
COLOUR_RGB = {
    'red': (255, 0, 0),
    'orange': (255, 113, 0),
    'yellow': (255, 255, 0),
    'green': (0, 230, 0),
    'blue': (0, 0, 255),
    'purple': (187, 0, 187),
    'brown': (117, 60, 0),
    'pink': (255, 186, 186),
    'black': (0, 0, 0),
    'grey': (150, 150, 150),
    'white': (255, 255, 255),
}
name_to_rgb = np.vectorize(COLOUR_RGB.get, otypes=[np.uint8, np.uint8, np.uint8])


def plot_predictions(model, lum=70, resolution=256):
    """
    Create a slice of LAB colour space with given luminance; predict with the model; plot the results.
    """
    wid = resolution
    hei = resolution
    n_ticks = 5

    # create a hei*wid grid of LAB colour values, with L=lum
    ag = np.linspace(-100, 100, wid)
    bg = np.linspace(-100, 100, hei)
    aa, bb = np.meshgrid(ag, bg)
    ll = lum * np.ones((hei, wid))
    lab_grid = np.stack([ll, aa, bb], axis=2)

    # convert to RGB for consistency with original input
    X_grid = lab2rgb(lab_grid)

    # predict and convert predictions to colours so we can see what's happening
    y_grid = model.predict(X_grid.reshape((wid*hei, 3)))
    pixels = np.stack(name_to_rgb(y_grid), axis=1) / 255
    pixels = pixels.reshape((hei, wid, 3))

    # plot input and predictions
    plt.figure(figsize=(10, 5))
    plt.suptitle('Predictions at L=%g' % (lum,))
    plt.subplot(1, 2, 1)
    plt.title('Inputs')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.ylabel('B')
    plt.imshow(X_grid.reshape((hei, wid, 3)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

def myRGB2LAB(X):
    X = np.array(X)
    # The following code is adapted from https://numpy.org/doc/stable/reference/generated/numpy.reshape.html .
    np.reshape(X, (1, -1, 3))
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html .
    from skimage.color import rgb2lab
    X = rgb2lab(X)
    np.reshape(X, (-1, 3))
    return X


def main(infile):
    data = pd.read_csv(infile)
    RGB = data[['R', 'G', 'B']]
    X = RGB / 255 # array with shape (n, 3). Divide by 255 so components are all 0-1.
    y = data['Label'] # array with shape (n,) of colour words.
    # y = np.array(y)

    # TODO: build model_rgb to predict y from X.
    # The following codes are adapted from the lecture notes: https://ggbaker.ca/data-science/content/ml.html .
    # print('---------------------------------------------')
    # print('The following is the output of model_rgb.')
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    # print('X train shape is %s.' % str(X_train.shape))
    # print('X valid shape is %s.' % str(X_valid.shape))
    # print('y train shape is %s.' % str(y_train.shape))
    # print('y valid shape is %s.' % str(y_valid.shape))
    from sklearn.naive_bayes import GaussianNB
    model_rgb = GaussianNB()
    model_rgb.fit(X_train, y_train)
    # TODO: print model_rgb's accuracy score
    rgb_accuracy = model_rgb.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of model_rgb is %g.' % rgb_accuracy)
    
    # TODO: build model_lab to predict y from X by converting to LAB colour first.
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    # print('---------------------------------------------')
    # print('The following is the output of model_lab.')
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import make_pipeline
    model_lab = make_pipeline(
        FunctionTransformer(myRGB2LAB),
        GaussianNB()
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    # print('X train shape is %s.' % str(X_train.shape))
    # print('X valid shape is %s.' % str(X_valid.shape))
    # print('y train shape is %s.' % str(y_train.shape))
    # print('y valid shape is %s.' % str(y_valid.shape))
    model_lab.fit(X_train, y_train)
    # TODO: print model_lab's accuracy score
    lab_accuracy = model_lab.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of model_lab is %g.' % lab_accuracy)
    # print('---------------------------------------------')
    
    plot_predictions(model_rgb)
    plt.savefig('predictions_rgb.png')
    plot_predictions(model_lab)
    plt.savefig('predictions_lab.png')


if __name__ == '__main__':
    main(sys.argv[1])
