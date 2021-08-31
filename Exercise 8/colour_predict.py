import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import sys


OUTPUT_TEMPLATE = (
    'Bayesian classifier:    {bayes_rgb:.3f} {bayes_convert:.3f}\n'
    'kNN classifier:         {knn_rgb:.3f} {knn_convert:.3f}\n'
    'Rand forest classifier: {rf_rgb:.3f} {rf_convert:.3f}\n'
)


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
    y_grid = model.predict(X_grid.reshape((-1, 3)))
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
    plt.imshow(X_grid.reshape((hei, wid, -1)))

    plt.subplot(1, 2, 2)
    plt.title('Predicted Labels')
    plt.xticks(np.linspace(0, wid, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.yticks(np.linspace(0, hei, n_ticks), np.linspace(-100, 100, n_ticks))
    plt.xlabel('A')
    plt.imshow(pixels)

# The following function is copied from my previous assignment.
def myRGB2LAB(X):
    X = np.array(X)
    # The following code is adapted from https://numpy.org/doc/stable/reference/generated/numpy.reshape.html .
    np.reshape(X, (1, -1, 3))
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html .
    from skimage.color import rgb2lab
    X = rgb2lab(X)
    np.reshape(X, (-1, 3))
    return X

def myRGB2HSV(X):
    X = np.array(X)
    # The following code is adapted from https://numpy.org/doc/stable/reference/generated/numpy.reshape.html .
    np.reshape(X, (1, -1, 3))
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html .
    from skimage.color import rgb2hsv
    X = rgb2hsv(X)
    np.reshape(X, (-1, 3))
    return X

def main():
    data = pd.read_csv(sys.argv[1])
    X = data[['R', 'G', 'B']].values / 255
    y = data['Label'].values

    # TODO: create some models
    # The following codes are adapted from my previous assignment.
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # This is for bayes_rgb_model.
    from sklearn.naive_bayes import GaussianNB
    bayes_rgb_model = GaussianNB()
    # bayes_rgb_model.fit(X_train, y_train)
    # TODO: print model_rgb's accuracy score
    # rgb_accuracy = bayes_rgb_model.score(X_valid, y_valid, sample_weight = None)
    # print('The accuracy score of bayes_rgb_model is %g.' % rgb_accuracy)

    
    # This is for bayes_rgb2lab_model.
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import make_pipeline
    model_lab = make_pipeline(
        FunctionTransformer(myRGB2LAB),
        GaussianNB()
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model_lab.fit(X_train, y_train)
    lab_accuracy = model_lab.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of bayes_lab is %g.' % lab_accuracy)
    '''
    

    # This is for bayes_rgb2hsv_model.
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import make_pipeline
    bayes_convert_model = make_pipeline(
        FunctionTransformer(myRGB2HSV),
        GaussianNB()
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    bayes_convert_model.fit(X_train, y_train)
    bayes_hsv_accuracy = bayes_convert_model.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of bayes_hsv is %g.' % bayes_hsv_accuracy)
    '''
    
    
    # After checking, converting to HSV of Bayes model is a better choice!

    # This is for knn_rgb_model
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html .
    from sklearn.neighbors import KNeighborsClassifier
    knn_rgb_model = KNeighborsClassifier(n_neighbors = 8)
    '''
    knn_rgb_model.fit(X_train, y_train)
    knn_accuracy = bayes_rgb_model.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of knn_rgb_model is %g.' % knn_accuracy)
    '''
    
    # This is for knn_rgb2lab_model
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html .
    knn_convert_model = make_pipeline(
        FunctionTransformer(myRGB2LAB),
        KNeighborsClassifier(n_neighbors = 8)
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    knn_lab.fit(X_train, y_train)
    knn_lab_accuracy = knn_lab.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of knn_lab is %g.' % knn_lab_accuracy)
    '''

    # This is for knn_rgb2hav_model
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html .
    knn_hsv = make_pipeline(
        FunctionTransformer(myRGB2HSV),
        KNeighborsClassifier(n_neighbors = 8)
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    knn_hsv.fit(X_train, y_train)
    knn_hsv_accuracy = knn_hsv.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of knn_hsv is %g.' % knn_hsv_accuracy)
    '''

    # After checking, converting to LAB of knn model is a better choice! (10 sample one-tailed t test: p-value = 0.0006062 alternative hypothesis: true mean is greater than 0)

    # This is for rf_rgb_model.
    # The following codes are adapted from https://www.datacamp.com/community/tutorials/random-forests-classifier-python .
    from sklearn.ensemble import RandomForestClassifier
    rf_rgb_model = RandomForestClassifier(n_estimators = 100)
    '''
    rf_rgb_model.fit(X_train,y_train)
    rf_accuracy = rf_rgb_model.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of rf_rgb_model is %g.' % rf_accuracy)
    '''
    
    # This is for rf_rgb2lab_model.
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    # The following codes are adapted from https://www.datacamp.com/community/tutorials/random-forests-classifier-python .
    rf_convert_model = make_pipeline(
        FunctionTransformer(myRGB2LAB),
        RandomForestClassifier(n_estimators = 100)
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    rf_lab.fit(X_train, y_train)
    rf_lab_accuracy = rf_lab.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of rf_lab is %g.' % rf_lab_accuracy)
    '''

    # This is for rf_rgb_model.
    # The following codes are adapted from https://ggbaker.ca/data-science/content/ml.html .
    # The following codes are adapted from https://www.datacamp.com/community/tutorials/random-forests-classifier-python .
    rf_hsv = make_pipeline(
        FunctionTransformer(myRGB2HSV),
        RandomForestClassifier(n_estimators = 100)
    )
    '''
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    rf_hsv.fit(X_train, y_train)
    rf_hsv_accuracy = rf_hsv.score(X_valid, y_valid, sample_weight = None)
    print('The accuracy score of rf_hsv is %g.' % rf_hsv_accuracy)
    '''

    # After checking, converting to LAB of rf model is a better choice! (10 sample one-tailed t test:p-value = 0.03618 alternative hypothesis: true mean is greater than 0 )
    
    # train each model and output image of predictions
    models = [bayes_rgb_model, bayes_convert_model, knn_rgb_model, knn_convert_model, rf_rgb_model, rf_convert_model]
    for i, m in enumerate(models):  # yes, you can leave this loop in if you want.
        m.fit(X_train, y_train)
        plot_predictions(m)
        plt.savefig('predictions-%i.png' % (i,))

    print(OUTPUT_TEMPLATE.format(
        bayes_rgb=bayes_rgb_model.score(X_valid, y_valid),
        bayes_convert=bayes_convert_model.score(X_valid, y_valid),
        knn_rgb=knn_rgb_model.score(X_valid, y_valid),
        knn_convert=knn_convert_model.score(X_valid, y_valid),
        rf_rgb=rf_rgb_model.score(X_valid, y_valid),
        rf_convert=rf_convert_model.score(X_valid, y_valid),
    ))
    


if __name__ == '__main__':
    main()
