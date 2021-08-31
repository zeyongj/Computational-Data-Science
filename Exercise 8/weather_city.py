import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# The following code is adapted from colour_predict_hint.py of this exercise.
OUTPUT_TEMPLATE = (
    'The score of the selected model is:    {model_score:g}'
)


def main():
    monthly_data_labelled = pd.read_csv(sys.argv[1])
    monthly_data_unlabelled = pd.read_csv(sys.argv[2])

    # The following codes are inspired from https://zhuanlan.zhihu.com/p/89464360 (Chinese Version).
    X = monthly_data_labelled.iloc[:, 1:] # First two colums are needed.
    y = monthly_data_labelled.iloc[:,0] # The column of label (i.e. first column) is needed.

    # The following codes are adapted from my previous assignment.
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # The following codes are adapted from https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html .
    # The following codes are also adapted from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html and https://www.datacamp.com/community/tutorials/random-forests-classifier-python .
    from sklearn.preprocessing import FunctionTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.naive_bayes import GaussianNB
    model_bayes = make_pipeline(
        StandardScaler(),
        GaussianNB()
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model_bayes.fit(X_train, y_train)
    model_bayes_accuracy = model_bayes.score(X_valid, y_valid, sample_weight = None)
    # print('The accuracy score of model_bayes is %g.' % model_bayes_accuracy)
    # AVG = 0.635172

    
    from sklearn.neighbors import KNeighborsClassifier
    model_knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors = 8)
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model_knn.fit(X_train, y_train)
    model_knn_accuracy = model_knn.score(X_valid, y_valid, sample_weight = None)
    # print('The accuracy score of model_knn is %g.' % model_knn_accuracy)
    # AVG = 0.671724

    from sklearn.ensemble import RandomForestClassifier
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators = 100)
    )
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)
    model.fit(X_train, y_train)
    model_score = model.score(X_valid, y_valid, sample_weight = None)
    # AVG = 0.731034

    # All the above values of AVG are calcuted after 10 executions. 
 
    print(OUTPUT_TEMPLATE.format(
        model_score = model.score(X_valid, y_valid)
    ))

    # The following code is inspired from https://zhuanlan.zhihu.com/p/89464360 (Chinese Version).
    data = monthly_data_unlabelled.iloc[:, 1:] # The column of city (i.e. second column) is needed.
    
    predictions = model.predict(data)
    
    # The following code is copied from the instruction.
    pd.Series(predictions).to_csv(sys.argv[3], index = False, header = False)

    # df = pd.DataFrame({'truth': y_valid, 'prediction': model.predict(X_valid)})
    # print(df[df['truth'] != df['prediction']])
    # The above two lines of comment are for the answers.txt .
    

if __name__ == '__main__':
    main()
