import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from data_loading.data_cleaner import get_cleaned_data



if __name__=="__main__":

    #Load the cleaned data
    df = get_cleaned_data('./data_loading/data.csv', './data_loading/teams.txt')

    #Create the data matrix and target vector
    X = df.drop(columns=['target'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train)

    model = LogisticRegression(max_iter = 1000)
    #model.set_params(eval_metric=["error", "logloss", "rmse"],)

    model.fit(X_train, y_train)

    #results = model.evals_result()
    #epochs = len(results['validation_0']['error'])


    y_predicted = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted>0.5)
    print(f'Accuracy Between predicted y and test y is {accuracy}')

    '''
    x_axis = range(0, epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGBoost Log Loss')

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend()
    plt.ylabel('RMSE')
    plt.title('XGBoost RMSE')
    '''
    plt.show()



    


