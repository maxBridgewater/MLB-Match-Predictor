from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from data_loading.data_cleaner import get_cleaned_data

if __name__=="__main__":

    #Load the cleaned data
    df = get_cleaned_data('./data_loading/data_full.csv', './data_loading/teams.txt', date=False)
    df_date_only = get_cleaned_data('./data_loading/data.csv', './data_loading/teams.txt')['days_elapsed']

    df['days_elapsed'] = df_date_only

    df = df.apply(lambda x:x.fillna(x.mean(), inplace = True))

    #Create the data matrix and target vector
    X = df.drop(columns=['target', 'result_run_home', 'reuslt_run_away'])
    y = df[['result_run_home', 'reuslt_run_away']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nan_locations = X_train.isna()

    print(f'nan locations are: {nan_locations}')

    print(X_train)
    print(y_train)

    model = LinearRegression()

    model.fit(X_train, y_train)

    #results = model.evals_result()
    #epochs = len(results['validation_0']['error'])


    y_predicted = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_predicted, squared=False)
    print(f'RMSE Between predicted y and test y is {rmse}')

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