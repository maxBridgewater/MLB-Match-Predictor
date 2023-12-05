import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_squared_error

from data_loading.data_cleaner import get_cleaned_data

if __name__=="__main__":

    #Load the cleaned data
    df = get_cleaned_data('./data_loading/data_full.csv', './data_loading/teams.txt', date=False)
    df_date_only = get_cleaned_data('./data_loading/data.csv', './data_loading/teams.txt')['days_elapsed']

    df['days_elapsed'] = df_date_only

    #Create the data matrix and target vector
    X = df.drop(columns=['target', 'result_run_home', 'reuslt_run_away'])
    y = df[['result_run_home', 'reuslt_run_away']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Calculate the mean of the training target vector
    y_t_mean = np.mean(y_train, axis=0)

    means = np.ones(y_test.shape)

    means[:, 0] = y_t_mean[0]
    means[:, 1] = y_t_mean[1]


    print(means)
    print(y_test)

    #rmse compared to test data
    rmse = mean_squared_error(y_test, means, squared=False)
    print(f'RMSE Between predicted y and test y is {rmse}')

    