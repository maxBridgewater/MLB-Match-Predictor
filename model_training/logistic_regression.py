from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import learning_curve
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
    
    #Define model
    model = LogisticRegression(max_iter = 10000)

    #Create scorer
    scorer = make_scorer(accuracy_score)

    train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1)

    # Calculate mean and standard deviation of training set scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation of test set scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Train")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Evaluation")
    plt.legend(loc="best")

    model.fit(X_train, y_train)

    #Plot feature importance
    coefs = model.coef_[0]
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefs)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)
    feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))  
    plt.xlabel('Relative Importance')
    plt.title('Logistic Regression Feature Importance')


    y_predicted = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_predicted>0.5)
    print(f'Accuracy Between predicted y and test y is {accuracy}')

    plt.show()



    


