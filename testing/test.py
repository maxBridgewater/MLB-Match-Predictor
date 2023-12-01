import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pandas as pd


def generate_random_polynomial(degree=3, num_points=50, coefficient_range=(-10, 10)):
    """
    Generate a random polynomial. CHAT GENERATED 

    Parameters:
    - degree: The degree of the polynomial.
    - num_points: The number of points to generate for (x, y) pairs.
    - coefficient_range: Tuple specifying the range of coefficients for the polynomial.

    Returns:
    - x: 1D NumPy array containing x values.
    - y: 1D NumPy array containing corresponding y values.
    - coefficients: 1D NumPy array containing the coefficients of the polynomial.
    """

    # Generate random coefficients for the polynomial
    coefficients = np.random.uniform(coefficient_range[0], coefficient_range[1], size=degree + 1)

    # Generate random x values
    x = np.linspace(-10, 10, num_points)

    # Evaluate the polynomial at x values to get y values
    y = np.polyval(coefficients, x)

    return x, y, coefficients

def make_training_data(N, num_points):
    X = np.zeros((N, num_points))

    degree = 3
    Y = np.zeros((N, degree + 1))

    for i in range(N):
        
        x_row, y_row, coefs = generate_random_polynomial(degree, num_points, (-10, 10)) 
        X[i] = y_row
        Y[i] = coefs

    return X, Y

def objective(space):
    model=xg.XGBRegressor(
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']))
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model.set_params(eval_metric=["error", "logloss"], early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=evaluation,verbose=False)
    

    y_predicted = model.predict(X_test)
    rmse_test = np.sum((y_predicted - y_test)**2) * (1 / y_test.shape[0])
    print ("rmse SCORE:", rmse_test)
    return {'loss': rmse_test, 'status': STATUS_OK }


if __name__=="__main__":

    #Make random polynomial data
    X, y = make_training_data(1000, 50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Test hyperparam optimization
    space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180,
        'seed': 0
    }

    trials = Trials()

    best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
    

    print("The best hyperparameters are : ","\n")
    print(best_hyperparams)



    
    #Define evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]

    #Deine model
    model = xg.XGBRegressor(n_estimators=180, max_depth = int(best_hyperparams['max_depth']), gamma = best_hyperparams['gamma'],
                    reg_alpha = int(best_hyperparams['reg_alpha']),min_child_weight=int(best_hyperparams['min_child_weight']),
                    colsample_bytree=int(best_hyperparams['colsample_bytree']))
    
    model.set_params(eval_metric=["error", "logloss", "rmse"],)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    results = model.evals_result()
    epochs = len(results['validation_0']['error'])

    # define model evaluation method
    #cv = RepeatedKFold(n_splits=2, n_repeats=3, random_state=1)
    # evaluate model
    #scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, verbose=True)
    # force scores to be positive
    #scores = np.abs(scores)
    #print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )


    #predict
    y_predicted = model.predict(X_test)

    x = np.linspace(-10, 10, 50)
    plt.figure()
    plt.plot(x, np.polyval(y_test[10], x), label='Actual Function')
    plt.plot(x, np.polyval(y_predicted[10], x), label='Predicted Function', linestyle='--')
    plt.legend()

    mae_test = np.sum((y_predicted - y_test)**2) * (1 / y_test.shape[0])
    print(f'MAE Between predicted y and test y is {mae_test}')

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


    plt.show()




    

