import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import pandas as pd

from data_loading.data_cleaner import get_cleaned_data


def objective(space):
    model=xg.XGBRegressor(
                    n_estimators =int(space['n_estimators']), max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']),
                    colsample_bytree=int(space['colsample_bytree']), learning_rate = space['learning_rate'])
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    model.set_params(eval_metric=["error", "rmse"], early_stopping_rounds=10)
    model.fit(X_train, y_train, eval_set=evaluation,verbose=False)
    

    y_predicted = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_predicted, squared=False)
    print ("score:", rmse)
    return {'loss': rmse, 'status': STATUS_OK }


if __name__=="__main__":

    #Load the cleaned data
    df = get_cleaned_data('./data_loading/data_full.csv', './data_loading/teams.txt', date=False)
    df_date_only = get_cleaned_data('./data_loading/data.csv', './data_loading/teams.txt')['days_elapsed']

    df['days_elapsed'] = df_date_only

    #Create the data matrix and target vector
    X = df.drop(columns=['target', 'result_run_home', 'reuslt_run_away'])
    y = df[['result_run_home', 'reuslt_run_away']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train)
    print(y_train)

    #Test hyperparam optimization
    space={'max_depth': hp.quniform("max_depth", 3, 20, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 30,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': hp.quniform('n_estimators', 100, 200, 1),
        'learning_rate': hp.uniform('learning_rate', 0.1, 0.2),
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

    #Define model
    model = xg.XGBRegressor(n_estimators=int(best_hyperparams['n_estimators']), max_depth = int(best_hyperparams['max_depth']), gamma = best_hyperparams['gamma'],
                    reg_alpha = int(best_hyperparams['reg_alpha']),min_child_weight=int(best_hyperparams['min_child_weight']),
                    colsample_bytree=int(best_hyperparams['colsample_bytree']))
    
    model.set_params(eval_metric=["error", "logloss", "rmse"],)
    model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

    #Plot feature importance
    xg.plot_importance(model)


    results = model.evals_result()
    epochs = len(results['validation_0']['error'])

    y_predicted = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_predicted, squared=False)
    print(f'RMSE Between predicted y and test y is {rmse}')

    x_axis = range(0, epochs)
  
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Evaluation')
    ax.legend()
    plt.ylabel('RMSE')
    plt.xlabel('Epoch')
    plt.title('XGBoost RMSE')
    

    plt.show()