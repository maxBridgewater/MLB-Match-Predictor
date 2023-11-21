import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import numpy as np

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
    x = np.random.uniform(-10, 10, num_points)

    # Evaluate the polynomial at x values to get y values
    y = np.polyval(coefficients, x)

    return x, y, coefficients

def make_training_data(N, num_points):
    X = np.zeros((N, num_points))
    Y = np.zeros((N, num_points))

    for i in range(N):
        degree = 3#np.random.randint(2, 10)
        x_row, y_row, coefs = generate_random_polynomial(degree, num_points, (-10, 10)) 
        X[i] = x_row
        Y[i] = y_row

    return X, Y


if __name__=="__main__":

    #Make random polynomial data
    X, y = make_training_data(1000, 25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Deine model
    model = xg.XGBRegressor()
    #model.fit(X_train, y_train)
    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # force scores to be positive
    scores = np.abs(scores)
    print('Mean MAE: %.3f (%.3f)' % (scores.mean(), scores.std()) )

    #predict
    #y_predicted = model.predict(X_test)

    #mae_test = np.sum((y_predicted - y_test)**2) * (1 / y_test.shape[0])

    #print(f'MAE Between predicted y and test y is {mae_test}')



    

