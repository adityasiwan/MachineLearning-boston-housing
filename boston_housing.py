
import numpy as np
import pandas as pd
import visuals as vs
from sklearn.cross_validation import ShuffleSplit

get_ipython().magic(u'matplotlib inline')

data = pd.read_csv('housing.csv')
prices = data['MDEV']
features = data.drop('MDEV', axis = 1)
    
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)


minimum_price = np.amin(prices)

maximum_price = np.amax(prices)

mean_price = np.mean(prices)

median_price = np.median(prices)

std_price = np.std(prices)

print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    performance_score=r2_score(y_true, y_predict)
    
    score = performance_score
    return score

score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)
from sklearn.cross_validation import train_test_split
X=features
y=prices
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80, random_state=42)

print "Training and testing split was successful."


vs.ModelLearning(features, prices)


vs.ModelComplexity(X_train, y_train)

from sklearn.tree import DecisionTreeRegressor 
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV


def fit_model(X, y):
    cv_sets = ShuffleSplit(X.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
    regressor = DecisionTreeRegressor()
    params = {"max_depth":range(1,10)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_

reg = fit_model(X_train, y_train)

print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])
client_data = [[5, 34, 15], # Client 1
               [4, 55, 22], # Client 2
               [8, 7, 12]]  # Client 3

for i, price in enumerate(reg.predict(client_data)):
    print "Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price)

vs.PredictTrials(features, prices, fit_model, client_data)