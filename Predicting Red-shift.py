# Importing necessary packages and libraries
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Checking the datasets
data = np.load('sdss_galaxy_colors.npy')
df = pd.DataFrame(data)
df.head()


# From the dataset, defining the color differences as our predictors (i.e. features) and
# the redshift as our response variable (i.e. target).

def get_features_targets(data):
    
    features = np.zeros((data.shape[0], 4))
    features[:, 0] = data['u']-data['g']
    features[:, 1] = data['g']-data['r']
    features[:, 2] = data['r']-data['i']
    features[:, 3] = data['i']-data['z']

    targets = data['redshift']  
    return (features, targets)

# Splitting the dataset into one train and one test subset
def split_data(features, targets):
    split = features.shape[0]//2
    train_features = features[:split]
    test_features = features[split:]
    train_targets = targets[:split]
    test_targets = targets[split:]
    return (train_features, test_features, train_targets, test_targets)

# Calculating the median residual error of our model, 
# i.e. the median of the difference between our predicted and actual redshifts.
# We will use this to test the accuracy and effectivity of our model.

def median_diff(pred, test):
    return np.median(abs(pred-test))

# Calculating the MSE, mean squared error for similar reason

def mse(pred, test):
    return np.mean((pred-test)**2)


## Cheking the accuracy of Decision tree model using one test and one train set

# get the train and test subset
features, targets = get_features_targets(data)
train_features, test_features, train_targets, test_targets = split_data(features, targets)
    
# train the model
dtr = DecisionTreeRegressor()
dtr.fit(train_features, train_targets)
    
# get the predicted_redshifts
predictions = dtr.predict(test_features)

# use median_diff function to calculate the accuracy
print("MSE:", mse(predictions,test_targets))
print("Diff:", median_diff(predictions,test_targets))


# Estimate the test set MSE with 10-fold cross validation
# w/o randomized fold selection.

n = len(data)
total_mse = 0

for i in range(10):
    test_x = []
    test_y = []
    train_x = []
    train_y = []
    
    # selecting test index
    test = range(i*(int(n/10)),(i+1)*(int(n/10)))
    
    test_x = features[test]
    test_y = targets[test]
    
    for j in range(n):
        if j not in test:
            train_x.append(features[j])
            train_y.append(targets[j])
    
    # train the model
    dtr = DecisionTreeRegressor()
    dtr.fit(train_x, train_y)
    
    # get the predicted_redshifts
    predictions = dtr.predict(test_x)

    # use median_diff function to calculate the accuracy
    MSE = mse(predictions,test_y)
    diff = median_diff(predictions,test_y)
    print("\nModel ",i,":")
    print("MSE:", MSE)
    print("Diff:", diff,'\n')
    
    total_mse = total_mse + MSE
        
print("Cross Validation Estimation of test MSE: ", total_mse/10)


# Estimate the test set MSE with 10-fold cross validation
# w/ randomized fold selection.

n = len(data)
total_mse = 0
index = (range(n))

for i in range(10):
    test_x = []
    test_y = []
    train_x = []
    train_y = []

    # selecting test index randomly
    test = random.sample(index,int(n/10))
    # exclude the selected indices from the overall range
    index = list(set(index) - set(test))
    
    test_x = features[test]
    test_y = targets[test]
    
    for j in range(n):
        if j not in test:
            train_x.append(features[j])
            train_y.append(targets[j])
    
    # train the model
    dtr = DecisionTreeRegressor()
    dtr.fit(train_x, train_y)
    
    # get the predicted_redshifts
    predictions = dtr.predict(test_x)

    # use median_diff function to calculate the accuracy
    MSE = mse(predictions,test_y)
    diff = median_diff(predictions,test_y)
    print("\nModel ",i,":")
    print("MSE:", MSE)
    print("Diff:", diff,'\n')
    
    total_mse = total_mse + MSE
        
print("Cross Validation Estimation of test MSE: ", total_mse/10)


# Estimate the test set MSE with k-fold cross validation
# w/ randomized fold selection.

def cross_validation(k):
    if k==0 or k==1:
        return 0
    n = len(data)
    total_mse = 0
    index = (range(n))

    for i in range(k):
        test_x = []
        test_y = []
        train_x = []
        train_y = []

        # selecting test index randomly
        test = random.sample(index,int(n/k))
        # exclude the selected indices from the overall range
        index = list(set(index) - set(test))

        test_x = features[test]
        test_y = targets[test]

        for j in range(n):
            if j not in test:
                train_x.append(features[j])
                train_y.append(targets[j])

        # train the model
        dtr = DecisionTreeRegressor()
        dtr.fit(train_x, train_y)

        # get the predicted_redshifts
        predictions = dtr.predict(test_x)

        # use median_diff function to calculate the accuracy
        MSE = mse(predictions,test_y)
        diff = median_diff(predictions,test_y)
        #print("\nModel ",i,":")
        #print("MSE:", MSE)
        #print("Diff:", diff,'\n')

        total_mse = total_mse + MSE

    print(k,"fold Cross Validation Estimation of test MSE: ", total_mse/k)
    return total_mse/k


## Use random forest to create and evaluate new model

from sklearn.ensemble import RandomForestRegressor

# Estimate the test set MSE with k-fold random forest cross validation
# w/ randomized fold selection.

def random_forest_cross_validation(k):
    if k==0 or k==1:
        return 0
    n = len(data)
    total_mse = 0
    index = (range(n))

    for i in range(k):
        test_x = []
        test_y = []
        train_x = []
        train_y = []

        # selecting test index randomly
        test = random.sample(index,int(n/k))
        # exclude the selected indices from the overall range
        index = list(set(index) - set(test))

        test_x = features[test]
        test_y = targets[test]

        for j in range(n):
            if j not in test:
                train_x.append(features[j])
                train_y.append(targets[j])

        # train the model
        rfr = RandomForestRegressor(n_estimators = 100)
        rfr.fit(train_x, train_y)

        # get the predicted_redshifts
        predictions = rfr.predict(test_x)

        # use median_diff function to calculate the accuracy
        MSE = mse(predictions,test_y)
        diff = median_diff(predictions,test_y)
        #print("\nModel ",i,":")
        #print("MSE:", MSE)
        #print("Diff:", diff,'\n')

        total_mse = total_mse + MSE

    print(k,"fold Cross Validation Estimation of test MSE: ", total_mse/k)
    return total_mse/k

random_forest_cross_validation(2)
random_forest_cross_validation(10)

# Use Random Forest on whole dataset and calculate the R^2 value 
# considering the out of bag (OOB) samples

rfr = RandomForestRegressor(n_estimators = 100, oob_score = True)
rfr.fit(features, targets)
rfr.score(features, targets)
# 0.9667004848145836
