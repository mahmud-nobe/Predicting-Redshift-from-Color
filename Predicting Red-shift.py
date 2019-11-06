import numpy as np
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
    import numpy as np
    data = np.load('sdss_galaxy_colors.npy')
    
    features = np.zeros((data.shape[0], 4))
    features[:, 0] = data['u']-data['g']
    features[:, 1] = data['g']-data['r']
    features[:, 2] = data['r']-data['i']
    features[:, 3] = data['i']-data['z']

    targets = data['redshift']  
    return (features, targets)

#
# The function should calculate the median residual error of our model, 
# i.e. the median difference between our predicted and actual redshifts.
# and return the median of their absolute differences.
#

# paste your median_diff function here
def median_diff(pred, test):
    return np.median(abs(pred-test))

# write a function that splits the data into training and testing subsets
def split_data(features, targets):
    split = features.shape[0]//2
    train_features = features[:split]
    test_features = features[split:]
    train_targets = targets[:split]
    test_targets = targets[split:]
    return (train_features, test_features, train_targets, test_targets)
    

# The function should take 3 arguments:
#
# model: the decision tree regressor;
# features - the features for the data set;
# targets - The targets for the data set.  
#
# Finally, it should measure the accuracy of the model 
# using median_diff on the test_targets and the predicted redshifts from test_features.
#
    
# trains the model and returns the prediction accuracy with median_diff
def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  train_features, test_features, train_targets, test_targets = split_data(features, targets)
    
  # train the model
  dtr = DecisionTreeRegressor()
  dtr.fit(train_features, train_targets)
    
  # get the predicted_redshifts
  predictions = dtr.predict(test_features)

  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)


if __name__ == "__main__":
  data = np.load('sdss_galaxy_colors.npy')
  features, targets = get_features_targets(data)

  # initialize model
  dtr = DecisionTreeRegressor()

  # validate the model and print the med_diff
  diff = validate_model(dtr, features, targets)
  print('Median difference: {:f}'.format(diff))
