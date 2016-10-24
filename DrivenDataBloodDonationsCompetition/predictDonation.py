# This program was run on Spyder with Python 3.5
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation
from sklearn import preprocessing


# Logarithmic Loss Function
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = (act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll


# Read train, test and results csv files
df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
#df3 = pd.read_csv('results.csv')

# Plot and see the correlation between different features and target
for col in (df.columns):
    plt.scatter(df['Made Donation in March 2007'], df[col])
    #plt.scatter(df['id'], df[col])
    plt.show()


# Split columnwise between features and target
# Created new feature 'Months per Donation'
# Ignore features - volume and Months since first donation 
# Re-ran the model with different feature combaination
X = df.ix[:,[1, 2, 5]]
#print (X.shape)
#print (X.head(5))
X = preprocessing.robust_scale(X)

# Get the target vector
y = df.ix[:,6]
#print (y.shape)
#print (y.head(5))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# Tried with different models
clf = linear_model.LogisticRegression(C=1e5)    
#clf = RandomForestClassifier(n_estimators=100)
#clf = KNeighborsClassifier(n_neighbors=1)    

# Fit the model
clf.fit(X_train, y_train)

# Grid Search to get best params
#param_grid = dict(C=[0.001, 0.01, 0.1, 1, 10, 100, 1000])
#grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
#grid.fit(X_train, y_train)

#print (grid.best_score_)
#print (grid.best_params_)
#print (grid.best_estimator_)

# Predict on the test data
y_pred = clf.predict(X_test)
print("Accuracy : %f" % accuracy_score(y_test, y_pred))
y_proba = clf.predict_proba(X_test)
#print (y_pred)
#print (y_proba)


# Now get the actual test data
X_test = df2.ix[:,[1, 2, 5]]
#print (X_test.shape)
#print (X_test.head(5))
X_test = preprocessing.robust_scale(X_test)

# predict on the actual test data
y_pred = clf.predict(X_test)

# Get the probability values
y_proba = clf.predict_proba(X_test)
#print (y_pred)
#print (y_proba)

probs = np.zeros(y_proba.shape[0])
# We need to pick the probability if target is 1
probs = y_proba[:,1]

#Use log loss function as per submission requirement
ll = logloss(y_pred, probs)
#print(ll)
#print (ll.shape[0])

      
# Save in the Submission CSV file
print ("Saving into submission file...")
df3[[1]] = ll
df3['Made Donation in March 2007'] = ll
#print(df3.shape)
#print(df3.head(5))
df3.to_csv("Results.csv")

