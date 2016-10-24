#This project is Walmart’s Kaggle challenge to classify 
#shopping trips using market basket analysis.  
#Ths was run on Enthought Canopy with Python 2.7
from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np
import time

import scipy as sp

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm


# Logarithmic Loss Function
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = (act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',', usecols=np.arange(0,7))
    return dataset
    
def loadcsv2(filename):
    dataset = np.genfromtxt(filename, delimiter=',', usecols=np.arange(0,6))
    return dataset

def loadcsv3(filename):
    dataset = np.genfromtxt(filename, delimiter=',', dtype=None, usecols=np.arange(0,7))
    return dataset
    
def loadcsv4(filename):
    dataset = np.genfromtxt(filename, delimiter=',', dtype=None, usecols=np.arange(0,6))
    return dataset
    
def savecsv(dataset, filename):
    np.savetxt(filename, dataset, delimiter=',')
    return 1  
 
def splitdataset(dataset, trainsize=645000, testsize=2000):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]
    Xtrain = dataset[randindices[0:trainsize],1:numinputs]
    ytrain = dataset[randindices[0:trainsize],0]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],1:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],0]

    n = ytrain.shape[0]
    temp = np.reshape(ytrain, (n, 1))
    ytrain = temp
    n = ytest.shape[0]
    temp = np.reshape(ytest, (n, 1))
    ytest = temp

    return ((Xtrain,ytrain), (Xtest,ytest))

def modifyDataset(dataset, dataset2):
    """
    Modifies one feature strings to integer
    """
    j = 100
    for i in range(len(dataset)):
        hash={}
        if dataset2[i][5] in hash: 
            dataset[i][5] = hash[dataset2[i][5]]
        else:
            hash[dataset2[i][5]] = j
            dataset[i][5] = j
            j += 1

    return (dataset)
    
def modifyTestset(dataset, dataset2):
    """
    Modifies one feature strings to integer
    """
    j = 1000
    for i in range(len(dataset)):
        hash={}
        if dataset2[i][4] in hash: 
            dataset[i][4] = hash[dataset2[i][4]]
        else:
            hash[dataset2[i][4]] = j
            dataset[i][4] = j
            j += 1

    return (dataset)
    

def splittestset(testset, testsize=653646):

    randindices = np.random.randint(0,testset.shape[0],testsize)
    numinputs = testset.shape[1]
    Xtest = testset[randindices[0:testsize],0:numinputs]
    
    return Xtest


def preprocess(Xtrain, ytrain, Xtest):
    """
    Preprocess the Train data
    """
        
    # Preprocess the data
    Xtrain = Imputer().fit_transform(Xtrain)
    Xtrain = preprocessing.robust_scale(Xtrain)
    Xtest = Imputer().fit_transform(Xtest)
    Xtest = preprocessing.robust_scale(Xtest)
    ytrain = Imputer().fit_transform(ytrain)
    #ytrain = preprocessing.robust_scale(ytrain)
    print ( "Checkinf for NaN and Inf")
    print ("np.inf=", np.where(np.isnan(Xtrain)))
    print ("is.inf=", np.where(np.isinf(Xtrain)))
    print ("np.max=", np.max(abs(Xtrain))) 
  
    for i in range(len(ytrain)):
        type = ytrain[i][0]
        if type == np.nan:
            np.delete(ytrain, i, 0)
            np.delete(Xtrain, i, 0)
        
    return Xtrain, ytrain, Xtest 


def preprocess2(Xtest):
    """
    Preprocess test real test data
    """
        
    # Preprocess the data
    Xtest = Imputer().fit_transform(Xtest)
    Xtest = preprocessing.robust_scale(Xtest)

    return Xtest 
    

def kNN(Xtrain, ytrain, Xtest):
    """
    k Nearest Neighbors
    """
    
    nn = 1
    knn = KNeighborsClassifier(n_neighbors=nn)    
    
    # Learn using the traindata 
    knn.fit(Xtrain, ytrain.ravel())
        
    ytest = knn.predict(Xtest)
    
    return ytest       
 
 
def randomForest(Xtrain, ytrain, Xtest):
    """
    Random Forest Classifier
    """
    
    nn = 100
    forest = RandomForestClassifier(n_estimators=nn)    
    
    # Learn using the traindata 
    forest=forest.fit(Xtrain, ytrain.ravel())
    ytest = forest.predict(Xtest)
    yProba = forest.predict_proba(Xtest)
    
    return ytest, yProba


def sgd(Xtrain, ytrain, Xtest):
    """
    Stochastic Gradient Descent
    """
    
    clf = SGDClassifier(loss='modified_huber', penalty='elasticnet', shuffle=True)    
    
    # Learn using the traindata 
    clf=clf.fit(Xtrain, ytrain.ravel())
    ytest = clf.predict(Xtest)
    
    yProba = clf.predict_proba(Xtest)
    
    return ytest, yProba
     
    
def decisionTree(Xtrain, ytrain, Xtest):
    """
    Decision Tree
    """
    
    clf = tree.DecisionTreeClassifier()  
    
    # Learn using the traindata 
    clf=clf.fit(Xtrain, ytrain.ravel())
        
    ytest = clf.predict(Xtest)
    
    return ytest   
      
      
def gnb(Xtrain, ytrain, Xtest):
    """
    Gaussian Naive Bayes
    """
    
    gnb = GaussianNB()  
    
    # Learn using the traindata 
    gnb.fit(Xtrain, ytrain.ravel())
        
    ytest = gnb.predict(Xtest)
    
    return ytest
    
    
def logisticReg(Xtrain, ytrain, Xtest):
    """
    Logistic Regression
    """
    
    logReg = LogisticRegression(C=1e5)  
    
    # Learn using the traindata 
    logReg.fit(Xtrain, ytrain.ravel())
        
    ytest = logReg.predict(Xtest)
    yProba = logReg.predict_proba(Xtest)
    #print "DBG", yProba[0]
    
    return ytest, yProba
 
    
def svm(Xtrain, ytrain, Xtest):
    """
    C-Support Vector Classifier
    """
    C=1.0
    clf = svm.SVC(kernel='linear', C=C)
    
    # Learn using the traindata 
    clf.fit(Xtrain, ytrain.ravel())
        
    ytest = clf.predict(Xtest)
    
    return ytest   
    
    
def getaccuracy(ytest, predictions):
    """
    Finds Accuracy of a Particular model
    """
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            #print "Predicted:", predictions[i], "Actual:", ytest[i]
            correct += 1
            
    return (correct/float(len(ytest))) * 100.0
    
    
def getCombined(combined, bestPredictions):
    """
    Finds combined prediction by max vote algorithm 
    """
    finalPredictions = np.zeros(len(bestPredictions))
    for i in range(len(bestPredictions)):
        count = {}
        for learner, predictions in combined.iteritems():
            prediction = int(predictions[i])
            if prediction in count:
                count[prediction] += 1
            else:
                count[prediction] = 1
        #Find the max voted one
        maxNo = 0
        for prediction, no in count.iteritems():
            if no > maxNo:
                maxNo = no
                maxVotedPredict = prediction    
        
        # Set final prediction vector, in case of tie select bestPrediction
        if (maxNo == 1):
            finalPredictions[i] = bestPredictions[i]
        else:    
            finalPredictions[i] = maxVotedPredict
        
    return finalPredictions
 
 
def summarizeAct(act, probs, Xtrain):
    """
    Walmart data has several rows per single trip
    but the output file needs to have trip type prediction per single visit.
    So output predictions need to be summarized from combined predictions
    """
    
    #Hardcode
    finalActs = np.zeros((95674, 38))
    #finalProbs = np.zeros((191348, 38))
    finalProbs = np.zeros((95674, 38))
    
    i = 0
    j = 0
    while ((i < 95674) and (j < Xtrain.shape[0])):
        A = np.zeros(38)
        P = np.zeros(38)
        initVisit = Xtrain[j][0]
        A = act[j]
        anchor = j
        loop = True
        
        # Calculations for finalActs
        while loop:
            j = j + 1
            curVisit = Xtrain[j][0]
            if (initVisit == curVisit):
                A = A + act[j]
            else:
                #Calculate final rows 
                maxA = 0
                indx1 = 0
                for k in range(38):
                    if A[k] > maxA:
                        maxA = A[k]
                        indx1 = k
                finalActs[i][indx1] = 1
                loop = False
                
        #Calculations for finalProbs
        r = anchor
        maxP = 0
        indx2 = r
        while r < j:
            if (probs[r][indx1] > maxP):
                maxP = probs[r][indx1]
                indx2 = r
            r += 1
            
        finalProbs[i] = probs[indx2]
        
        i += 1
            
    return finalActs,finalProbs

 
if __name__ == '__main__':
    filename = 'train.csv'
    filename2 = 'test.csv'
    filename3 = 'result.csv'
    
    dataset = loadcsv(filename)
    dataset2 = loadcsv3(filename)
    dataset = modifyDataset(dataset, dataset2)
    trainset, testset = splitdataset(dataset)
    
    realTestset = loadcsv2(filename2)
    realTestset2 = loadcsv4(filename2)
    realTestset = modifyTestset(realTestset, realTestset2)
    realTestset = splittestset(realTestset)

    # Preprocess the train and test data
    Xtrain, ytrain, Xtest = preprocess(trainset[0], trainset[1], testset[0])
    
    classalgs = {
                 'Random Forest': randomForest,
                 'Stochastic Gradient Descent': sgd,
                 'Logistic Regression': logisticReg
                 }
                 
    ######## Run the Models on Train Data  #########
    combined = {}
    for learnername, learner in classalgs.items():
    
        print ('Running model ' + learnername)
        
        # Train and Test model
        predictions, probs = learner(Xtrain, ytrain, Xtest)
        combined[learner] = predictions
        if (learnername == 'Logistic Regression'):
            bestPredictions = predictions
        
        # Check accuracy
        accuracy = getaccuracy(testset[1], predictions)
        print ('Accuracy for ' + learnername + ': ' + str(accuracy))
      
    # Combine all model predictions
    combinedPredictions = getCombined(combined, bestPredictions)
    
    # Check combined accuracy
    accuracy = getaccuracy(testset[1], combinedPredictions)
    print ('Combined Accuracy for ' + ': ' + str(accuracy))
    
    
    ######## Run the Models on real Test Data #########
    realTestset = preprocess2(realTestset)  
    combined = {}
    for learnername, learner in classalgs.iteritems():
    
        print ('Running model ' + learnername)
        
        # Train and Test on different models
        predictions, probs = learner(Xtrain, ytrain, realTestset)
        combined[learner] = predictions
        if (learnername == 'Logistic Regression'):
            bestPredictions = predictions
            bestProbs = probs
            
    # Combine all models predictions
    combinedPredictions = getCombined(combined, bestPredictions)
  
    act = np.zeros((bestProbs.shape[0], bestProbs.shape[1]))
    
    #Classes are not in sequnce, they need to be hardcode
    hardTypes =[3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 999]
    
    #Fill up act matrix with zero or ones as per the predicted class 
    for i in range(bestProbs.shape[0]):
        for j in range(len(hardTypes)):
            if (int(combinedPredictions[i]) == hardTypes[j]):
                act[i][j] = 1
    
    act, probs = summarizeAct(act, bestProbs, Xtrain)
    
    #Use log loss function as per submission requirement
    ll = logloss(act, probs)
        
    # Move to the Submission CSV file
    print ("Saving into submission file...")
    z = savecsv(ll, filename3)
