#This is the main driver code to classify the test dataset using 
#multiple classification algorithms implemented in classalgorithms.py
import csv
import random
import math
import numpy as np
import time

import classalgorithms as algs

start_time = time.time()
 
def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset
 
def splitdataset(dataset, trainsize=2000, testsize=1000):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    Xtrain0 = Xtrain
    Xtest0 = Xtest
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest), (Xtrain0,ytrain), (Xtest0,ytest))
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0
    
 
if __name__ == '__main__':
    filename = 'susysubset.csv'
    dataset = loadcsv(filename)
    trainset, testset, trainset0, testset0 = splitdataset(dataset)
    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    classalgs = {'Random': algs.Classifier(),
                 'Linear Regression': algs.LinearRegressionClass(),
                 'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 'Naive Bayes Ones': algs.NaiveBayes(),
                 'Logistic Regression': algs.LogitReg(),
                 'Neural Network': algs.NeuralNet(nnparams),
                 'Elastic Logit Regression': algs.ElasticReg()
                 }
    
    for learnername, learner in classalgs.items():
        print ('Running learner = ' + learnername)
        # Train model
        if learnername != 'Naive Bayes':
            learner.learn(trainset[0], trainset[1])
        else:
            learner.learn(trainset0[0], trainset0[1])
        
        # Test model
        if learnername != 'Naive Bayes':
            predictions = learner.predict(testset[0])
        else:
            predictions = learner.predict(testset0[0])
        
        # Check accuracy
        if learnername != 'Naive Bayes':
            accuracy = getaccuracy(testset[1], predictions)
        else:
            accuracy = getaccuracy(testset0[1], predictions)
        
        print ('Accuracy for ' + learnername + ': ' + str(accuracy))
        print ('Time :', (time.time() - start_time))
 
