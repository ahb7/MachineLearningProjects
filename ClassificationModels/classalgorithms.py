#Multiple Classification algorithms are implemented here

from __future__ import division  # floating point division
import scipy
import numpy as np
import utilities as utils

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        """ probs is a column of random probabilities """
        ytest = utils.threshold_probs(probs)
        return ytest


class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

        yt = np.copy(ytrain)
        yt[yt == 0] = -1
         
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
  
              
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.gnb = None
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
    

    def separateByClass(self, Xtrain, ytrain):
        """ Separating Xtrain Dataset by class 0 or 1 """
        separated = {}
        for i in range(len(Xtrain)):
            vector = Xtrain[i]
            if (ytrain[i] not in separated):
                separated[ytrain[i]] =[]
            separated[ytrain[i]].append(vector)
        return separated
        

    def calculateClassProbabilities(self, summaries, inputVector):
        """ Calculates for each class the probability that the input in that class """
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= utils.calculateprob(x, mean, stdev)
        return probabilities
                            
                                     
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        """ Creates mean and standard deviation for each attribute for each class """
        self.summaries = {} 
        separated = self.separateByClass(Xtrain, ytrain)
        for classValue, instances in separated.items():
            summary = [(utils.mean(attribute), utils.stdev(attribute)) for attribute in zip(*instances)] 
            self.summaries[classValue] = summary
        
 
    def predict(self, Xtest):
        """ Calculates probability for each input for each class """
        """ and classifies into the class with highest probability """
        predictions = []
        summaries2 = self.summaries
        for inputVector in Xtest:
            probabilities = self.calculateClassProbabilities(summaries2, inputVector)
            bestLabel, bestProb = None, -1
            for classValue, probability in probabilities.items():
                if bestLabel is None or probability > bestProb:
                    bestProb = probability
                    bestLabel = classValue
            predictions.append(bestLabel)
        return predictions
                   
    
class LogitReg(Classifier):
    """ Logistic regression """

    def __init__( self, params=None ):
        self.weights = None
        

    def learn(self, Xtrain, ytrain):
        """ Calculates parameter vector w """
        
        # No of input samples
        m = Xtrain.shape[0]
        
        # No of features
        d = Xtrain.shape[1]
        
        # Initialize parameter vector w to zeros
        self.w = np.zeros( d )
        delta = np.ones( d )
        
        self.w = ((np.linalg.inv((Xtrain.T).dot(Xtrain))).dot(Xtrain.T)).dot(ytrain)  
        delta = abs(self.w)
        
        x = 1000
        while x > 0 :
            
            pi = np.zeros(m)
            P = np.zeros((m, m))
            # I is an n×n identity matrix. 
            I = np.identity(m)
                    
            # P is an n×n diagonal matrix with P[i][i] = pi 
            i = 0
            for i in range(m):
                P[i][i] = utils.sigmoid(Xtrain[i].dot(self.w.T))
                pi[i] = P[i][i]
            
            # The Hessian matrix H_ (without - sign) can now be calculated 
            H_ = (((Xtrain.T).dot(P)).dot(I-P)).dot(Xtrain)
            self.w = self.w + ((np.linalg.inv(H_)).dot(Xtrain.T)).dot(ytrain-pi)
            
            # Reduce delta
            x = x - 1
            delta = abs(delta - self.w)
                        
 
    def predict(self, Xtest):
        """ Calculates probability for each input using sigmoid function """
        """ and if probability is > 0.5 class 1, or class 0 """
        predictions = []
        for inputVector in Xtest:
            probability = utils.sigmoid(inputVector.dot(self.w.T))
            if probability > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)   
        return predictions
                        

class NeuralNet(Classifier):
    """ Two-layer neural network """
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid

        # Set step-size
        self.stepsize = 0.01

        # Number of repetitions over the dataset
        self.reps = 5
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """
        #i = 1        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp],ytrain[samp])
                #self.stepsize = self.stepsize/i
                #i = i + 1
               
    def predict(self, Xtest):
        """ Calculates probability for each input using evaluate function """
        """ and if probability is > 0.5, class = 1, else class = 0 """
        predictions = []
        for inputVector in Xtest:
            ah, probability = self.evaluate(inputVector)
            if probability > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)   
        return predictions
        
    def evaluate(self, inputs):
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)

    def update(self, inp, out):
        # Change to matrix format dx1
        f = inp.shape[0]
        inp = np.reshape(inp, (f,1))
            
        h = utils.sigmoid(self.wi.dot(inp))
        yTilda = utils.sigmoid(self.wo.dot(h))
        
        deltaI = ((-out/yTilda)+(1-out)/(1-yTilda))*yTilda*(1-yTilda)
        
        gradientWo = deltaI*(h.T)
        gradientWi = deltaI*((((self.wo).T)*(h*(1-h))).dot(inp.T))
        
        self.wo = self.wo - self.stepsize * gradientWo
        self.wi = self.wi - self.stepsize * gradientWi
        
        
class ElasticReg(Classifier):
    """ Logistic regression using Elastic Net Regulizer """

    def __init__( self, params=None ):
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Calculates parameter vector w """
        
        # No of input samples
        m = Xtrain.shape[0]
        
        # No of features
        d = Xtrain.shape[1]
        
        # Initialize parameter vector w to zeros
        self.w = np.zeros( d )
        delta = np.ones( d )
        # Create a threshold vector of 0.01
        alpha = 0.001
        
        self.w = ((np.linalg.inv((Xtrain.T).dot(Xtrain))).dot(Xtrain.T)).dot(ytrain)  
        delta = abs(self.w)
        
        x = 1000
        while x > 0 :
            
            pi = np.zeros(m)
            P = np.zeros((m, m))
            # I is an n×n identity matrix. 
            I = np.identity(m)
                    
            # P is an n×n diagonal matrix with P[i][i] = pi 
            i = 0
            for i in range(m):
                P[i][i] = utils.sigmoid(Xtrain[i].dot(self.w.T))
                pi[i] = P[i][i]
            
            # The Hessian matrix H_ (without - sign) can now be calculated 
            H_ = (((Xtrain.T).dot(P)).dot(I-P)).dot(Xtrain)
            # Calculate Elastic Penalty
            elastic = (1-alpha)*(self.w) + alpha*(self.w)*(self.w)
            self.w = self.w + ((np.linalg.inv(H_)).dot(Xtrain.T)).dot(ytrain-pi) - elastic
            
            # Reduce delta
            x = x - 1
            delta = abs(delta - self.w)
                
   
    def predict(self, Xtest):
        """ Calculates probability for each input using sigmoid function """
        """ and if probability is > 0.5 class = 1, else class = 0 """
        predictions = []
        for inputVector in Xtest:
            probability = utils.sigmoid(inputVector.dot(self.w.T))
            if probability > 0.5:
                predictions.append(1)
            else:
                predictions.append(0)   
        return predictions
                        
            
    
