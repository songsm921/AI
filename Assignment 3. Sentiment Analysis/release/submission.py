#!/usr/bin/python

import random
import collections
import math
import sys
import nltk
import sympy
from sympy import symbols, diff
from nltk import ngrams
from collections import Counter
from util import *


############################################################
# Problem 1: hinge loss
############################################################


def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        pretty, good, bad, plot, not, scenery
    """
    # BEGIN_YOUR_ANSWER (our solution is 1 lines of code, but don't worry if you deviate from this)
    return {'pretty' : 1, 'good' : -1 , 'bad' : -1, 'plot' : -1, 'not' : 2, 'scenery' : 0}  # remove this line before writing code
    # END_YOUR_ANSWER


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction


def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x:
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 6 lines of code, but don't worry if you deviate from this)
    word_features = dict()
    for i in x.split(' '):
        if i not in word_features:
            word_features[i]=1
        else:
            word_features[i]+=1
    return word_features  # remove this line before writing code
    # END_YOUR_ANSWER


############################################################
# Problem 2b: stochastic gradient descent


def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    """
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    """
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(n))

    # BEGIN_YOUR_ANSWER (our solution is 14 lines of code, but don't worry if you deviate from this)
    def linear_predict(x):
        phi = featureExtractor(x)
        if dotProduct(weights,phi)<0.0:
            return -1
        else:
            return 1
    
    for x,y in trainExamples:
        for feature in featureExtractor(x):
            weights[feature] = 0
    for i in range(numIters):
        for x, y in trainExamples:
            phi = featureExtractor(x)
            if y == 1:
                increment(weights,eta*sigmoid(dotProduct(weights,phi)),phi)
            if y == -1:
                increment(weights,-1*eta*sigmoid(dotProduct(weights,phi))*math.exp(dotProduct(weights,phi)),phi)
                    
                    
                    
            
    # END_YOUR_ANSWER
    return weights


############################################################
# Problem 2c: bigram features


def extractBigramFeatures(x):
    """
    Extract unigram and bigram features for a string x, where bigram feature is a tuple of two consecutive words. In addition, you should consider special words '<s>' and '</s>' which represent the start and the end of sentence respectively. You can exploit extractWordFeatures to extract unigram features.

    For example:
    >>> extractBigramFeatures("I am what I am")
    {('am', 'what'): 1, 'what': 1, ('I', 'am'): 2, 'I': 2, ('what', 'I'): 1, 'am': 2, ('<s>', 'I'): 1, ('am', '</s>'): 1}
    """
    # BEGIN_YOUR_ANSWER (our solution is 5 lines of code, but don't worry if you deviate from this)
    phi = dict()
    msg_split = x.split()
    for i in list(ngrams(msg_split,2)):
        if i not in phi:
            phi[i] =1
        else:
            phi[i]+=1
    for i in x.split(' '):
        if i not in phi:
            phi[i]=1
        else:
            phi[i]+=1
    phi[('<s>',msg_split[0])] = 1
    phi[(msg_split[len(msg_split)-1]),'</s>'] = 1
    
    # END_YOUR_ANSWER
    return phi
