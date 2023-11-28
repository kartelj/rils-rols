from math import nan, sqrt
from statistics import mean
from numpy.random import RandomState
import numpy as np
from sklearn.metrics import r2_score
from sympy import preorder_traversal

def complexity_sympy(model):
    c=0
    for _ in preorder_traversal(model):
        c += 1
    return c

# described in Apendix 4 of paper Contemporary Symbolic Regression Methods and their Relative Performance
def noisefy(y, noise_level, random_state):
    yRMSE = 0
    for i in range(len(y)):
        yRMSE+=(y[i]*y[i])
    yRMSE=sqrt(yRMSE/len(y))
    yRMSE_noise_SD = noise_level*yRMSE
    rg = RandomState(random_state)
    noise = rg.normal(0, yRMSE_noise_SD, len(y))
    y_n = []
    for i in range(len(y)):
        y_n.append(y[i]+noise[i])
    return y_n

def logistic(yp):
    return 1.0/(1.0+np.exp(-yp))

def binarize(yp):
    #yp = logistic(yp)
    return (yp > 0.5)*1

def proba(yp):
    yp = logistic(yp)
    return np.vstack([1 - yp, yp]).T