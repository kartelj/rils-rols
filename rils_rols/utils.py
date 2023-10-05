from math import nan, sqrt
from statistics import mean
from numpy.random import RandomState
import numpy as np
from sklearn.metrics import r2_score

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

def RMSE(yt, yp):
    return np.sqrt(np.mean((yt-yp)**2))
    
def ResidualVariance(yt, yp, complexity):
    return np.mean((yt-yp)**2)/(len(yp)-complexity)

def percentile_abs_error(yt, yp, percentile):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        errors = []
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            if err<0:
                err*=-1
            if err == nan:
                return nan
            errors.append(err)
        errors.sort()
        idx = int(percentile*(len(yp)-1)/100)
        return errors[idx]
    except OverflowError:
        return nan

def R2(yt, yp):
    return r2_score(yt, yp)
