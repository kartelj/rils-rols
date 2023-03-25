from math import nan, sqrt
from statistics import mean
from numpy.random import RandomState

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
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        rmse = 0.0
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            err*=err
            if err == nan:
                return nan
            rmse+=err
        rmse = sqrt(rmse/len(yp))
        return rmse
    except OverflowError:
       return nan
    
def ResidualVariance(yt, yp, complexity):
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        var = 0.0
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            err*=err
            if err == nan:
                return nan
            var+=err
        var = var/(len(yp)-complexity)
        return var
    except OverflowError:
       return nan

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
    if len(yp)!=len(yt):
        raise Exception("Vectors of predicted and true y values should be of same size.")
    try:
        yt_mean = mean(yt)
        ss_res = 0
        ss_tot = 0
        for i in range(len(yp)):
            err = yp[i]-yt[i]
            err*=err
            if err == nan:
                return nan
            ss_res+=err
            var_err = yt_mean-yt[i]
            var_err*=var_err
            ss_tot+=var_err
        if ss_tot<0.000000000001:
            ss_tot=1
        return 1-ss_res/ss_tot
    except OverflowError:
        return nan