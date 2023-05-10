import copy
from random import Random
from statistics import mean
from rils_rols.rils_rols import MonotonicityType, RILSROLSRegressor
from math import cos, log, sqrt, sin, exp
from scipy.stats import pearsonr
from rils_rols.utils import R2, RMSE, noisefy

scenario = 2 # CHANGE THIS FOR SCENARIO
max_seconds = 1000
max_fit_calls = 100000
rseed = 42
train_perc = 0.75

rg = Random(rseed)

if scenario==1:
    monotonicities = [False, True]
    lipschitz_epss = [0]
    X = [[rg.random(), rg.random(), rg.random()] for _ in range(200)]
    name = "1000*x0+sqrt(200+x1+x2)"
    y = [1000*x0+sqrt(200+x1+x2) for x0, x1, x2 in X]    
    # test if it is really monotone
    corr, _ = pearsonr([x[0] for x in X], y)
    assert(corr>=0.999999)
    noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
elif scenario==2:
    monotonicities = [False]
    lipschitz_epss = [0, 0.1, 0.01]
    X = []
    for _ in range(100):
        c = [rg.random(), rg.random(), rg.random()]
        c_neighbor = [c[0]+rg.random()*0.02-0.01]+c[1:]
        X.append(c)
        X.append(c_neighbor)
    name = "pow(sin(x0)*cos(x1)*sin(x2),3)*sin(x0*x1*x2)"
    y = [pow(sin(x0)*cos(x1)*sin(x2),3)*sin(x0*x1*x2) for x0, x1, x2 in X]
    noise_levels = [0.01]

train_cnt = round(len(X)*train_perc)
X_train = X[:train_cnt]
y_train = y[:train_cnt]
X_test = X[train_cnt:]
y_test = y[train_cnt:]

for noise_level in noise_levels:
    y_train_noisy = noisefy(y_train, noise_level, rseed)
    #corr, _ = pearsonr([x[0] for x in X], y_train)
    for mono_use in monotonicities:
        for lipschitz_continuity_eps in lipschitz_epss:
            regressor = RILSROLSRegressor(verbose=True, max_seconds=max_seconds, max_fit_calls=max_fit_calls, initial_sample_size=1, monotonicity=(0, MonotonicityType.INCREASING, mono_use), lipschitz_continuity_eps=lipschitz_continuity_eps)
            regressor.fit(X_train, y_train_noisy)
            # this prints out the learned simplified model
            print("Final model is:\t"+str(regressor.model_simp))
            # this prints some additional information as well
            output_string = regressor.fit_report_string(X_train, y_train_noisy)
            print("Detailed output:\t"+output_string)
            r2 = -1
            rmse = -1
            try:
                yp = regressor.predict(X_test)
                r2 = R2(y_test, yp)
                rmse = RMSE(y_test, yp)
                print("R2=%.8f\tRMSE=%.8f\texpr=%s"%(r2, rmse, regressor.model_simp))
            except Exception as ex:
                print("ERROR during test "+str(ex))
            with open("results.txt", "a") as f:
                f.write("{0}\tnoise_level={1}\tmono_use={2}\tlipshitz_eps={3}\tTestR2={4:.8f}\tTestRMSE={5:.8f}\t{6}\n".format(name, noise_level, mono_use, lipschitz_continuity_eps, r2, rmse, output_string))