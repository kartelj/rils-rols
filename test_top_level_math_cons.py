from random import Random
from statistics import mean
from rils_rols.rils_rols import MonotonicityType, RILSROLSRegressor
from math import cos, log, sqrt, sin, exp
from scipy.stats import pearsonr
from rils_rols.utils import R2, RMSE, noisefy
import inspect
import sys

if len(sys.argv)<2:
    print("You must select scenario: 1, 2 or 3.")
    sys.exit(1)

scenario = int(sys.argv[1])

max_seconds = 1000
max_fit_calls = 100000
rseed = 42
train_perc = 0.75

rg = Random(rseed)

f0 = ("f0", lambda x0, x1, x2: 1000*x0+sqrt(200+x1+x2))
f1 = ("f1", lambda x0, x1, x2: cos((exp(x0)+exp(2))*exp(x2*cos(x1))))
f2 = ("f2", lambda x0, x1, x2: pow(sin(x0)*cos(x1)*sin(x2),3)*sin(x0*x1*x2))
f3 = ("f3", lambda x0, x1, x2: pow(cos(x0+x1)*sin(x1*x2), 4))

if scenario==1:
    distribution_fits_penalties = [0]
    monotonicities = [False, True]
    lipschitz_continuities = [(0, 0)]
    X = [[rg.random(), rg.random(), rg.random()] for _ in range(200)]
    formulas = [f0]
    # test if it is really monotone
    #corr, _ = pearsonr([x[0] for x in X], y)
    #assert(corr>=0.999999)
    noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
elif scenario==2:
    distribution_fits_penalties = [0]
    monotonicities = [False]
    lipschitz_continuities = [(0, 0), (0.01, 1), (0.01, 10), (0.01, 100)]
    X = []
    for _ in range(100):
        c = [rg.random(), rg.random(), rg.random()]
        c_neighbor = [c[0]+rg.random()*0.02-0.01]+c[1:]
        X.append(c)
        X.append(c_neighbor)
    formulas = [f1, f2, f3]
    noise_levels = [0]
elif scenario==3:
    distribution_fits_penalties = [1, 10] 
    monotonicities = [False, True]
    lipschitz_continuities = [(0, 0)]
    X = []
    X = [[rg.random(), rg.random(), rg.random()] for _ in range(200)]
    formulas = [f0] 
    noise_levels = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] 
else:
    print("Scenario "+str(scenario)+"does not exist.")
    sys.exit(1)

train_cnt = round(len(X)*train_perc)

for name, formula in formulas:
    y = [formula(x0, x1, x2) for x0, x1, x2 in X]    
    X_train = X[:train_cnt]
    y_train = y[:train_cnt]
    X_test = X[train_cnt:]
    y_test = y[train_cnt:]
    for noise_level in noise_levels:
        y_train_noisy = noisefy(y_train, noise_level, rseed)
        #corr, _ = pearsonr([x[0] for x in X], y_train)
        for mono_use in monotonicities:
            for lipschitz_conf in lipschitz_continuities:
                for distribution_fit_penalty in distribution_fits_penalties:
                    regressor = RILSROLSRegressor(verbose=True, max_seconds=max_seconds, max_fit_calls=max_fit_calls, initial_sample_size=1, monotonicity=(0, MonotonicityType.INCREASING, mono_use), lipschitz_continuity=lipschitz_conf, distribution_fit_penalty=distribution_fit_penalty)
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
                        f.write("scenario={0}\tformula={1}\tnoise_level={2}\tmono_use={3}\tlipschitz_eps={4}\tlipshitz_pen={5}\tdistribution_fit_pen={6}\tTestR2={7:.8f}\tTestRMSE={8:.8f}\t{9}\n".format(scenario, name, noise_level, mono_use, lipschitz_conf[0], lipschitz_conf[1], distribution_fit_penalty, r2, rmse, output_string))