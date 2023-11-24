from rils_rols.rils_rols import RILSROLSRegressor
from rils_rols.rils_rols_ensemble import RILSROLSEnsembleRegressor
from math import sin, log


''' RILSROLSRegressor parameters:
    1. max_fit_calls=100000             -- maximal number of fitness function calls
    2. max_seconds=100                  -- maximal running time in seconds
    3. complexity_penalty=0.001         -- expression size penalty (used for FitnessType.PENALTY) -- larger value means size is more important
    4. initial_sample_size=0.01         -- the size of the sample taken from the training part (initially)
    5. verbose=False                    -- if True, the output during the program execution contains more details
    6. random_state=0                   -- random seed -- when 0 (default), the algorithm might produce different results in different runs
'''

''' RILSROLSEnsembleRegressor parameters:
    1. max_fit_calls=100000             -- maximal number of fitness function calls
    2. max_seconds=100                  -- maximal running time in seconds
    3. complexity_penalty=0.001         -- expression size penalty (used for FitnessType.PENALTY) -- larger value means size is more important
    4. initial_sample_size=0.01         -- the size of the sample taken from the training part (initially)
    5. parallelism=8                    -- determines the number of RILS-ROLS regressors used in the ensemble
    6. verbose=False                    -- if True, the output during the program execution contains more details
    7. random_state=0                   -- random seed -- when 0 (default), the algorithm might produce different results in different runs
'''

regressors = [RILSROLSRegressor()] #, RILSROLSEnsembleRegressor()]

# toy dataset 
X = [[3, 4], [1, 2], [-10, 20], [10, 10], [100, 100], [22, 23]]
y = [sin(x1)+2.3*log(x2) for x1, x2 in X]

# RILSROLSRegressor and RILSROLSEnsembleRegressor inherit BaseEstimator (sklearn), so we have well-known fit method
for regressor in regressors:
    regressor.fit(X, y)

    # this prints out the learned simplified model
    print("Final model is:\t"+str(regressor.model_simp))

    # this prints some additional information as well
    output_string = regressor.fit_report_string(X, y)
    print("Detailed output:\t"+output_string)

    # applies the model to a list of input vectors, also well-known predict method
    X_test = [[4, 4], [3, 3]]
    y_test = regressor.predict(X_test)
    print(y_test)