import math
from random import Random
import time
from sklearn.base import BaseEstimator
from sympy import *
from .rils_rols import RILSROLSRegressor
from joblib import Parallel, delayed

import warnings

warnings.filterwarnings("ignore")

class RILSROLSEnsembleRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, initial_sample_size=0.01, parallelism = 8, verbose=False, random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.parallelism = parallelism
        self.verbose = verbose
        self.initial_sample_size = initial_sample_size
        rg = Random(random_state)
        random_states = [rg.randint(10000, 99999) for i in range(self.parallelism)]
        self.base_regressors = [RILSROLSRegressor(max_fit_calls=max_fit_calls, max_seconds=max_seconds,
                                                  complexity_penalty=complexity_penalty, sample_size=initial_sample_size,verbose=verbose, random_state=random_states[i]) 
                                                  for i in range(len(random_states))]

# TODO: fix ensemble
'''
    def fit(self, X, y):
        self.start = time.time()
        # now run each base regressor (RILSROLSRegressor) as a separate process
        results = Parallel(n_jobs=len(self.base_regressors))(delayed(reg.fit)(X, y) for reg in self.base_regressors)
        print("All regressors have finished now")
        best_model, best_model_simp = results[0]
        best_fit = best_model.fitness(X, y, False)
        for model, model_simp in results:
            model_fit = model.fitness(X,y, False)
            if self.base_regressors[0].compare_fitness(model_fit, best_fit)<0:
                best_fit = model_fit
                best_model = model
                best_model_simp = model_simp
            print('Model '+str(model)+'\t'+str(model_fit))
        self.time_elapsed = time.time()-self.start
        self.model = best_model
        self.model_simp = best_model_simp
        print('Best simplified model is '+str(self.model_simp) + ' with '+str(best_fit))

    def predict(self, X):
        Node.reset_node_value_cache()
        return self.model.evaluate_all(X, False)

    def size(self):
        if self.model is not None:
            return self.model.size()
        return math.inf

    def modelString(self):
        if self.model_simp is not None:
            return str(self.model_simp)
        return ""

    def fit_report_string(self, X, y):
        if self.model==None:
            raise Exception("Model is not build yet. First call fit().")
        fitness = self.model.fitness(X,y, False)
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\texpr={8}\texprSimp={9}\fitType={10}\tinitSampleSize={11}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, 1-fitness[0], fitness[1], self.complexity(), self.time_elapsed, self.model, self.model_simp, self.fitness_type, self.initial_sample_size)

    def complexity(self):
        c=0
        for arg in preorder_traversal(self.model_simp):
            c += 1
        return c
'''