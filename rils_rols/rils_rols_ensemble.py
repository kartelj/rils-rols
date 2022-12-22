import math
from random import Random
from sklearn.base import BaseEstimator
import copy
from sympy import *
from .node import Node
from .rils_rols import RILSROLSRegressor
from joblib import Parallel, delayed

import warnings

from .solution import Solution
warnings.filterwarnings("ignore")

class RILSROLSEnsembleRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, error_tolerance=1e-16, parallelism = 8, random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.complexity_penalty = complexity_penalty
        self.random_state = random_state
        self.error_tolerance = error_tolerance
        self.parallelism = parallelism
        rg = Random(random_state)
        random_states = [rg.randint(10000, 99999) for i in range(self.parallelism)]
        self.base_regressors = [RILSROLSRegressor(max_fit_calls=max_fit_calls, max_seconds=max_seconds, complexity_penalty=complexity_penalty,
            error_tolerance=error_tolerance, random_perturbations_order=True, random_state=rgs) for rgs in random_states]

    def fit(self, X, y):
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
        self.model = best_model
        self.model_simp = best_model_simp
        print('Best model is '+str(self.model) + ' with '+str(best_fit))

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
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tR2={4:.7f}\tRMSE={5:.7f}\tsize={6}\tsec={7:.1f}\tmainIt={8}\tlsIt={9}\tfitCalls={10}\texpr={11}\texprSimp={12}\terrTol={13}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty, 1-fitness[0], fitness[1], self.complexity(), 0, 0, 0,Solution.fit_calls, self.model, self.model_simp, self.error_tolerance)

    def complexity(self):
        c=0
        for arg in preorder_traversal(self.model_simp):
            c += 1
        return c
