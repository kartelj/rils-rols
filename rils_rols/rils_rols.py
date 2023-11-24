from cmath import inf
from random import Random, shuffle
import numpy as np
from sklearn.base import BaseEstimator
import copy
from sympy import *
import rils_rols_cpp
from .utils import binarize, proba
import warnings

warnings.filterwarnings("ignore")

class RILSROLSRegressor(BaseEstimator):

    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, verbose=False, random_state=0):
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.max_complexity = max_complexity
        self.complexity_penalty = complexity_penalty
        self.sample_size = sample_size
        self.verbose = verbose
        self.random_state = random_state
    
    def fit(self, X, y):
        self.rr_cpp = rils_rols_cpp.rils_rols(self.max_fit_calls,self.max_seconds,self.complexity_penalty,self.max_complexity,self.sample_size,self.verbose,self.random_state)
        X = np.array(X)
        data_cnt = X.shape[0]
        feat_cnt = X.shape[1]
        X.resize((data_cnt*feat_cnt, 1))
        y = np.array(y)
        # now call CPP fit method to finish the job
        self.rr_cpp.fit(X, y, data_cnt, feat_cnt)
        self.model = self.rr_cpp.get_model_string()
        self.best_time = self.rr_cpp.get_best_time()
        self.total_time = self.rr_cpp.get_total_time()
        self.fit_calls = self.rr_cpp.get_fit_calls()
        self.model_simp = simplify(self.model, ratio=1)
        return (self.model, self.model_simp)
    
    def check_model(self):
        if self.rr_cpp is None or self.model is None:
            raise Exception("Cannot predict because model is not build yet. First call fit().")
        
    def predict(self, X):
        self.check_model()
        X = np.array(X)
        data_cnt = X.shape[0]
        feat_cnt = X.shape[1]
        X.resize((data_cnt*feat_cnt, 1))
        y = self.rr_cpp.predict(X, data_cnt, feat_cnt)
        return y

    def model_string(self):
        self.check_model()
        return self.model_simp

    def fit_report_string(self, X, y):
        self.check_model()
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tmaxComplexity={4}\tsampleShare={5}\ttotalTime={6:.1f}\tbestTime={7}\tfitCalls={8}\tsimpSize={9}\texpr={10}\texprSimp={11}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty,self.max_complexity, self.sample_size, self.total_time,self.best_time, self.fit_calls, self.complexity(self.model_simp),  self.model, self.model_simp)

    def complexity(self, sm):
        c=0
        for arg in preorder_traversal(sm):
            c += 1
        return c

class RILSROLSClassifier(RILSROLSRegressor):
    
    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, verbose=False, random_state=0):
        super().__init__(max_fit_calls, max_seconds, complexity_penalty, max_complexity, sample_size, verbose,  random_state)

    def predict(self, X: np.ndarray):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.

        Returns:
            numpy.ndarray: Returns predicted values.
        """
        preds = RILSROLSRegressor.predict(self, X)
        return binarize(preds)
    
    def predict_proba(self, X: np.ndarray):
        """Predict using the symbolic model.

        Args:
            - X (numpy.ndarray): Samples.

        Returns:
            numpy.ndarray, shape = [n_samples, n_classes]: The class probabilities of the input samples.
        """
        preds = RILSROLSRegressor.predict(self, X)
        return proba(preds)