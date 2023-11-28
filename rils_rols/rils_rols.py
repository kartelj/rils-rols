import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sympy import sympify, simplify
import rils_rols_cpp
from .utils import binarize, proba, complexity_sympy
import warnings
import multiprocessing.pool
import functools
warnings.filterwarnings("ignore")

class RILSROLSBase(BaseEstimator):

    def __init__(self, classification=None, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, verbose=False, random_state=0):
        self.classification = classification
        self.max_seconds = max_seconds
        self.max_fit_calls = max_fit_calls
        self.max_complexity = max_complexity
        self.complexity_penalty = complexity_penalty
        self.sample_size = sample_size
        self.verbose = verbose
        self.random_state = random_state
        self.rr_cpp = None
        self.model = None
        self.model_simp = None
    
    def timeout(max_timeout):
        """Timeout decorator, parameter in seconds."""
        def timeout_decorator(item):
            """Wrap the original function."""
            @functools.wraps(item)
            def func_wrapper(*args, **kwargs):
                """Closure for function."""
                pool = multiprocessing.pool.ThreadPool(processes=1)
                async_result = pool.apply_async(item, args, kwargs)
                # raises a TimeoutError if execution exceeds max_timeout
                return async_result.get(max_timeout)
            return func_wrapper
        return timeout_decorator

    @timeout(10.0)
    def call_model_symplify(self):
        self.model_simp = simplify(self.model, ratio=1)

    def fit(self, X, y):
        self.rr_cpp = rils_rols_cpp.rils_rols(self.classification,int(self.max_fit_calls),int(self.max_seconds),self.complexity_penalty,self.max_complexity,self.sample_size,self.verbose,int(self.random_state))
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
        print('Finished otimization, doing final symplification...')
        try:
            self.call_model_symplify()
        except:
            # otherwise, just sympify it -- this does not stuck
            print("Simplification failed withint given timeout, so just doing sympify.")
            self.model_simp = sympify(self.model)
        return (self.model, self.model_simp)
    
    def check_model(self):
        if self.model is None or self.rr_cpp is None:
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

    def fit_report_string(self):
        self.check_model()
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tmaxComplexity={4}\tsampleShare={5}\ttotalTime={6:.1f}\tbestTime={7}\tfitCalls={8}\tsimpSize={9}\texpr={10}\texprSimp={11}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty,self.max_complexity, self.sample_size, self.total_time,self.best_time, self.fit_calls, complexity_sympy(self.model_simp),  self.model, self.model_simp)

class RILSROLSRegressor(RILSROLSBase):
        
    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, verbose=False, random_state=0):
        super().__init__(False, max_fit_calls, max_seconds, complexity_penalty, max_complexity, sample_size, verbose,  random_state)

    def score(self, X, y):
        yp = self.predict(X)
        return r2_score(y, yp)

class RILSROLSClassifier(RILSROLSBase):
    
    def __init__(self, max_fit_calls=100000, max_seconds=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, verbose=False, random_state=0):
        super().__init__(True, max_fit_calls, max_seconds, complexity_penalty, max_complexity, sample_size, verbose,  random_state)

    def predict(self, X: np.ndarray):
        preds = super().predict(X)
        return binarize(preds)
    
    def predict_proba(self, X: np.ndarray):
        preds = self.predict(X)
        return proba(preds)
    
    def score(self, X, y):
        yp = self.predict(X)
        return accuracy_score(y, yp)
    