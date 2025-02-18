from math import inf
import time
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sympy import sympify, simplify
import rils_rols_cpp
from .utils import binarize, proba, complexity_sympy
import warnings
import multiprocessing.pool
import functools
import pandas as pd
warnings.filterwarnings("ignore")

class RILSROLSBase(BaseEstimator):

    def __init__(self, classification=None, max_fit_calls=100000, max_time=100, complexity_penalty=0.001, max_complexity=50, sample_size=1, verbose=False, random_state=0):
        print(f'Calling with max_fit_calls={max_fit_calls} max_time={max_time} complexity_penalty={complexity_penalty} max_complexity={max_complexity} sample_size={sample_size} verbose={verbose} random_state={random_state}')
        self.classification = classification
        self.max_time = max_time
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

    @timeout(2.0)
    def call_model_symplify(self):
        self.model_simp = simplify(self.model, ratio=1)

    def fit(self, X, y):
        if self.sample_size==0:
            print('Automatically tuning sample size:')
            if len(X)<=10000:
                print('Training size is smaller than 10000 so setting it to full sample 1 (100%).')
                self.sample_size = 1
            else:
                start = time.time()
                total_max_fit_calls = self.max_fit_calls
                max_fit_calls_tuning = self.max_fit_calls/100 # in total 2% of the time budget because we check for two alternatives: 1%, 10%
                tuning_fit_calls = 0
                self.max_fit_calls = max_fit_calls_tuning
                best_ss = 1 # this is a default option
                for ss in [0.1]: #[0.01, 0.1]:
                    X_sample,_, y_sample,_ = train_test_split(X, y, train_size=ss, random_state=self.random_state)
                    self.sample_size = 1 # use the whole sample, because we already took the sample with command above
                    self.fit_inner(X_sample, y_sample)
                    tuning_fit_calls+=self.max_fit_calls
                    yp_sample = self.predict(X_sample)
                    yp = self.predict(X) # check performance on the whole training set
                    try:
                        r2_sample = r2_score(y_sample, yp_sample)
                        r2 = r2_score(y, yp)
                        print(f'Sample size {ss} --> R2={r2} R2_sample={r2_sample}')
                        if abs(r2-r2_sample)<0.01:
                            # this means that sample is representative enough on the whole training data
                            best_ss = ss
                            break
                    except Exception:
                        print('Error while calculating R2.')
                        r2 = -inf
                print(f'Setting sample_size={best_ss}')
                self.sample_size = best_ss
                tuning_seconds = time.time() - start
                self.max_time-=tuning_seconds
                self.max_fit_calls = total_max_fit_calls - tuning_fit_calls
        elif self.sample_size<0 or self.sample_size>1:
            raise Exception(f'Sample size parameter must belong to interval (0, 1], while value 0 means it is automatically tuned.')
        self.fit_inner(X, y)


    def fit_inner(self, X, y):
        if isinstance(X, pd.DataFrame):
            if self.verbose:
                print('Converting X dataframe to just list of lists.')
            X = X.values.tolist()
        if isinstance(y, pd.DataFrame):
            if self.verbose:
                print('Converting y dataframe to list of lists.')
            y = y.values.tolist()
        self.rr_cpp = rils_rols_cpp.rils_rols(self.classification,int(self.max_fit_calls),int(self.max_time),self.complexity_penalty,self.max_complexity,self.sample_size,self.verbose,int(self.random_state))
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
        if self.classification is True:
            if self.verbose:
                print('Finished optimization, not doing symplification for the classification scenario.')
            self.model_simp = self.model
        else:
            if self.verbose:
                print('Finished otimization, doing final symplification...')
            try:
                self.call_model_symplify()
            except:
                # otherwise, just sympify it -- this does not stuck
                if self.verbose:
                    print("Simplification failed within given timeout, so just doing sympify.")
                self.model_simp = sympify(self.model)
        return (self.model, self.model_simp)
    
    def check_model(self):
        if self.model is None or self.rr_cpp is None:
            raise Exception("Cannot predict because model is not build yet. First call fit().")
        
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            if self.verbose:
                print('Converting X dataframe to just list of lists.')
            X = X.values.tolist()
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
            self.max_time,self.max_fit_calls,self.random_state,self.complexity_penalty,self.max_complexity, self.sample_size, self.total_time,self.best_time, self.fit_calls, complexity_sympy(self.model_simp),  self.model, self.model_simp)

class RILSROLSRegressor(RILSROLSBase):
        
    def __init__(self, max_fit_calls=100000, max_time=100, complexity_penalty=0.001, max_complexity=50, sample_size=1, verbose=False, random_state=0):
        super().__init__(False, max_fit_calls, max_time, complexity_penalty, max_complexity, sample_size, verbose,  random_state)

    def score(self, X, y):
        yp = self.predict(X)
        return r2_score(y, yp)

class RILSROLSBinaryClassifier(RILSROLSBase):
    
    def __init__(self, max_fit_calls=100000, max_time=100, complexity_penalty=0.001, max_complexity=50, sample_size=1, verbose=False, random_state=0):
        super().__init__(True, max_fit_calls, max_time, complexity_penalty, max_complexity, sample_size, verbose,  random_state)

    def check_binary_targets(self, y):
        for yi in y:
            if yi!=0 and yi!=1:
                raise Exception('The classifier works only for binary targets, so allowed target values are 0 or 1.')

    def predict(self, X):
        preds = super().predict(X)
        return binarize(preds)
    
    def predict_proba(self, X):
        preds = self.predict(X)
        return proba(preds)
    
    def score(self, X, y):
        self.check_binary_targets(y)
        yp = self.predict(X)
        return accuracy_score(y, yp)
    
    def fit(self, X, y):
        self.check_binary_targets(y)
        return super().fit(X, y)
    