import math
import time
from sklearn.base import BaseEstimator
from sympy import *
from .rils_rols import RILSROLSBinaryClassifier, RILSROLSRegressor
#from joblib import Parallel, delayed
from .utils import complexity_sympy
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

class RILSROLSEnsembleBase(BaseEstimator):

    def __init__(self, classification=None, validation_size=0.5, max_fit_calls_per_estimator=100000, max_seconds_per_estimator=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, estimator_cnt = 8, verbose=False, random_state=0):
         # true random numbers from random.org
        random_states = [5953,7138,7578,5452,1924,823,8269,5439,4188,5507,280,9610,2715,8212,4916,2026,4283,5664,9406,8120,835,6746,2970,4427,803,8092,1466,8895,9428,4106,918,2243,2575,1962,3123,4121,7806,3623,3130,2920,5796,7565,3826,758,5604,6510,8781,1040,8386,5460,6953,8030,9809,5274,6661,4712,3605,5195,9652,7768,2956,268,6707,8171,3462,8539,9377,5956,8137,9457,4624,2125,6821,4522,6634,8946,9558,9812,7225,4552,735,3539,3765,3817,5383,7864,8020,5376,1555,3367,6145,7044,8771,7486,3302,837,7596,8799,2116,4444]
        if estimator_cnt>len(random_states):
            raise Exception(f'Maximal number of estimators is {len(random_states)}')
        self.classification = classification
        self.validation_size = validation_size
        self.max_seconds = max_seconds_per_estimator
        self.max_fit_calls = max_fit_calls_per_estimator
        self.complexity_penalty = complexity_penalty
        self.max_complexity = max_complexity
        self.random_state = random_state
        self.estimator_cnt = estimator_cnt
        self.verbose = verbose
        self.sample_size = sample_size
        self.random_state = random_state
        self.base_estimators = []
        for i in range(estimator_cnt):
            if classification is True:
                est = RILSROLSBinaryClassifier(max_fit_calls=max_fit_calls_per_estimator, max_seconds=max_seconds_per_estimator,
                                                  complexity_penalty=complexity_penalty, sample_size=sample_size,verbose=verbose, random_state=random_states[i])
            elif classification is False:
                est = RILSROLSRegressor(max_fit_calls=max_fit_calls_per_estimator, max_seconds=max_seconds_per_estimator,
                                                  complexity_penalty=complexity_penalty, sample_size=sample_size,verbose=verbose, random_state=random_states[i])
            else:
                raise Exception('Estimator must be either classifier or regressor, there is no third option at the moment.')
            self.base_estimators.append(est)

    def fit(self, X, y):
        # currently, this works by selecting the best of base estimators w.r.t. performance on the validation data part
        # TODO: other options are averaging on the regression or voting on the classification
        self.start = time.time()
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.validation_size,random_state=self.random_state)
        # now run each base regressor as a separate process 
        # TODO: this is not working because of serialization of cpp object rr_cpp probably...
        #Parallel(n_jobs=len(self.base_estimators))(delayed(reg.fit)(X_train, y_train) for reg in self.base_estimators)
        # doing only sequentially for now
        for est in self.base_estimators:
            est.fit(X_train, y_train)
        print("All base estimators have finished now")
        # checking performances on the validation set
        best_score = -math.inf
        self.best_est = None
        for est  in self.base_estimators:
            score = est.score(X_valid, y_valid)
            print(f'Score: {score} model: {est.model_string()}')
            if score>best_score:
                best_score = score
                self.best_est = est
        self.time_elapsed = time.time()-self.start
        self.model = self.best_est.model
        self.model_simp = self.best_est.model_simp
        print(f'Best simplified model is {self.model_simp} with score {best_score}')

    def check_model(self):
        if self.model is None or self.best_est is None:
            raise Exception("Cannot predict because model is not build yet. First call fit().")

    def predict(self, X):
        self.check_model()
        return self.best_est.predict(X)
    
    def score(self, X, y):
        self.check_model()
        return self.best_est.score(X, y)

    def model_string(self):
        self.check_model()
        return self.model_simp

    def fit_report_string(self):
        self.check_model()
        return "maxTime={0}\tmaxFitCalls={1}\tseed={2}\tsizePenalty={3}\tmaxComplexity={4}\tsampleShare={5}\ttotalTime={6:.1f}\tsimpSize={7}\texpr={8}\texprSimp={9}".format(
            self.max_seconds,self.max_fit_calls,self.random_state,self.complexity_penalty,self.max_complexity, self.sample_size, self.time_elapsed, complexity_sympy(self.model_simp),  self.model, self.model_simp)

class RILSROLSEnsembleBinaryClassifier(RILSROLSEnsembleBase):
    
    def __init__(self, max_fit_calls_per_estimator=100000, max_seconds_per_estimator=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1,estimator_cnt = 8, validation_size=0.5, verbose=False, random_state=0):
        super().__init__(True, validation_size=validation_size, max_fit_calls_per_estimator=max_fit_calls_per_estimator, max_seconds_per_estimator=max_seconds_per_estimator, complexity_penalty=complexity_penalty, max_complexity=max_complexity, sample_size=sample_size, estimator_cnt=estimator_cnt, verbose=verbose,  random_state=random_state)

    def predict_proba(self, X):
        self.check_model()
        return self.best_est.predict_proba(X)

class RILSROLSEnsembleRegressor(RILSROLSEnsembleBase):
    
    def __init__(self, max_fit_calls_per_estimator=100000, max_seconds_per_estimator=100, complexity_penalty=0.001, max_complexity=200, sample_size=0.1, estimator_cnt = 8, validation_size=0.5, verbose=False, random_state=0):
        super().__init__(False, validation_size=validation_size,  max_fit_calls_per_estimator=max_fit_calls_per_estimator, max_seconds_per_estimator=max_seconds_per_estimator, complexity_penalty=complexity_penalty, max_complexity=max_complexity, sample_size=sample_size, estimator_cnt=estimator_cnt, verbose=verbose,  random_state=random_state)
