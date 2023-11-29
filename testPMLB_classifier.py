from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from rils_rols.rils_rols import RILSROLSClassifier
from rils_rols.rils_rols_ensemble import RILSROLSEnsembleClassifier
import sys
#from sklearn.utils.estimator_checks import check_estimator
import numpy as np
import pandas as pd
from pmlb import fetch_data, classification_dataset_names
import time
import cProfile

#check_estimator(RILSROLSRegressor())

'''
if len(sys.argv)!=4:
    print("Usage: <random seed> <max fit calls> <threads>")
    print("Passed parameters were:")
    print(sys.argv[1:])
    sys.exit(1)

RANDOM_STATE = int(sys.argv[1])
ITER_LIMIT = int(sys.argv[2])
THREADS = int(sys.argv[3])
'''
RANDOM_STATE = 23654
ITER_LIMIT = 100000
SAMPLE_SIZE = 1
TIME_LIMIT = 100
CLASSIFIERS_CNT = 10
COMPLEXITY_PENALTY = 0.001

DATASET_MIN_SIZE = 1000
MAX_FEATURES = 10000
datasets={}
for name in classification_dataset_names:
    df = fetch_data(dataset_name=name, local_cache_dir="../pmlb/datasets")
    if len(df) < DATASET_MIN_SIZE or len(df.columns)>MAX_FEATURES:
        continue
    df.to_csv(f"class_datasets/{name}.csv")
    # binary classification problem
    if df['target'].nunique() == 2:
        # some datasetts have classes [1 2] or [0 2]
        if df['target'].max() > 1:
            df['target'] = df['target'].apply(lambda x: 1 if x == 2 else 0)
        datasets[name] = df
        print(name, df.shape, df['target'].unique(), df.isnull().values.any(), df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all())

results = pd.DataFrame(columns=['dataset', 'samples', 'features', 'regressor','time','acc_score', 'll_score', 'model'])
classificators = [
    #['AdaBoostClassifier', AdaBoostClassifier, {'random_state':RANDOM_STATE}],
    #['LogisticRegression', LogisticRegression, {'random_state':RANDOM_STATE}],
    #['DecisionTreeClassifier', DecisionTreeClassifier, {'random_state':RANDOM_STATE}],
    #['RandomForestClassifier', RandomForestClassifier, {'random_state':RANDOM_STATE}],
    ['RILSROLSClassifier', RILSROLSClassifier, {'sample_size':SAMPLE_SIZE, 'complexity_penalty':COMPLEXITY_PENALTY, 'random_state':RANDOM_STATE, 'max_fit_calls':ITER_LIMIT, 'max_seconds':TIME_LIMIT, 'verbose':True}],
    #['RILSROLSEnsembleClassifier', RILSROLSEnsembleClassifier, {'sample_size':SAMPLE_SIZE/10,'max_fit_calls_per_estimator':ITER_LIMIT, 'max_seconds_per_estimator':TIME_LIMIT/CLASSIFIERS_CNT,'estimator_cnt': CLASSIFIERS_CNT,  'complexity_penalty':COMPLEXITY_PENALTY, 'random_state':RANDOM_STATE, 'verbose':True}],
  ]

for name, df in datasets.items():
    _X = df.drop('target', axis=1)
    X = _X.to_numpy()
    y = df[['target']].to_numpy().ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=RANDOM_STATE)
    
    print(f'{name} size {_X.shape}')
    
    for clf_name, clf_type, clf_params in classificators:
        clf = clf_type(**clf_params)
        start = time.time()
        #cProfile.run('clf.fit(X_train, y_train)', 'restats')
        clf.fit(X_train, y_train)
        fit_time = time.time() - start
        try:
            eq = clf.model_string()
        except:
            eq = None
        preds = np.nan_to_num(clf.predict(X_test))
        preds_train = np.nan_to_num(clf.predict(X_train))
        proba = np.nan_to_num(clf.predict_proba(X_test))
        acc_train = accuracy_score(y_train, preds_train)
        acc = accuracy_score(y_test, preds)
        ll = log_loss(y_test, proba[:,1])
        log_txt = f'{name} {clf_name} {acc_train} {acc} {ll} {fit_time}s {eq}'
        print(log_txt)
        with open('log.txt', 'a') as flog:
            flog.write(log_txt+'\n')
        results.loc[len(results)] = [name, _X.shape[0], _X.shape[1], clf_name, fit_time, acc, ll, eq]   

results.to_csv(f'results_cp_{COMPLEXITY_PENALTY}_sz_{SAMPLE_SIZE}_il_{ITER_LIMIT}.csv')

classifiers = results['regressor'].unique()
print('MEAN')
for clf in classifiers:
    time = results[results['regressor']==clf]['time'].mean()
    acc = results[results['regressor']==clf]['acc_score'].mean()
    log_loss = results[results['regressor']==clf]['ll_score'].mean()
    print(f'{clf} time:{time} acc:{acc} log_loss:{log_loss}')
    
print('MEDIAN')
for clf in classifiers:
    time = results[results['regressor']==clf]['time'].median()
    acc = results[results['regressor']==clf]['acc_score'].median()
    log_loss = results[results['regressor']==clf]['ll_score'].median()
    print(f'{clf} time:{time} acc:{acc} log_loss:{log_loss}')

