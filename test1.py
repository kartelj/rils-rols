from math import sqrt
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from rils_rols.rils_rols import RILSROLSRegressor

PROFILE = False

dataset = 'feynman_I_15_3x'
pmlb_cache = "../pmlb/datasets"
out_path = "tmp.txt"
label="target"
test_size = 0.25
seed = 12345
max_fit_calls = 10000
max_seconds = 10000

input_data = fetch_data(dataset,  local_cache_dir=pmlb_cache)

feature_names = [x for x in input_data.columns.values if x != label]
feature_names = np.array(feature_names)

X = input_data.drop(label, axis=1).values.astype(float)
y = input_data[label].values

assert(X.shape[1] == feature_names.shape[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

rils = RILSROLSRegressor(max_fit_calls=max_fit_calls, max_seconds=max_seconds, random_state = seed, verbose=True)
rils.fit(X_train, y_train)
report_string = rils.fit_report_string()
rils_R2 = ""
rils_RMSE = ""
try:
    yp = rils.predict(X_test)
    rils_R2 = r2_score(y_test, yp)
    rils_RMSE = sqrt(mean_squared_error(y_test, yp))
    print("%s\tR2=%.8f\tRMSE=%.8f\texpr=%s"%(dataset, rils_R2, rils_RMSE, rils.model_simp))
except:
    print("ERROR during test.")
with open(out_path, "a") as f:
    f.write("{0}\t{1}\tTestR2={2:.8f}\tTestRMSE={3:.8f}\n".format(dataset, report_string, rils_R2, rils_RMSE))