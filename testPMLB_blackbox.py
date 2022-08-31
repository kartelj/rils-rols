from datetime import datetime
from pmlb import regression_dataset_names
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import utils
from sklearn import svm
from rils import RILS

#print(regression_dataset_names)

pmlb_cache = "../pmlb/datasets"
time = 100
seed = 12345
test_perc = 0.25
outPath = "out_{0}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
size_penalty = None
scale_x = True
scale_y = False

#regression_dataset_names = ["feynman_I_29_16"]
for dataset in regression_dataset_names:
    X, y = fetch_data(dataset, return_X_y=True,  local_cache_dir=pmlb_cache)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_perc, random_state=seed)

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    if scale_y:
        print('scaling y')
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train

    ridge = linear_model.Ridge(alpha=.5)
    ridge.fit(X_train_scaled, y_train)
    ridgeScore = round(ridge.score(X_test_scaled, y_test),3)
    print(dataset+"\tRidgeR2="+str(ridgeScore))

    svr = svm.SVR(C=1.0, epsilon=0.2)
    svr.fit(X_train_scaled, y_train)
    svrScore = round(svr.score(X_test_scaled, y_test),3)
    print(dataset+"\tSvrR2="+str(svrScore))

    ilsr = RILS(100000,time, complexity_penalty=size_penalty, random_state = seed)
    ilsr.fit(X_train_scaled, y_train_scaled)
    reportString = ilsr.fit_report_string(X_train_scaled, y_train_scaled)
    yp = ilsr.predict(X_test_scaled)
    if scale_y:
        yp = sc_y.inverse_transform(yp)
    ilsrR2= round(utils.R2(y_test, yp),7)
    ilsrRMSE = round(utils.RMSE(y_test, yp),7)
    print("%s\tR2=%.3f\tRMSE=%.3f"%(dataset, ilsrR2, ilsrRMSE))
    with open(outPath, "a") as f:
        f.write(dataset+"\t"+reportString+"\tTestR2="+str(ilsrR2)+"\tTestRMSE="+str(ilsrRMSE)+"\tRidgeR2="+str(ridgeScore)+"\tSvrR2="+str(svrScore)+"\n")
