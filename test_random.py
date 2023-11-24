from datetime import datetime
from multiprocessing import freeze_support
from random import Random
from rils_rols.rils_rols import RILSROLSRegressor
from rils_rols import utils
from os import listdir
from os.path import isfile, join

from rils_rols.rils_rols_ensemble import RILSROLSEnsembleRegressor

instances_dir = "paper_resources/random_12345_data" 
random_state = 23654
train_perc = 0.75
time = 1200
max_fit = 1000000
noise_level = 0
complexity_penalty = 0.001 # 0.001 default
max_complexity = 200
parallelism = 1
sample_size = 0.01

instance_files = [f for f in listdir(instances_dir) if isfile(join(instances_dir, f))]

out_path = "out_{0}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
with open(out_path, "w") as f:
    f.write("Tests started\n")

for fpath in  instance_files:
    print("Running instance "+fpath)
    with open(instances_dir+"/"+ fpath) as f:
        lines = f.readlines()
        train_cnt = int(len(lines)*train_perc)
        rg = Random(random_state)
        rg.shuffle(lines)
        X_train = []
        y_train= []    
        X_test = []
        y_test = []
        for i in range(len(lines)):
            line = lines[i]       
            tokens = line.split(sep="\t")
            newX = [float(t) for t in tokens[:len(tokens)-1]]
            newY = float(tokens[len(tokens)-1])
            if i<train_cnt:
                X_train.append(newX)
                y_train.append(newY)
            else:
                X_test.append(newX)
                y_test.append(newY)
        y_train = utils.noisefy(y_train, noise_level, random_state)

    rils = RILSROLSRegressor(max_fit_calls=max_fit, max_seconds=time, random_state = random_state, sample_size=sample_size, max_complexity=max_complexity, complexity_penalty=complexity_penalty, verbose=True)
    
    rils.fit(X_train, y_train)
    report_string = rils.fit_report_string(X_train, y_train)
    rils_R2 = -1
    rils_RMSE = -1
    try:
        yp = rils.predict(X_test)
        rils_R2 = utils.R2(y_test, yp)
        rils_RMSE = utils.RMSE(y_test, yp)
        print("R2=%.8f\tRMSE=%.8f\texpr=%s"%(rils_R2, rils_RMSE, rils.model))
    except:
        print("ERROR during test.")
    with open(out_path, "a") as f:
        f.write("{0}\t{1}\tTestR2={2:.8f}\tTestRMSE={3:.8f}\n".format(fpath, report_string, rils_R2, rils_RMSE))