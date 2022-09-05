from datetime import datetime
from random import Random
from rils_rols.rils_rols import RILSRegressor
from rils_rols import utils
from os import listdir
from os.path import isfile, join


instances_dir = "random_12345_data" 
random_state = 12345
train_perc = 0.75
time = 200
max_fit = 1000000

#instance_files = [f for f in listdir(instances_dir) if isfile(join(instances_dir, f))]
instance_files = ["random_04_01_0010000_04.data"]

out_path = "out_{0}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
with open(out_path, "w") as f:
    f.write("Tests started\n")

for fpath in instance_files:
    #if not "random_04_01_0010000_04" in fpath:
    #    continue
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

    vnl = RILSRegressor(max_fit_calls=max_fit, max_seconds=time, random_state = random_state)
    vnl.fit(X_train, y_train)
    reportString = vnl.fit_report_string(X_train, y_train)
    rils_R2 = ""
    rils_RMSE = ""
    try:
        yp = vnl.predict(X_test)
        rils_R2 = round(utils.R2(y_test, yp),7)
        rils_RMSE = round(utils.RMSE(y_test, yp),7)
        print("R2=%.7f\tRMSE=%.7f\texpr=%s"%(rils_R2, rils_RMSE, vnl.model))
    except:
        print("ERROR during test.")
    with open(out_path, "a") as f:
        f.write(fpath+"\tTestRMSE="+str(rils_RMSE)+"\tTestR2="+str(rils_R2)+"\t"+reportString+"\n")
