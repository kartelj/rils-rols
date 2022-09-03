from datetime import datetime
from random import Random
from rils_rols.rils_rols import RILSRegressor
from rils_rols import utils
from os import listdir
from os.path import isfile, join


instances_dir = "random_12345_data" 
random_state = 12345
train_perc = 0.75
time = 100

instance_files = [f for f in listdir(instances_dir) if isfile(join(instances_dir, f))]

out_path = "out_{0}.txt".format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
with open(out_path, "w") as f:
    f.write("Tests started\n")

for fpath in instance_files:
    #if not "random_10_05_01_0010000_05" in fpath:
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

    vnl = RILSRegressor(max_seconds=time, random_state = random_state)
    vnl.fit(X_train, y_train)
    reportString = vnl.fit_report_string(X_train, y_train)
    yp = vnl.predict(X_test)
    print("%s\tRMSE=%.3f\tR2=%.3f"%(vnl, utils.RMSE(y_test, yp), utils.R2(y_test, yp)))
    with open(out_path, "a") as f:
        f.write(fpath+"\tTestRMSE="+str(round(utils.RMSE(y_test, yp),7))+"\tTestR2="+str(round(utils.R2(y_test, yp),7))+"\t"+reportString+"\n")
