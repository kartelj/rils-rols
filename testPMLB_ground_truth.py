from tabnanny import check
from pmlb import fetch_data
from sklearn.model_selection import train_test_split
from rils_rols.rils_rols import RILSRegressor
from rils_rols.utils import noisefy, R2, RMSE
from os import listdir, stat
import sys
from sklearn.utils.estimator_checks import check_estimator
import numpy as np

#check_estimator(RILSRegressor())

if len(sys.argv)!=8:
    print("Usage: <working part or -1> <parts >=1> <random seed> <max time (s)> <max fit calls> <size penalty> <noise level>")
    print("Passed parameters were:")
    print(sys.argv[1:])
    sys.exit(1)

part = int(sys.argv[1])
parts = int(sys.argv[2])

seed = int(sys.argv[3])
max_seconds = int(sys.argv[4])
max_fit_calls = int(sys.argv[5])
complexity_penalty = float(sys.argv[6])
noise_level = float(sys.argv[7])
test_size = 0.25
label="target"

pmlb_cache = "../../pmlb/datasets"
excluded_feynman = ["feynman_I_26_2", "feynman_I_30_5", "feynman_test_10"] # this are using arcsin or arcos
ground_truth_regr_datasets = list(filter(lambda x: (x.startswith("feynman") or x.startswith("strogatz")) and not x in excluded_feynman, listdir(pmlb_cache)))

tmp_dict = {}
for dataset in ground_truth_regr_datasets:
    size = stat(pmlb_cache+"/"+dataset+"/"+dataset+".tsv.gz").st_size
    tmp_dict[dataset]=size

ground_truth_regr_datasets.clear()
for dataset in sorted(tmp_dict.items(), key=lambda x: x[1]):
    ground_truth_regr_datasets.append(dataset[0])

#ground_truth_regr_datasets = ["strogatz_vdp2", "feynman_I_12_5"]
ground_truth_regr_datasets = ['strogatz_vdp2','feynman_I_12_5','feynman_I_39_1','feynman_I_12_1','feynman_III_12_43','feynman_I_34_27','feynman_I_25_13','feynman_I_29_4','feynman_II_27_18','feynman_I_14_4','feynman_II_8_31','feynman_II_3_24','feynman_I_43_31','feynman_I_14_3','feynman_II_34_2','feynman_II_15_4','feynman_II_38_14','feynman_II_15_5','feynman_III_7_38','feynman_II_34_2a','feynman_II_34_29a','feynman_I_18_12','strogatz_glider1','feynman_II_37_1','feynman_III_15_27','feynman_II_4_23','feynman_I_39_22','feynman_II_34_11','feynman_I_39_11','feynman_III_21_20','feynman_I_34_8','feynman_I_43_16','feynman_I_47_23','feynman_II_38_3','feynman_I_12_4','feynman_II_8_7','feynman_III_17_37','feynman_III_15_14','feynman_II_10_9','feynman_II_34_29b','feynman_I_12_2','feynman_I_34_1','feynman_I_43_43','strogatz_lv2','feynman_III_13_18','strogatz_barmag2','feynman_I_44_4','feynman_II_6_11','feynman_II_27_16','feynman_I_11_19','strogatz_barmag1','feynman_II_11_20','strogatz_glider2','feynman_I_24_6','feynman_II_11_3','feynman_II_2_42','feynman_I_13_12','feynman_I_37_4','feynman_test_17','feynman_III_19_51','feynman_I_13_4','strogatz_lv1','feynman_test_18','feynman_I_32_5','feynman_II_36_38','feynman_test_19','feynman_I_50_26','feynman_I_38_12','feynman_I_27_6','feynman_III_15_12','feynman_II_13_17','strogatz_vdp1','feynman_I_12_11','feynman_I_18_14','strogatz_shearflow2','feynman_I_48_2','feynman_test_2','strogatz_bacres1','strogatz_bacres2','feynman_test_7','strogatz_shearflow1','feynman_III_9_52','feynman_test_20','feynman_I_6_2','feynman_II_6_15b','feynman_test_3','strogatz_predprey2','feynman_I_40_1','strogatz_predprey1','feynman_I_15_3x','feynman_test_1','feynman_I_16_6','feynman_III_8_54','feynman_I_9_18','feynman_III_10_19','feynman_test_13','feynman_I_30_3','feynman_test_11','feynman_test_16','feynman_I_32_17','feynman_II_13_34','feynman_test_4','feynman_test_9','feynman_I_34_14','feynman_I_41_16','feynman_II_24_17','feynman_test_8','feynman_I_6_2b','feynman_test_5','feynman_II_35_21','feynman_II_6_15a','feynman_test_12','feynman_test_14','feynman_test_6','feynman_I_29_16','feynman_II_35_18','feynman_III_14_14','feynman_III_4_32','feynman_I_15_10','feynman_II_21_32','feynman_II_11_27','feynman_I_10_7','feynman_I_15_3t','feynman_I_8_14','feynman_II_11_28','feynman_II_13_23','feynman_III_4_33','feynman_test_15','feynman_I_6_2a','feynman_I_18_4',]


out_path = "out_sp"+str(complexity_penalty)+"_nl"+str(noise_level)+".txt"
for i in range(len(ground_truth_regr_datasets)):
    if i%parts!=part:
        continue
    dataset = ground_truth_regr_datasets[i]
    #true_model = get_sym_model(dataset)
    print("Doing "+dataset)
   
    input_data = fetch_data(dataset,  local_cache_dir=pmlb_cache)
        
    feature_names = [x for x in input_data.columns.values if x != label]
    feature_names = np.array(feature_names)

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    y_train = noisefy(y_train, noise_level, seed)
    
    rils = RILSRegressor(max_fit_calls,max_seconds, random_state = seed, complexity_penalty=complexity_penalty)
    rils.fit(X_train, y_train)
    report_string = rils.fit_report_string(X_train, y_train)
    rils_R2 = ""
    rils_RMSE = ""
    try:
        yp = rils.predict(X_test)
        rils_R2 = round(R2(y_test, yp),7)
        rils_RMSE = round(RMSE(y_test, yp),7)
        print("%s\tR2=%.7f\tRMSE=%.7f\texpr=%s"%(dataset, rils_R2, rils_RMSE, rils.model))
    except:
        print("ERROR during test.")
    with open(out_path, "a") as f:
        f.write(dataset+"\t"+report_string+"\tTestR2="+str(rils_R2)+"\tTestRMSE="+str(rils_RMSE)+"\n")
