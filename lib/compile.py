import pandas as pd
import numpy as np
import os

def extract_segments(dir_name):
    parts = dir_name.split("_")
    bl_part = parts[0] + "_" + parts[1]
    fl_part = parts[2] + "_" + parts[3][:parts[3].find('WSSVQ')]
    return bl_part[3:], fl_part[3:]

def compile_metrics():
    datanames = os.listdir()
    model_names = ["Basic_LSTM", "GRU", "Stacked_LSTM"]
    master = pd.DataFrame()

    for dn in datanames:

        
        for model in model_names:
            path = dn + "/" + model + "/" 
            if not os.path.isdir(path) or not os.path.isfile(path + model + "_metrics_low_flow.csv"):
                continue
            
            name = dn + "_" + model
            segments = extract_segments(dn)

            metrics = pd.read_csv(path + model + "_metrics_low_flow.csv")

            col = pd.Series(metrics["Value"])
            col["BL"] = segments[0]
            col["FL"] = segments[1]
            col["model"] = model

            master[name] = col

    master.T.to_csv("metrics_master_lowflow.csv", index=False)


def compile_predictions(datanames):
    model_names = ["Basic_LSTM", "GRU", "Stacked_LSTM"]
    master = pd.DataFrame()
    predicts_list = []
    for dn in datanames:

        for model in model_names:
            path = dn + "/" + model + "/" 
            if not os.path.isdir(path):
                continue
            
            name = dn + "_" + model
            segments = extract_segments(dn)

            predicts = pd.read_csv(path + "/predict_results/" + model + "_predicts.csv")
            #predicts = pd.DataFrame(predicts.loc['2/10/2022 0:00':'3/10/2022 0:00'])
            predicts = predicts.set_index("datetime")
            predicts = pd.DataFrame(predicts["0"])
    
            #master[name] = predicts["0"]

            predicts["BL"] = segments[0][3:]
            predicts["FL"] = segments[1][3:]
            predicts["model"] = model

            predicts_list.append(predicts)

            print(predicts)
    #master.to_csv("master_predictions_rowwise.csv", index=False)

    pd.concat(predicts_list).to_csv("master_predictions.csv", index="datetime")

def compile_validation(datanames):

    model_names = ["Basic_LSTM", "GRU", "Stacked_LSTM"]
    master = pd.DataFrame()
    val_list = []
    for dn in datanames:

        for model in model_names:
            path = "validation_" + model + "_" + dn + ".csv"
            
            
            validation = pd.read_csv(rf"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\lib\lib\model_results\VALIDATION\{dn}")
            print(validation)
            val_list.append(validation)

    #master.to_csv("master_predictions_rowwise.csv", index=False)

    pd.concat(val_list).to_csv("master_validation.csv", index=False)



#compile_predictions(os.listdir())
#compile_metrics()

compile_validation(os.listdir(r"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\lib\lib\model_results\VALIDATION"))