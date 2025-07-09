# Import required packages and scripts
import os, sys, argparse, re
import evaluate, ingest
import pandas as pd
from keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')


# =============================================================================
# Configure the run with arguments
# =============================================================================

# Assign the arguments
data = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv" # args.data
saveto = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\model_results" # args.saveto
model_names = "all" # args.model 
train_range = "['1/1/2017 0:00','12/31/2021 23:45']" #args.train_range
n_past = 48 #args.n_past  # Back Looking (BL) steps of 15-minute increments (for this data)
n_future = 72 #args.n_future  # Forward Looking (FL) steps of 15-minute increments (for this data)
test_range = [pd.Timestamp("1/1/2022 0:00"), pd.Timestamp("12/31/2022 23:45")]
event_range = [pd.Timestamp('2/10/2022 0:00'), pd.Timestamp('3/17/2022 23:45')]
dataname = "7_7_18hr_FL_12hr_BL"#args.dn

# From outdated version, use if you want to combine new postprocess.py with model_noCLI.py
#train = "n" #args.train
#test_range = "['1/1/2022 0:00','12/31/2022 23:45']"#args.test_range
#event_ranges = [
#    (pd.Timestamp('2/10/2022 0:00'), pd.Timestamp('3/17/2022 23:45')),  # First flood event
#    (pd.Timestamp('5/1/2022 0:00'), pd.Timestamp('5/26/2022 23:45')),   # Third flood event
#    (pd.Timestamp('7/20/2022 0:00'), pd.Timestamp('8/7/2022 23:45'))    # Low flow event
#]  # Version with timestamps (modifying code to work with this for robustness)
# If you want to support single events too, allow event_range as an alias for a single entry
#event_range = event_ranges[0]  # Default to the first event, maintaining compatibility


# Print statement to show raw arguments
print("Raw arguments:", sys.argv)


# =============================================================================
# Run Model
# =============================================================================


# Current model definition
WSSVQ_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V", "Q": "Q"}, "Name": "WSSVQ_WL"}

# Waterfall LSTM Definitions (old)
#WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}
#WSSV_Q = {"target": "Q", "features": { "WSS": "WSS", "V": "V"}, "Name": "WSSV_Q"}
#WSSVQ_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V", "Q": "Q"}, "Name": "WSSVQ_WL"}

# Other LSTM variations (to play with to prove concept)
WSS_V = {"target": "V", "features": { "WSS": "WSS"}, "Name": "WSS_V"}
#V_Q = {"target": "Q", "features": {"V": "V"}, "Name": "V_Q"}
V_WL = {"target": "WL", "features": {"V": "V"}, "Name": "V_WL"}
#Q_WL = {"target": "WL", "features": {"Q": "Q"}, "Name": "Q_WL"}
WSS_WL = {"target": "WL", "features": {"WSS": "WSS"}, "Name": "WSS_WL"}
#WSS_Q = {"target": "Q", "features": {"WSS": "WSS"}, "Name": "WSS_Q"}
WL_WL = {"target": "WL", "features": { "WL":"WL"}, "Name": "Persistence_WL"}
WSSV_WL = {"target": "WL", "features": { "WSS": "WSS", "V": "V"}, "Name": "WSSV_WL"}

# Define tests
tests= [WSSVQ_WL]  #WSSVQ_WL, WSS_V, WSSV_Q, WL_WL]


for test in tests:
        
    data_name = dataname + f"{test['Name']}"
    print(f"\n=============Running {data_name} =============\n")
    
    event_start, event_end = event_range[0], event_range[1]
    #for event_start, event_end in event_ranges:  # Loop through all event ranges # For multi-event
        #print(f"Processing event from {event_start} to {event_end}") # For multi-event
   
    evaluate.evaluate(data, saveto, test["features"], test["target"],
                    data_name, train_range=train_range, test_range=test_range,
                    event_start=event_start, event_end=event_end,n_past=n_past,# epochs=epochs,
                    n_future=n_future, train_flag=train_range, #predict_flag= True, 
                    plotstep=None)