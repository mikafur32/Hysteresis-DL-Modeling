import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
import os
from keras import mixed_precision

from datetime import datetime

'''

TODO: memory management!!! 

'''
from keras import backend as K

#os.chdir(".\\lib")
#print(os.getcwd())

#import models_base


import models_cuda
import ingest, predict


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate 13GB of memory on the first GPU
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

#policy = mixed_precision.Policy('mixed_float16')

#mixed_precision.set_global_policy(
#    policy
#)

# Function to train and evaluate ML forecasting models
  # Saves trained models, predictions, and validation loss to files
def evaluate(csv, saveto, columns, target, data_name, event_start, event_end, 
             epochs= 10, train_test_ratio=0.8, train_range=None, test_range=None, 
             n_past= 96, n_future= 12, train_flag= True, plotstep= "Month", scaler= True):

    date = datetime.now().strftime("%B_%d_%Y_%H_%M")  # Get current date-time for file naming

    # Split data into train/test periods
    if train_range == None or test_range == None:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_test_ratio= 0.8)

    else:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
    
    trainX, trainY = ingest.reshape(train_scaled, n_past, n_future)#, timestep_type= "hr")
    testX, testY = ingest.reshape(test_scaled,  n_past, n_future)#, timestep_type= "hr")

    # List of models to evaluate
    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']#'Bidirectional_LSTM',]

    # Ensure event_start and event_end are lists for multiple test events
    if not isinstance(event_start, (list, tuple)):
        event_start = [event_start]
    if not isinstance(event_end, (list, tuple)):
        event_end = [event_end]

    # Print debug info about event ranges
    print(f"event_start: {event_start} (Type: {type(event_start)})")
    print(f"event_end: {event_end} (Type: {type(event_end)})")

    # Create list of event time ranges
    event_ranges = [(start, end) for start, end in zip(event_start, event_end)]

    # Loop through each model for training and evaluation
    validation_loss_list = []
    trained_models = {}

    # TRAIN MODELS ONCE
    for model_name in model_names:
            if train_flag:

                print(f'Training {model_name}')

                # Train
                model = models_cuda.train_models(model_name, trainX, trainY, epochs, batch_size=32, loss= "mse", load_models=False, data_name= data_name)
                
                trained_models[model_name] = model # Save the trained model

                # Compute Validation Loss
                validation_loss = models_cuda.evaluate_model(model, testX, testY)
                validation_loss_list.append([validation_loss, data_name, model_name])

                # Plot Validation & Training losses
                models_cuda.plot_model(model_name, validation_loss, data_name)

                # Clear TensorFlow/Keras session to free up memory
                K.clear_session()

            else:
                # If already trained, get model & predict
                model = models_cuda.get_model(model_name, saveto=saveto, data=data_name)
                trained_models[model_name] = model  # Load trained model

    # Evaluate trained models per event            
    for model_name, model in trained_models.items():
        print(f'Generating predictions for {model_name}')
        _predict(saveto, event_ranges, model_name, testX, testY, test_dates, data_name, plotstep=plotstep, scaler=scaler)
        
    # Save Validation Losses
    validation_loss_df = pd.DataFrame(validation_loss_list, columns=['Validation Loss', 'Data Name', 'Model Name'])
    csv_path = os.path.join(saveto, f"{data_name}_validation.csv")
    validation_loss_df.to_csv(csv_path, index=False)
    

# Old function that works but trains multi times for multi events
'''def evaluate(csv, saveto, columns, target, data_name, event_start, event_end, 
             epochs= 1, train_test_ratio=0.8, train_range=None, test_range=None, 
             n_past= 96, n_future= 12, train_flag= True, plotstep= "Month", scaler= True):

    date = datetime.now().strftime("%B_%d_%Y_%H_%M")  # Get current date-time for file naming

    # Split data into train/test periods
    if train_range == None or test_range == None:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_test_ratio= 0.8)

    else:
        train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
    
    trainX, trainY = ingest.reshape(train_scaled, n_past, n_future)#, timestep_type= "hr")
    testX, testY = ingest.reshape(test_scaled,  n_past, n_future)#, timestep_type= "hr")

    # List of models to evaluate
    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']#'Bidirectional_LSTM',]

    # Ensure event_start and event_end are lists for multiple test events
    if not isinstance(event_start, (list, tuple)):
        event_start = [event_start]
    if not isinstance(event_end, (list, tuple)):
        event_end = [event_end]

    # Print debug info about event ranges
    print(f"event_start: {event_start} (Type: {type(event_start)})")
    print(f"event_end: {event_end} (Type: {type(event_end)})")

    # Create list of event time ranges
    event_ranges = [(start, end) for start, end in zip(event_start, event_end)]

    # Loop through each model for training and evaluation
    validation_loss_list = []
    for model_name in model_names:
            if train_flag:

                print(f'Evaluating {model_name}')

                # Train
                model = models_cuda.train_models(model_name, trainX, trainY, epochs, batch_size=32, loss= "mse", load_models=False, data_name= data_name)
                
                # Predictions and plot results
                _predict(saveto, event_ranges, model_name, testX, testY, test_dates, data_name, plotstep=plotstep, scaler=scaler) 

                # Extracting word segments from data_name for validation loss csv  
                def extract_segments(dir_name):
                    parts = dir_name.split("_")
                    bl_part = parts[0] + "_" + parts[1]
                    fl_part = parts[2] + "_" + parts[3][:parts[3].find('WSSVQ')]  # May have to change this for other model configurations
                    return bl_part, fl_part
                
                # Compute Validation Loss
                validation_loss = models_cuda.evaluate_model(model, testX, testY)
                seg = extract_segments(data_name)
                validation_loss_list.append([validation_loss, seg[0] ,seg[1] , model_name ])

                # Plot Validation & Training losses
                models_cuda.plot_model(model_name, validation_loss, data_name)

                # Clear TensorFlow/Keras session to free up memory
                K.clear_session()

                # Saving Validation Loss to CSV
                validation_loss_df = pd.DataFrame(validation_loss_list, columns=['Validation Loss', 'BL', 'FL', 'Model Name'])
                csv_path = rf"{saveto}/{data_name}\{model_name}\{model_name}_{data_name}_validation.csv"
                
                # Create the directory if it doesn't exist 
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                validation_loss_df.to_csv(csv_path, index=False)
            else:
                # If already trained, get model & predict
                model = models_cuda.get_model(model_name, saveto=saveto, data=data_name)
                _predict(saveto, event_start, event_end, model_name, testX, testY, test_dates, data_name, plotstep=plotstep, scaler= scaler)'''

# MAYBE HERE needs to be edited to keep the scaled OBS values for the metrics calc

# Makes predictions using the trained model and plots results (used within above evaluate())
def _predict(saveto, event_ranges, model_name, testX, testY, test_dates, data_name, plotstep="Month", scaler=None):  # changed from scaler=True, plotstep="Month"): 3/20
    
    print(f"Predicting {model_name}")

    # Debug: Print the event ranges received
    print(f"Received event ranges: {event_ranges} (Type: {type(event_ranges)})")

    # Generate predictions using the trained model
    predicts = predict.predict(model_name, testX, saveto, data_name)

    # Debug: Check shape and type of predictions
    print(f"Predictions shape: {predicts.shape if hasattr(predicts, 'shape') else 'Unknown'}")

    # Debug: Print test data info
    print(f"testX shape: {testX.shape if hasattr(testX, 'shape') else 'Unknown'}")
    print(f"testY shape: {testY.shape if hasattr(testY, 'shape') else 'Unknown'}")
    print(f"test_dates length: {len(test_dates) if hasattr(test_dates, '__len__') else 'Unknown'}")

    # Loop through each event range and plot predictions
    for tstart, tend in event_ranges:  # Unpacking each tuple (start, end)
        print(f"Processing event range: {tstart} to {tend}")

        predict.plot_predicts(saveto, model_name, predicts, testY, test_dates, data_name, 
                              scaler=scaler, event_ranges=[(tstart, tend)], event_plotstep=plotstep)
    
    # Loop through each event range (medium old version)
    #for event_range in event_ranges:
    #    tstart, tend = event_range  # Unpack start and end time
    #    print(f"Processing event range: {tstart} to {tend}")

# Old _predict function
'''def _predict(saveto, tstart, tend, model_name, testX, testY, test_dates, data_name, scaler=True, plotstep= "Month"):
    
    event_range = [tstart, tend]
    print(f"predicting {model_name} over {event_range}")

    predicts = predict.predict(model_name, testX, saveto, data_name)
    predict.plot_predicts(saveto, model_name, predicts, testY, test_dates, data_name, scaler=True, event_range= event_range, event_plotstep= plotstep)'''