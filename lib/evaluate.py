# Import required packages 
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
from keras import backend as K


# Import required scripts/modules 
import models_cuda  
import ingest, predict

# Function to train and evaluate ML forecasting models
  # Saves trained models, predictions, and validation loss to files
def evaluate(csv, saveto, columns, target, data_name, event_start, event_end, 
             n_past, n_future, plotstep,
             epochs= 10, train_range=None, test_range=None, 
             train_flag= True, scaler= True):

    # Split data into train/test periods 
    train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler = ingest.ingest(csv, target, renames= columns, train_range= train_range, test_range= test_range)#train_test_ratio= 0.8)
    
    trainX, trainY = ingest.reshape(train_scaled, n_past, n_future)
    testX, testY = ingest.reshape(test_scaled,  n_past, n_future)

    # List of models to evaluate
    model_names = ['Basic_LSTM', "GRU", 'Stacked_LSTM']

    # Print info about event ranges
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
        # TODO: replace with the new post processing
 
    # Save Validation Losses
    validation_loss_df = pd.DataFrame(validation_loss_list, columns=['Validation Loss', 'Data Name', 'Model Name'])
    csv_path = os.path.join(saveto, f"{data_name}_validation.csv")
    validation_loss_df.to_csv(csv_path, index=False)
    

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

    # Loop through each event range and plot predictions (if multiple test events, works either way)
    for tstart, tend in event_ranges:  # Unpacking each tuple (start, end)
        print(f"Processing event range: {tstart} to {tend}")

        predict.plot_predicts(saveto, model_name, predicts, testY, test_dates, data_name, 
                              scaler=scaler, event_ranges=[(tstart, tend)], event_plotstep=plotstep)
    