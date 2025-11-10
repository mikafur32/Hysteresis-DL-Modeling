
import models_cuda
#import models
import ingest


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np


# Function to predict output using a ML model and test data
def predict(model_name, testX, saveto, dataname):
    model = models_cuda.get_model(model_name, dataname)   # Load the specified model
    print("predicting")
    return pd.DataFrame(model.predict(testX, verbose= 1))  # Run prediction and return results as df

# Function to plot predicted vs. observed values for the ML models
def plot_predicts(saveto, model_name, predicts, testY, test_dates, dataname, scaler=True, event_ranges=None, event_plotstep="Day"):
    testY = testY.flatten()
    predicts = predicts.to_numpy().flatten()

    predicts = pd.concat([pd.Series(predicts), pd.Series(predicts), pd.Series(predicts), pd.Series(predicts)], axis=1)
    testY = pd.concat([pd.Series(testY), pd.Series(testY), pd.Series(testY), pd.Series(testY)], axis=1)

    # Get the smallest shape
    shape = min(test_dates.shape[0], predicts.shape[0])

    testY.index = pd.to_datetime(test_dates[:shape])

    # Ensure uniform types
    predicts = predicts.astype(np.float64)
    predicts = pd.DataFrame(predicts)

    # Export predicts
    if not os.path.exists(f"{saveto}/{dataname}/{model_name}/predict_results"):
        os.makedirs(f"{saveto}/{dataname}/{model_name}/predict_results", exist_ok=True)

    # Set datetime indices
    predicts["datetime"] = test_dates[:shape].index
    predicts = predicts.set_index("datetime")

    # Evaluate the overall metrics
    metrics_df = evaluate_metrics(predicts, testY, dataname, model_name)

    # ----------------------
    # MODIFICATION: Loop Through Multiple Event Ranges
    # ----------------------
    if event_ranges is not None:
        print(f"Total event ranges provided: {len(event_ranges)}")
        print(f"event_ranges: {event_ranges} (Type: {type(event_ranges)})")
        for i, event_range in enumerate(event_ranges):
            print(f"Checking event range {i}: {event_range} (Type: {type(event_range)})")  # Debugging line
            print(f"Test range: {test_dates} (Type: {type(test_dates)})")
            print(f"Start type: {type(test_dates[0])}, End type: {type(test_dates[1])}")
            print(f"Min test date: {test_dates.min()}, Max test date: {test_dates.max()}")
            
            # Ensure event_range is unpacked correctly
            if isinstance(event_range, tuple) and len(event_range) == 2:
                start, end = event_range  # Unpack tuple
            else:
                raise TypeError(f"Unexpected format for event_range: {event_range} (Type: {type(event_range)})")
    
            # Extract event data and include time (00:00:00 for start, 23:45:00 for end)
            t_start = pd.to_datetime(event_range[0]).replace(hour=0, minute=0, second=0, microsecond=0)  # Set time to 00:00:00
            t_end = pd.to_datetime(event_range[1]).replace(hour=23, minute=45, second=0, microsecond=0)  # Set time to 23:45:00

            # Now, t_start and t_end will include the time part as well
            print(f"t_start: {t_start}, t_end: {t_end}")
            
            # Ensure that testY index is a DatetimeIndex
            testY.index = pd.to_datetime(testY.index)
            predicts.index = pd.to_datetime(predicts.index)

            # Check the first few indices of testY and predicts
            print(f"testY index type: {type(testY.index[0])}, predicts index type: {type(predicts.index[0])}")
            print(f"testY index (first 10): {testY.index[:10]}")
            print(f"predicts index (first 10): {predicts.index[:10]}")

            # Try to slice using the event range
            try:
                eventY = testY.loc[t_start:t_end]
                eventPredicts = predicts.loc[t_start:t_end]
            except KeyError as e:
                print(f"KeyError encountered: {e}")
                print("Available keys in testY:", testY.index[:10])
                print("Available keys in predicts:", predicts.index[:10])

            # Evaluate metrics for the event
            metrics_df = evaluate_metrics(eventPredicts, eventY, dataname, model_name)

            # Plot event predictions
            plt.figure()
            plt.title(f"{model_name} Predictions for Event {i+1}: {t_start} - {t_end}")
            plt.plot(pd.to_datetime(eventY.index), eventY.iloc[:, 0], label='Actual')
            plt.plot(pd.to_datetime(eventY.index), eventPredicts.iloc[:, 0], label='Predicted')

            # Format the x-axis
            ax = plt.gca()
            if event_plotstep == "Month":
                ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
            elif event_plotstep == "Week":
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))  # Display one tick per week
            elif event_plotstep == "Day":
                ax.xaxis.set_major_locator(mdates.DayLocator())  # Display one tick per day
            elif event_plotstep == "Hour":
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility
            plt.ylabel("WL (ft)")  
            plt.xlabel("Datetime")  
            plt.legend()
            plt.tight_layout()

            # Format timestamps to remove invalid characters
            start_str = event_range[0].strftime("%Y-%m-%d")
            end_str = event_range[1].strftime("%Y-%m-%d")

            # Save the file with a safe filename
            plt.savefig(f"{saveto}/{dataname}/{model_name}/predict_results/{model_name}_event_{start_str}_{end_str}_predictions.png")
            plt.close()

    # ----------------------
    # End of Event Loop
    # ----------------------

    # Save overall predictions
    predicts.to_csv(f"{saveto}/{dataname}/{model_name}/predict_results/{model_name}_predicts.csv")

    # Plot overall predictions
    plt.figure()
    plt.title(f"{model_name} Predictions")
    plt.plot(pd.to_datetime(testY.index), testY.iloc[:, 0], label='Observed')
    plt.plot(pd.to_datetime(testY.index), predicts.iloc[:, 0], label='Predicted')

    # Format the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Display one tick per month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set date format
    plt.ylabel("WL (ft)")  
    plt.xlabel("Datetime") 
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")  # Rotate the x-axis labels for better visibility
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{saveto}/{dataname}/{model_name}/predict_results/{model_name}_predictions.png")  
    plt.close()

    # Check if plots are actually being saved
    print(os.listdir(f"{saveto}/{dataname}/{model_name}/predict_results"))


# Function to evaluate model performance with metrics
def evaluate_metrics(predicts, y, dataname, model_name):
    import pandas as pd
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.spatial import KDTree
    from permetrics.regression import RegressionMetric

    # Convert w and predicts to numpy arrays for calculation
    y,predicts = y[0].to_numpy(), predicts[0].to_numpy()
    evaluator = RegressionMetric(y, predicts)

    # Calculate Kling-Gupta Efficiency (KGE)
    kge = evaluator.kling_gupta_efficiency(multi_output="raw_values")
        
    # Calculate Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Bias
    mse = np.mean((predicts - y) ** 2)
    rmse = np.sqrt(mse)
    bias = np.mean(predicts - y)

    # Print MSE, RMSE, and Bias
    print(mse,rmse,bias)

    # Find peaks in the true and predicted values 
    true_peaks, _ = find_peaks(y,distance=672)
    predicted_peaks, _ = find_peaks(predicts,distance=672)

    # Match peaks based on nearest neighbors
    tree = KDTree(true_peaks.reshape(-1, 1))
    distances, indices = tree.query(predicted_peaks.reshape(-1, 1))

    # If there are any peaks, calculate the mean peak error and peak timing error
    if len(true_peaks) > 0 and len(predicted_peaks) > 0:
        mean_peak_error = np.mean(np.abs(y[true_peaks[indices]] - predicts[predicted_peaks]))
        peak_timing_error = np.mean(np.abs(true_peaks[indices] - predicted_peaks))
    else:
        mean_peak_error = None
        peak_timing_error = None

    # Print the peak errors
    print(mean_peak_error, peak_timing_error)

    # Create a DataFrame from the results
    results_df = pd.DataFrame({
        'Metric': ['MSE', 'RMSE', 'Bias','KGE', 'Mean Peak Error', 'Peak Timing Error'],
        'Value': [mse, rmse, bias, kge, mean_peak_error, peak_timing_error]
    })
    print(results_df)

    # Ensure the directory exists before saving
    save_dir = f"lib/model_results/{dataname}/{model_name}"
    os.makedirs(save_dir, exist_ok=True)  # This line creates the directory if it doesn’t exist

    # Save the results DataFrame to a CSV file in the specified directory
    results_df.to_csv(f"{save_dir}/{model_name}_metrics.csv")
    return results_df
