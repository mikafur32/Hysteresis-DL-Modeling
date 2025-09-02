import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # for loading saved scalers
import numpy as np
from scipy.signal import find_peaks


# Read in ML model predictions
# For a given dataname (i.e., "4_5_BL_12hr_FL_1dyWSSVQ_WL", "4_17_1dy_FL_1dyWSSVQ_WL")
# For all three trained models
# Specify variables (USER - hi)
dataname = "8_29_15hr_FL_1hr_BLWSSVQ_WL"
shift = 64   # shift = n_past + n_future
model_types = ["GRU", "Basic_LSTM", "Stacked_LSTM"]
base_path = rf"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\model_results\{dataname}"
data = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv"# args.data

# Load and clean the predictions and observations
def load_and_clean_predictions(model_type):
    pred_path = os.path.join(base_path, model_type, "predict_results", f"{model_type}_predicts.csv")
    df = pd.read_csv(pred_path)
    df = df.loc[:, ~df.columns.duplicated()]  # remove duplicated columns
    df = df.rename(columns={df.columns[0]: "datetime", df.columns[1]: "WL_pred"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def load_observed_data():
    df_obs = pd.read_csv(data, parse_dates=["datetime"])
    df_obs = df_obs[["datetime", "WL"]]
    df_obs = df_obs.rename(columns={"WL": "WL_obs"})
    df_obs["WL_obs"] = df_obs["WL_obs"] * 0.3048
    return df_obs

#scaler_path = rf"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\model_results\{dataname}\scaler.save"

#load scaler for targetted variable - change between runs as needed
scaler_WL = joblib.load("scaler_WL.save")

# Unscale predictions using saved StandardScaler
def unscale_predictions(df, scaler):
    # Reshape to (n_samples, 1) as expected by inverse_transform
    df["WL_pred"] = scaler_WL.inverse_transform(df[["WL_pred"]])
    df["WL_pred"] = df["WL_pred"] * 0.3048
    return df

# OR?
# Unscale your predictions (assumes 2D)
#preds_unscaled = scaler_WL.inverse_transform(df)

# Smooth the predictions!!!
# 5-point Moving Average 
def smooth_predictions(df, window=50):
    df["WL_pred_smooth"] = df["WL_pred"].rolling(window=window, center=True, min_periods=1).mean()
    return df


# Merge with observations
#def merge_with_observations(pred_df, obs_df):
#    return pd.merge(pred_df, obs_df, on="datetime", how="inner")
# With SHIFTING:
def merge_with_observations(pred_df, obs_df, shift=shift):
    # Sort both DataFrames just in case
    pred_df = pred_df.sort_values("datetime").reset_index(drop=True)
    obs_df = obs_df.sort_values("datetime").reset_index(drop=True)
    
    # Shift the predicted values forward (to later timestamps)
    pred_df_shifted = pred_df.copy()
    pred_df_shifted["WL_pred"] = pred_df_shifted["WL_pred"].shift(shift)
    
    # Drop rows with NaN after shift (these are at the beginning)
    pred_df_shifted = pred_df_shifted.dropna(subset=["WL_pred"]).reset_index(drop=True)
    
    # Keep only the original datetimes for alignment
    pred_df_shifted["datetime"] = pred_df["datetime"][shift:shift+len(pred_df_shifted)].reset_index(drop=True)
    
    # Merge on datetime
    merged = pd.merge(pred_df_shifted, obs_df, on="datetime", how="inner")
    return merged


def calculate_peak_metrics(y_true, y_pred, datetime_index):
    """Calculate peak metrics with enhanced validation and debugging"""
    print("\n=== Debugging Peak Detection ===")
    print(f"Input sizes - y_true: {len(y_true)}, y_pred: {len(y_pred)}, times: {len(datetime_index)}")
    
    metrics = {
        "Peak_Magnitude_Diff": np.nan,
        "Peak_Timing_Diff": np.nan,
        "Observed_Peak_Value": np.nan,
        "Predicted_Peak_Value": np.nan
    }

    try:
        # Ensure we have valid data
        if y_true.isna().all() or y_pred.isna().all():
            print("Warning: All values are NaN")
            return metrics
            
        # Convert to numpy arrays (handling pandas Series)
        y_true = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        y_pred = y_pred.values if hasattr(y_pred, 'values') else np.array(y_pred)
        datetime_index = datetime_index.values if hasattr(datetime_index, 'values') else np.array(datetime_index)
        
        # Find peaks using nanargmax (ignores NaN values)
        obs_peak_idx = np.nanargmax(y_true)
        pred_peak_idx = np.nanargmax(y_pred)
        
        print(f"Peak indices - Obs: {obs_peak_idx}, Pred: {pred_peak_idx}")
        print(f"Sample values around peaks:")
        print(f"Observed: {y_true[obs_peak_idx-2:obs_peak_idx+3]}")
        print(f"Predicted: {y_pred[pred_peak_idx-2:pred_peak_idx+3]}")
        
        # Get peak values and times
        obs_peak_value = y_true[obs_peak_idx]
        pred_peak_value = y_pred[pred_peak_idx]
        obs_peak_time = datetime_index[obs_peak_idx]
        pred_peak_time = datetime_index[pred_peak_idx]
        
        # Calculate magnitude difference
        metrics["Peak_Magnitude_Diff"] = float(abs(obs_peak_value - pred_peak_value))
        metrics["Observed_Peak_Value"] = float(obs_peak_value)
        metrics["Predicted_Peak_Value"] = float(pred_peak_value)
        
        # Calculate timing difference (handle both datetime and timedelta)
        if isinstance(obs_peak_time, np.datetime64) and isinstance(pred_peak_time, np.datetime64):
            time_diff = (pred_peak_time - obs_peak_time) / np.timedelta64(1, 'h')
        else:
            time_diff = np.nan
            print("Warning: Unexpected time format - cannot calculate time difference")
        
        metrics["Peak_Timing_Diff"] = float(time_diff)
        
        print("\n=== Peak Metrics ===")
        print(f"Observed Peak: {obs_peak_value:.2f} m at {obs_peak_time}")
        print(f"Predicted Peak: {pred_peak_value:.2f} m at {pred_peak_time}")
        print(f"Magnitude Difference: {metrics['Peak_Magnitude_Diff']:.2f} m")
        print(f"Timing Difference: {metrics['Peak_Timing_Diff']:.2f} hours")
        
    except Exception as e:
        print(f"Error in peak detection: {str(e)}", exc_info=True)
    
    return metrics

def plot_event(df, start_date, end_date, model_type):
    event_df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    
    # Calculate all metrics
    base_metrics = {
        "MSE": mean_squared_error(event_df["WL_obs"], event_df["WL_pred_smooth"]),
        "MAE": mean_absolute_error(event_df["WL_obs"], event_df["WL_pred_smooth"]),
        "R2": r2_score(event_df["WL_obs"], event_df["WL_pred_smooth"])
    }
    
    peak_metrics = calculate_peak_metrics(event_df["WL_obs"], event_df["WL_pred_smooth"], 
                                        event_df["datetime"])
    
    # Create figure with tight layout and adjusted subplot parameters
    plt.figure(figsize=(11, 5.5))
    plt.subplots_adjust(bottom=0.18, left=0.12, right=0.88, top=0.95)
    
    plt.plot(event_df["datetime"], event_df["WL_obs"], label="Observed WL", color="black", linewidth=2)
    plt.plot(event_df["datetime"], event_df["WL_pred_smooth"], label="Predicted WL", color="fuchsia", linewidth=2)
    
    # Format metrics text boxes
    # MODIFY THIS IF WANT TO REMOVE SOME METRICS FROM PLOT
    metrics_text = (
        f"MAE: {base_metrics['MAE']:.3f}\n"
        f"R²: {base_metrics['R2']:.3f}\n"
        f"Magnitude: {peak_metrics['Peak_Magnitude_Diff']:.2f} m\n"
        f"Timing: {peak_metrics['Peak_Timing_Diff']:.2f} hrs"
    )
    
    # Add text box
    plt.gcf().text(
        0.63, 0.72,  # Adjusted position (left to right, top to bottom)
        #0.7, 0.8,   # YEAR Adjusted position (left to right, top to bottom)
        metrics_text,
        fontsize=20,
        bbox=dict(facecolor='white', alpha=1.0)
    )
    
    plt.xlabel("Date", fontsize=24)
    plt.ylabel("Water Level (m)", fontsize=24)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    output_path = os.path.join(base_path, model_type, "predict_results", 
                             f"{model_type}_event_{start_date}_{end_date}_predictions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    return {**base_metrics, **peak_metrics}
    
# RUN IT ALL!
if __name__ == "__main__":
    obs_df = load_observed_data()

    for model in model_types:
        pred_df = load_and_clean_predictions(model)
        pred_df = unscale_predictions(pred_df, scaler_WL)

        # Before merging, print key details (debugging)
        print("Before shift:")
        print(obs_df.head(10))

        #print("After shift:")
        #print(obs_df_shifted.head(10))

        print("Predictions:")
        print(pred_df.head(10))
        
        merged_df = merge_with_observations(pred_df, obs_df)
        merged_df = smooth_predictions(merged_df)

        # User can specify event ranges here
        event_ranges = [
            ("2022-02-10", "2022-03-17"),
            ("2022-03-28", "2022-04-21"),
            ("2022-07-01", "2022-07-20"),
            ("2022-05-01", "2022-05-26"),
            ("2022-07-20", "2022-08-07"), # Add more events if needed
            ("2022-01-01", "2022-12-31")  # Full test year
        ]

        for start, end in event_ranges:
            plot_event(merged_df, start, end, model)

        print(f"Finished processing {model}")


# Some paneled figures?

# Some comparison figures, as before with the error statistics over each permutation
# Or just fix the one that is currently in main script

# Compare Basic, Stacked LSTM, and GRU and BL with same FL