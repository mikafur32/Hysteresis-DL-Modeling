import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuration
dataname = "Persistence_Model_1wk_Q"
shift = 672  # 12 hours in 15-minute intervals
data_path = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv"
output_dir = r"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\model_results\Persistence_Model_1wk_Q"
os.makedirs(output_dir, exist_ok=True)

# Load scaler
scaler_Q = joblib.load("scaler_Q.save")

def load_observed_data():
    """Load and prepare observed data"""
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    df = df[["datetime", "Q"]].rename(columns={"Q": "Q_obs"})
    df["Q_obs"] = df["Q_obs"] * 0.02831 # convert from cfs to m^3/s
    return df

def create_persistence_predictions(obs_df, shift):
    """Create persistence predictions by shifting observed data"""
    pred_df = obs_df.copy()
    pred_df["Q_pred"] = pred_df["Q_obs"].shift(shift)
    pred_df = pred_df.dropna().reset_index(drop=True)
    #pred_df["Q_pred"] = pred_df["Q_pred"] * 0.02831 
    return pred_df

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }
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
        print(f"Magnitude Difference: {metrics['Peak_Magnitude_Diff']:.2f} cms")
        print(f"Timing Difference: {metrics['Peak_Timing_Diff']:.2f} hours")
        
    except Exception as e:
        print(f"Error in peak detection: {str(e)}", exc_info=True)
    
    return metrics

def plot_event(df, start_date, end_date, output_dir):
    """Create and save plot matching the ML model style"""
    event_df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    
    # Calculate all metrics (same as ML version)
    base_metrics = {
        "MSE": mean_squared_error(event_df["Q_obs"], event_df["Q_pred"]),
        "MAE": mean_absolute_error(event_df["Q_obs"], event_df["Q_pred"]),
        "R2": r2_score(event_df["Q_obs"], event_df["Q_pred"])
    }
    
    # Calculate peak metrics (same function as ML version)
    peak_metrics = calculate_peak_metrics(event_df["Q_obs"], event_df["Q_pred"], 
                                        event_df["datetime"])
    
    # Create figure with identical dimensions and margins
    plt.figure(figsize=(11, 5.5))
    plt.subplots_adjust(bottom=0.18, left=0.12, right=0.88, top=0.95)
    
    # Plot with same styling
    plt.plot(event_df["datetime"], event_df["Q_obs"], label="Observed Q", color="black", linewidth=2)
    plt.plot(event_df["datetime"], event_df["Q_pred"], label="Persistence Model (12hr)", color="fuchsia", linewidth=2)
    
    # Format metrics text box 
    # MODIFY THIS IF WANT TO REMOVE SOME METRICS FROM PLOT
    metrics_text = (
        f"MAE: {base_metrics['MAE']:.3f}\n"
        f"R²: {base_metrics['R2']:.3f}\n"
        f"Magnitude: {peak_metrics['Peak_Magnitude_Diff']:.2f} cms\n"
        f"Timing: {peak_metrics['Peak_Timing_Diff']:.2f} hrs"
    )
    
    # Identical text box positioning and styling
    plt.gcf().text(
        0.63, 0.73,
        metrics_text,
        fontsize=18,
        bbox=dict(facecolor='white', alpha=1.0)
    )
    
    # Same axis styling
    plt.xlabel("Date", fontsize=24)
    plt.ylabel("Discharge (cms)", fontsize=24)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Save with same parameters
    plot_filename = f"persistence_event_{start_date}_{end_date}_predictions.png"
    plt.savefig(os.path.join(output_dir, plot_filename), 
               dpi=300, 
               bbox_inches='tight', 
               pad_inches=0.1)
    plt.close()
    
    return {**base_metrics, **peak_metrics}

def save_predictions(df, output_dir):
    """Save predictions to CSV"""
    df.to_csv(os.path.join(output_dir, "persistence_predictions.csv"), index=False)

if __name__ == "__main__":
    # Load and prepare data
    obs_df = load_observed_data()

    
    # Debug print
    print("Original Data Summary:")
    print(obs_df["Q_obs"].describe())
    
    # Create persistence predictions
    pred_df = create_persistence_predictions(obs_df, shift)
    
    # Debug print
    print("\nUnscaled Data Summary:")
    print(pred_df[["Q_obs", "Q_pred"]].describe())
    
    # Smooth predictions
 #   pred_df = smooth_predictions(pred_df)
    
    # Save predictions
    save_predictions(pred_df, output_dir)
    
    # Define event ranges
    event_ranges = [
        ("2022-02-10", "2022-03-17"),
        ("2022-07-01", "2022-07-20"),
        ("2022-05-01", "2022-05-26"),
        ("2022-07-20", "2022-08-07"),
        ("2022-01-01", "2022-12-31")  # Full test year
    ]
    
    # Create plots for each event
    for start, end in event_ranges:
        plot_event(pred_df, start, end, output_dir)
    
    print("Persistence model (Q) processing complete!")