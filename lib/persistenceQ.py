import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configuration
dataname = "Persistence_Model_12hr_Q"
shift = 48  # 12 hours in 15-minute intervals
data_path = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv"
output_dir = r"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\model_results\Persistence_Model_12hr_Q"
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


def smooth_predictions(df, window=50):
    """Apply smoothing to predictions"""
    df["Q_pred_smooth"] = df["Q_pred"].rolling(window=window, center=True, min_periods=1).mean()
    return df

def calculate_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

def plot_event(df, start_date, end_date, output_dir):
    """Create and save plot for a specific event period"""
    event_df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    
    # Debug print to check values
    print("\nEvent Data Sample:")
    print(event_df[["datetime", "Q_obs", "Q_pred_smooth"]].head())
    
    metrics = calculate_metrics(event_df["Q_obs"], event_df["Q_pred_smooth"])
    
    plt.figure(figsize=(12, 6))
    plt.plot(event_df["datetime"], event_df["Q_obs"], label="Observed Q", color="black", linewidth=2)
    plt.plot(event_df["datetime"], event_df["Q_pred_smooth"], label="Persistence Model (12hr)", color="fuchsia", linewidth=2)

    # Format metrics text
    mse = f"{metrics['MSE']:.4g}"
    if metrics['MSE'] >= 1000:
        mse = f"{float(mse):,.4g}"
    mae = f"{metrics['MAE']:.4g}"
    
    metrics_text = (
        f"MSE: {mse}\n"
        f"MAE: {mae}\n"
        f"R²: {metrics['R2']:.3f}"
    )
    
    plt.gcf().text(
        0.75, 0.75,
        metrics_text,
        fontsize=16,
        bbox=dict(facecolor='white', alpha=1.0)
    )
    
    plt.title(f"Persistence Model (12hr) vs Observed ({start_date} to {end_date})", fontsize=16)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Discharge (cms)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.grid()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    # Save plot
    plot_filename = f"persistence_event_{start_date}_{end_date}_predictions.png"
    plt.savefig(os.path.join(output_dir, plot_filename), dpi=300)
    plt.close()

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
    
    # Unscale predictions and observations
   # pred_df = unscale_predictions(pred_df, scaler_Q)
    
    # Debug print
    print("\nUnscaled Data Summary:")
    print(pred_df[["Q_obs", "Q_pred"]].describe())
    
    # Smooth predictions
    pred_df = smooth_predictions(pred_df)
    
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