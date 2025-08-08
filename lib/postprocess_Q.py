# New Post-processing script
# 4/5/25 

# Later, turn into functions and work into main script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib  # for loading saved scalers

# Read in ML model predictions
# For a given dataname (i.e., "4_5_BL_12hr_FL_1dyWSSVQ_WL", "4_17_1dy_FL_1dyWSSVQ_WL")
# For all three trained models
# Specify variables (USER - hi)
dataname = "8_6_12hr_FL_12hr_BLWSSV_Q"
shift = 96   # shift = n_past + n_future
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
    df_obs = df_obs[["datetime", "Q"]]
    df_obs = df_obs.rename(columns={"Q": "Q_obs"})
    return df_obs

#scaler_path = rf"C:\Users\Mikey\Documents\Github\Hysteresis-ML-Modeling\model_results\{dataname}\scaler.save"

#load scaler for targetted variable - change between runs as needed
scaler_Q = joblib.load("scaler_Q.save")

# Unscale predictions using saved StandardScaler
def unscale_predictions(df, scaler):
    # Reshape to (n_samples, 1) as expected by inverse_transform
    df["WL_pred"] = scaler_Q.inverse_transform(df[["WL_pred"]])
    return df

# OR?
# Unscale your predictions (assumes 2D)
#preds_unscaled = scaler_WL.inverse_transform(df)

# Smooth the predictions!!!
# 5-point Moving Average 
def smooth_predictions(df, window=50):
    df["Q_pred_smooth"] = df["WL_pred"].rolling(window=window, center=True, min_periods=1).mean()
    return df


# Merge with observations
#def merge_with_observations(pred_df, obs_df):
#    return pd.merge(pred_df, obs_df, on="datetime", how="inner")
# With SHIFTING:
def merge_with_observations(pred_df, obs_df, shift = shift):
    # Sort both DataFrames just in case
    pred_df = pred_df.sort_values("datetime").reset_index(drop=True)
    obs_df = obs_df.sort_values("datetime").reset_index(drop=True)
    # Shift the observed values backward (to earlier timestamps)
    obs_df_shifted = obs_df.copy()
    obs_df_shifted["Q_obs"] = obs_df_shifted["Q_obs"].shift(-shift)
    # Drop rows with NaN after shift (these are at the end)
    obs_df_shifted = obs_df_shifted.dropna(subset=["Q_obs"]).reset_index(drop=True)
    # Keep only the original datetimes for alignment
    obs_df_shifted["datetime"] = obs_df["datetime"][:len(obs_df_shifted)].reset_index(drop=True)
    # Merge on datetime
    merged = pd.merge(pred_df, obs_df_shifted, on="datetime", how="inner")
    return merged




# Recalculate the evaluation statistics
def calculate_metrics(y_true, y_pred):
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred)
    }

# Create plot for entire test period or event
# DEFINE start_date and end_date probably!!!!
# Include evaluation statistics 
def plot_event(df, start_date, end_date, model_type):
    event_df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]

    metrics = calculate_metrics(event_df["Q_obs"], event_df["Q_pred_smooth"])
    plt.figure(figsize=(12, 6))
    plt.plot(event_df["datetime"], event_df["Q_obs"], label="Observed Q", color="black", linewidth=2)
    plt.plot(event_df["datetime"], event_df["Q_pred_smooth"], label="Predicted Q (Smoothed)", color="fuchsia", linewidth=2)

    # Show metrics on plot
    metrics_text = f"MSE: {metrics['MSE']:.3f}\nMAE: {metrics['MAE']:.3f}\nR²: {metrics['R2']:.3f}"
    plt.gcf().text(0.75, 0.65, metrics_text, fontsize=16, bbox=dict(facecolor='white', alpha=0.5))

    plt.title(f"{model_type} Predictions vs Observed ({start_date} to {end_date})", fontsize=16)
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Discharge (Q)", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid()

    output_path = os.path.join(base_path, model_type, "predict_results", f"{model_type}_event_{start_date}_{end_date}_predictions.png")
    plt.savefig(output_path, dpi=300)
    plt.close()


# RUN IT ALL!
if __name__ == "__main__":
    obs_df = load_observed_data()

    for model in model_types:
        pred_df = load_and_clean_predictions(model)
        pred_df = unscale_predictions(pred_df, scaler_Q)

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