import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.dates as mdates
import joblib

# Function to read CSV data and preprocess
def read_in(csv, target, renames={}):
    df = pd.read_csv(csv, low_memory=False)

    # Rename columns based on provided mapping
    df = df.rename(columns= renames)
    
    # Reorganizes columns to be "datetime", target, feature values (renames) 
    df =  df[['datetime'] + [target] + list(filter(lambda x: x!=target, list(renames.values())))] if( target in list(renames.values())) else df[['datetime'] +[target] + list(renames.values())]
    df = df.set_index('datetime')  # Set datetime as index

    # drop any rows with missing values
    df.dropna(axis = 0, inplace = True)

    # Reorder so that target is first.
    df = df[[target] + [x for x in renames.values() if x != target]]

    # Extract all datetime values for later use
    all_dates = df.index.to_series()

    return df, all_dates


# Function to split the data into training and testing sets
def train_test_split(df, train_range, test_range):
    train_from = train_range[0]
    train_to = train_range[1]

    test_from = test_range[0]
    test_to = test_range[1]
    
    # Return the subset of data within the training/testing date ranges
    return df[train_from:train_to], df[test_from:test_to]


def ingest(csv, target, n_past=96, n_future=12, renames={}, train_range=None, test_range=None, train_test_ratio=None, scaler=True):
    df, all_dates = read_in(csv, target, renames)

    # If no range is assigned, will use the full range for training & testing
    train_range, test_range = [all_dates[0], all_dates[-1]], [all_dates[0], all_dates[-1]]

    if scaler:
        # Define a dictionary of scalers
        scalers = {
            "WL": StandardScaler(),
            "Q": StandardScaler(),
            "V": StandardScaler(),
            "WSS": StandardScaler()
        }

        # Apply scaling only if the variable exists in the dataset
        for var, sc in scalers.items():
            if var in df.columns:
                df[var] = sc.fit_transform(df[var].to_numpy().reshape(-1, 1))

        transformed_df = df

    else: 
        transformed_df = df

    # Ensure transformed_df maintains the original column structure
    transformed_df = pd.DataFrame(transformed_df, columns=df.columns, index=df.index)

    # Split into train and test
    train_scaled, test_scaled = train_test_split(transformed_df, train_range, test_range)

    # Get train and test timestamps for plotting
    train_dates = train_scaled.index.to_series()
    test_dates = test_scaled.index.to_series()

    train_scaled = train_scaled.to_numpy()
    test_scaled = test_scaled.to_numpy()

    # Return all scalers, but only those that were used will be relevant
    return train_scaled, test_scaled, train_dates, test_dates, all_dates, scalers.get("WL", None)

# Function to reshape data into sequences for time series forecasting
def reshape(scaled, n_past, n_future, timestep_type= 'hr'):
    X = []
    Y = []

    #Reformat input data into a shape: (n_samples x timesteps x n_features)

    for i in range(n_past, len(scaled) - n_future + 1):
        X.append(scaled[i - n_past : i, 1:scaled.shape[1]]) # 1:scaled.shape[1] = all columns except for the target column.
        
        ## NOTE: This assumes the target values are the 1st column.

        Y.append(scaled[i + n_future - 1 : i + n_future, 0]) 

    # Convert lists to numpy arrays
    X, Y = np.array(X), np.array(Y)

    return X, Y

