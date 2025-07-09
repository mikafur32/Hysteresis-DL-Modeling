import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.dates as mdates
import joblib

# Script for reading, preprocessing, and splitting time series data 
# for ML model input


# Function to read CSV data and preprocess
def read_in(csv, target, renames={}):

    # Read CSV file into df
    df = pd.read_csv(csv, low_memory=False)

    # Rename columns based on provided mapping
    df = df.rename(columns= renames)
    
    '''
    ", dtype= {"0": str, 
                                                        "1":np.float32,
                                                        "2":np.float32,
                                                        "3":np.float32})"
    '''
    
    # Reorganizes columns to be "datetime", target, feature values (renames) 
    df =  df[['datetime'] + [target] + list(filter(lambda x: x!=target, list(renames.values())))] if( target in list(renames.values())) else df[['datetime'] +[target] + list(renames.values())]
    df = df.set_index('datetime')  # Set datetime as index

    # NOTE
    # drop any rows with missing values
    df.dropna(axis = 0, inplace = True)

    # Reorder so that target is first.
    df = df[[target] + [x for x in renames.values() if x != target]]

    # Extract all datetime values for later use
    all_dates = df.index.to_series()
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")

    #print("ingest 2", df.head())

    return df, all_dates


# Function to split the data into training and testing sets
def train_test_split(df, train_range, test_range):
    train_from = train_range[0]
    train_to = train_range[1]

    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
   
    #print("TEST RANGE",test_range)

    test_from = test_range[0]
    test_to = test_range[1]
    
    #print("TEST RANGE 0", test_range[0])
    #print(test_range[1])

    # Return the subset of data within the training/testing date ranges
    return df[train_from:train_to], df[test_from:test_to]


# old ingest function - doesn't work for persistence model
# Function to process and scale data for training and testing
def ingest(csv, target, n_past=96, n_future=12, renames={}, train_range= None, test_range= None, train_test_ratio= None, scaler=True):
    
    df, all_dates = read_in(csv, target, renames)


    # If no range is assigned, will use the full range for training & testing
    train_range, test_range = [all_dates[0], all_dates[-1]], [all_dates[0], all_dates[-1]]
    
    """elif(train_test_ratio):
        train_range  = [all_dates[0], all_dates.iloc[int(np.floor(len(all_dates) * train_test_ratio))]]
        test_range   = [all_dates.iloc[int(np.ceil(len(all_dates) * train_test_ratio))], all_dates[-1]]
   """
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
    #print("pre-scale", df.head())

    ###### IF YOU ADD VARIABLES, add them here, too ######
    # Apply scaling to selected variables if scaler is enabled
    if scaler:
        scaler_WL = StandardScaler()
        scaler_Q = StandardScaler()
        scaler_V = StandardScaler()
        scaler_WSS = StandardScaler()

        # Fit and transform each scaler
        df['WL'] = scaler_WL.fit_transform(df["WL"].to_numpy().reshape(-1,1))
        df['Q'] = scaler_Q.fit_transform(df["Q"].to_numpy().reshape(-1,1))
        df['V'] = scaler_V.fit_transform(df["V"].to_numpy().reshape(-1,1))
        df['WSS'] = scaler_WSS.fit_transform(df["WSS"].to_numpy().reshape(-1,1))

        # Save each scaler to disk
        joblib.dump(scaler_WL, "scaler_WL.save")
        joblib.dump(scaler_Q, "scaler_Q.save")
        joblib.dump(scaler_V, "scaler_V.save")
        joblib.dump(scaler_WSS, "scaler_WSS.save")

        transformed_df = df

    else: 
        transformed_df = df
    
    # Validate validity of not having all cols in renames. 
    # transformed_df = pd.DataFrame(transformed_df, columns= list(renames.values()), index=df.index)
    transformed_df = pd.DataFrame(transformed_df, columns= df.columns, index=df.index)
    
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
    #print("post-scale", transformed_df.head())

    # Split into train and test
    train_scaled, test_scaled = train_test_split(transformed_df, train_range, test_range)

    # Get train and test timestamps for plotting
    train_dates = train_scaled.index.to_series()
    test_dates = test_scaled.index.to_series()

    # Convert DataFrame to NumPy arrays for model training
    train_scaled = train_scaled.to_numpy()
    test_scaled = test_scaled.to_numpy()

    return train_scaled, test_scaled, train_dates, test_dates, all_dates, scaler_WL


# new ingest function.. idk doesn't quite work
#def ingest(csv, target, n_past=96, n_future=12, renames={}, train_range=None, test_range=None, train_test_ratio=None, scaler=True):
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

        Y.append(scaled[i + n_future - 1 : i + n_future, 0]) #0 = Discharge 

    # Convert lists to numpy arrays
    X, Y = np.array(X), np.array(Y)

    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
    #print("X", X[0:5])
    #print("+++++++++++++++++++++++++++++++++++++++++++++++++\n")
    #print("Y", Y[0:5])

    return X, Y

