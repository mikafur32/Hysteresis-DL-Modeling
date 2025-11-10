# Script to analyze the hydraulic data at Henry, IL (reduced-complexity model generated data)
  # to determine the relationships between peak-phasing and flood magnitude!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks

# Load the data
file_path = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Henry_4vars_2017_2023.csv"
#file_path = "C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Ohio_River_data_preprocessed.csv"
df = pd.read_csv(file_path, skiprows=3)

print(df.columns)  # This will show the actual column names
print(df.shape)    # This will show the number of rows and columns
df = df.iloc[:, :5] # Drop the last column it's empty and giving issues

df.columns = ['datetime', 'Q', 'WL', 'V', 'WSS']
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year

# Smooth the Q data to reduce noise
df['Q_smooth'] = df['Q'].rolling(window=5, center=True).mean()

# Define event size categories based on peak Q thresholds
bins = [5297, 44002, 52477, 81223, np.inf]   # cfs (model-generated data)
#bins = [4000, 11593, 13653, 14600, np.inf]   # cms (KD's observed data)
labels = ['Base', 'Minor', 'Moderate', 'Major']

# Identify peaks in Q to detect events
# Adjust the parameters: 'height' to set a minimum peak height, 'distance' to set a minimum number of samples between peaks, and 'prominence' to filter out insignificant peaks
q_peaks, _ = find_peaks(df['Q_smooth'].fillna(0), height=4000, distance=500, prominence=5000)  # 3/7 changed distance from 1000

events = []

for peak_idx in q_peaks:
    event_time = df.loc[peak_idx, 'datetime']
    peak_q = df.loc[peak_idx, 'Q']
    
    # Define a window around peak Q
    window = df.iloc[max(0, peak_idx-50): min(len(df), peak_idx+50)]
    
    # Find peaks for WSS and WL within the window
    wss_peak_idx = window['WSS'].idxmax()
    wl_peak_idx = window['WL'].idxmax()
    
    wss_peak_time = df.loc[wss_peak_idx, 'datetime']
    wl_peak_time = df.loc[wl_peak_idx, 'datetime']
    
    # Calculate phase lags
    lag_wss_q = (event_time - wss_peak_time).total_seconds() / 3600  # hours
    lag_wss_wl = (wl_peak_time - wss_peak_time).total_seconds() / 3600
    
    # Classify event size based on peak Q
    eventsize_category = pd.cut([peak_q], bins=bins, labels=labels)[0]

    # Find base times for peak intensity calculations
    before_peak = df.loc[:peak_idx, 'Q']
    base_start_idx = before_peak.idxmin()
    after_peak = df.loc[peak_idx:, 'Q']
    base_end_idx = after_peak.idxmin()
    
    base_start_time = df.loc[base_start_idx, 'datetime']
    base_end_time = df.loc[base_end_idx, 'datetime']
    
    time_to_rise = (event_time - base_start_time).total_seconds() / 3600
    total_time = (base_end_time - base_start_time).total_seconds() / 3600
    
    intensity_ratio = time_to_rise / total_time if total_time > 0 else np.nan
    
    # Classify intensity
    if intensity_ratio < 0.3:
        intensity_category = "Low"
    elif 0.3 <= intensity_ratio < 0.6:
        intensity_category = "Average"
    else:
        intensity_category = "High"
    
    events.append({
        'peak_Q': peak_q,
        'lag_WSS_Q': lag_wss_q,
        'lag_WSS_WL': lag_wss_wl,
        'peak_time_Q': event_time,
        'peak_time_WSS': wss_peak_time,
        'peak_time_WL': wl_peak_time,
        'time_to_rise': time_to_rise,
        'total_time' : total_time,
        'intensity_ratio': intensity_ratio,
        'intensity_category': intensity_category,
        'eventsize_category': eventsize_category
    })

# Convert to DataFrame and save CSV
events_df = pd.DataFrame(events)
#events_df.to_csv("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\event_lag_analysis_OH.csv", index=False)


# Categorize events by peak Q (worked this into above I think)
#bins = [5297, 44002, 52477, 81223, np.inf]
#labels = ['Base', 'Minor', 'Moderate', 'Major']
events_df['Category'] = pd.cut(events_df['peak_Q'], bins=bins, labels=labels)

# Compute statistics for each category
stats = events_df.groupby('Category')[['lag_WSS_Q', 'lag_WSS_WL']].agg(['mean', 'std'])

# Display results
print(stats)

# Save to CSV (moved up)
#events_df.to_csv("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\event_lag_analysis.csv", index=False)

# Create box plots
plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='lag_WSS_Q', data=events_df, palette='Blues')
plt.xlabel('Event Size Category', fontsize=16)
plt.ylabel('Lag WSS-Q (hours)', fontsize=16)
plt.title('Box Plot of Lag WSS-Q by Event Size', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.savefig("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\boxplot_lag_WSS_Q_OH.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Category', y='lag_WSS_WL', data=events_df, palette='Reds')
plt.xlabel('Event Size Category', fontsize=16)
plt.ylabel('Lag WSS-WL (hours)', fontsize=16)
plt.title('Box Plot of Lag WSS-WL by Event Size', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.savefig("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\boxplot_lag_WSS_WL_OH.png")
plt.show()


# Define streamflow thresholds for categories
# cfs (model-generated data)
category_thresholds = {
    "Base": 5297,
    "Minor": 44002,
    "Moderate": 52477,
    "Major": 81223,
    "Extreme": np.inf  # Extreme is an open-ended category, so we won't plot a line for it
}
# cms (KD's observational data)
'''category_thresholds = {
    "Base": 4000,
    "Minor": 11593,
    "Moderate": 13653,
    "Major": 14600,
    "Extreme": np.inf  # Extreme is an open-ended category, so we won't plot a line for it
}'''


# Plot time series for each year with detected peaks and category lines
for year in range(2017, 2024):
    yearly_data = df[df['year'] == year]
    yearly_peaks = events_df[events_df['peak_time_Q'].dt.year == year]
    
    plt.figure(figsize=(12, 6))
    plt.plot(yearly_data['datetime'], yearly_data['Q'], label='Q', color='blue')
    plt.scatter(yearly_peaks['peak_time_Q'], yearly_peaks['peak_Q'], color='red', label='Peaks', s=100, zorder=3)
    
    # Get max datetime value for x-positioning of labels
    max_time = yearly_data['datetime'].min() 

    # Add faint horizontal lines with labels on the right side
    for category, threshold in category_thresholds.items():
        if threshold != np.inf:  # Skip plotting for "Extreme"
            plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
            plt.text(x=max_time, y=threshold, s=category, color='black', fontsize=10, 
                     verticalalignment='bottom', horizontalalignment='right', 
                     bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Q', fontsize=16)
    plt.title(f'Streamflow (Q) Time Series - {year}', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend()
    plt.savefig(f"C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Q_time_series_{year}_OH.png")
    #plt.show()  # Commenting out for now


'''# INTENSITY CORRELATIONS
# Define function to find baseflow start and end for each peak
def find_base_times(df, peak_idx):
    """Finds the base start and base end time for a given peak index."""
    peak_time = df.loc[peak_idx, 'datetime']
    
    # Search backwards for base start (first local minimum before peak)
    before_peak = df.loc[:peak_idx, 'Q']
    base_start_idx = before_peak.idxmin()
    
    # Search forward for base end (first local minimum after peak)
    after_peak = df.loc[peak_idx:, 'Q']
    base_end_idx = after_peak.idxmin()
    
    base_start_time = df.loc[base_start_idx, 'datetime']
    base_end_time = df.loc[base_end_idx, 'datetime']
    
    return base_start_time, base_end_time

# Process each peak to determine intensity
intensity_events = []

for peak_idx in q_peaks:
    event_time = df.loc[peak_idx, 'datetime']
    peak_q = df.loc[peak_idx, 'Q']
    
    base_start_time, base_end_time = find_base_times(df, peak_idx)
    
    time_to_rise = (event_time - base_start_time).total_seconds() / 3600  # in hours
    total_time = (base_end_time - base_start_time).total_seconds() / 3600  # in hours
    
    if total_time > 0:  # Avoid division by zero
        intensity_ratio = time_to_rise / total_time
    else:
        intensity_ratio = np.nan  # Skip problematic values
    
    # Classify intensity
    if intensity_ratio < 0.3:
        intensity_category = "Low"
    elif 0.3 <= intensity_ratio < 0.6:
        intensity_category = "Average"
    else:
        intensity_category = "High"
    
    intensity_events.append({'peak_Q': peak_q, 'time_to_rise': time_to_rise, 'total_time': total_time,
                             'intensity_ratio': intensity_ratio, 'intensity_category': intensity_category,
                             'peak_time_Q': event_time})

# Convert to DataFrame
intensity_df = pd.DataFrame(intensity_events)


# Merge intensity classifications into events_df
events_df = events_df.merge(intensity_df[['peak_time_Q', 'intensity_category']], on='peak_time_Q', how='left')

# Ensure the new column exists
print(events_df[['peak_time_Q', 'intensity_category']].head())  # Debugging check

# TODO: Save like you did before, too!
'''

# Define color mapping for intensity categories
intensity_colors = {'High': '#d73027', 'Average': '#e69f00', 'Low': '#1a9850'}

# Compute statistics for each category
stats = events_df.groupby('intensity_category')[['lag_WSS_Q', 'lag_WSS_WL']].agg(['mean', 'std'])

# Display results
print(stats)

# Create box plots for phase lag based on intensity category
plt.figure(figsize=(12, 6))
sns.boxplot(x='intensity_category', y='lag_WSS_Q', data=events_df, palette=intensity_colors)
plt.xlabel('Peak Intensity Category')
plt.ylabel('Lag WSS-Q (hours)')
plt.title('Box Plot of Lag WSS-Q by Peak Intensity')
plt.savefig("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\boxplot_lag_WSS_Q_intensity_OH.png")
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='intensity_category', y='lag_WSS_WL', data=events_df, palette=intensity_colors)
plt.xlabel('Peak Intensity Category')
plt.ylabel('Lag WSS-WL (hours)')
plt.title('Box Plot of Lag WSS-WL by Peak Intensity')
plt.savefig("C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\boxplot_lag_WSS_WL_intensity_OH.png")
plt.show()

# Plot time series for each year with intensity-colored peaks
for year in range(2017, 2024):
    yearly_data = df[df['year'] == year]
    yearly_peaks = events_df[events_df['peak_time_Q'].dt.year == year]

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_data['datetime'], yearly_data['Q'], label='Q', color='blue')

    # Plot peaks with different colors based on intensity category
    for category, color in intensity_colors.items():
        subset = yearly_peaks[yearly_peaks['intensity_category'] == category]
        plt.scatter(subset['peak_time_Q'], subset['peak_Q'], color=color, label=f'{category} Intensity', s=100, zorder=3)

    plt.xlabel('Date')
    plt.ylabel('Q')
    plt.title(f'Streamflow (Q) Time Series - {year}')
    plt.legend()
    plt.savefig(f"C:\\Users\\Mikey\\Documents\\Github\\Hysteresis-ML-Modeling\\data\\Q_time_series_intensity_{year}_OH.png")
    plt.show()