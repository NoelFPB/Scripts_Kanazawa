import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('waveform_20250108_152846.csv')

# Define colors to match the oscilloscope
scope_colors = {
    'CHANnel1': 'yellow',
    'CHANnel2': 'cyan',
    'CHANnel3': 'magenta',
    'CHANnel4': 'blue'
}

# Create the plot
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')  # Use dark background to match scope

for channel in data['Channel'].unique():
    channel_data = data[data['Channel'] == channel]
    plt.plot(channel_data['Time (s)'], channel_data['Voltage (V)'], 
             color=scope_colors[channel], 
             label=channel)

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage vs Time')
plt.legend()
plt.grid(True, alpha=0.3)  # Semi-transparent grid
plt.show()