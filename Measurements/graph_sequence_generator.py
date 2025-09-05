import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

def load_data_from_string(data_string):
    """Load data from the string format"""
    return pd.read_csv(StringIO(data_string), sep=' ')

def load_data_from_file(filename):
    """Alternative: Load data from CSV file"""
    return pd.read_csv(filename)

def identify_states(df):
    """Identify the logic states based on A and B values"""
    states = []
    for _, row in df.iterrows():
        if row['A'] < 1 and row['B'] < 1:
            states.append('00')
        elif row['A'] < 1 and row['B'] > 4:
            states.append('01')
        elif row['A'] > 4 and row['B'] < 1:
            states.append('10')
        elif row['A'] > 4 and row['B'] > 4:
            states.append('11')
        else:
            states.append('??')  # Transition or unknown state
    
    df['state'] = states
    return df

def create_simple_continuous_graphs(df, save_plots=True):
    """Alternative: Create simple continuous line plots without state coloring"""
    
    # Set up the figure with 3 subplots sharing x-axis
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Sort dataframe by timestamp to ensure continuous lines
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    
    # Graph 1: Input A vs Time - Simple continuous line
    ax1.plot(df_sorted['timestamp'], df_sorted['A'], linewidth=2, color='blue', alpha=0.8)
    ax1.set_ylabel('Input A (V)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 5.5)
    
    # Graph 2: Input B vs Time - Simple continuous line
    ax2.plot(df_sorted['timestamp'], df_sorted['B'], linewidth=2, color='orange', alpha=0.8)
    ax2.set_ylabel('Input B (V)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 5.5)
    
    # Graph 3: Output vs Time - Simple continuous line
    ax3.plot(df_sorted['timestamp'], df_sorted['OUT'], linewidth=2, color='red', alpha=0.8)
    ax3.set_ylabel('Output (V)', fontsize=12)
    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines to show state transitions
    state_changes = []
    prev_state = df_sorted['state'].iloc[0]
    for i, row in df_sorted.iterrows():
        if row['state'] != prev_state:
            state_changes.append(row['timestamp'])
            prev_state = row['state']
    
    for change_time in state_changes:
        ax1.axvline(x=change_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axvline(x=change_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax3.axvline(x=change_time, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the plots if requested
    if save_plots:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        filename = f'three_aligned_simple_continuous_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    return fig

def print_summary(df):
    """Print a summary of the data"""
    print("=== DATA SUMMARY ===")
    print(f"Total data points: {len(df)}")
    print(f"Time range: {df['timestamp'].min():.2f}s to {df['timestamp'].max():.2f}s")
    print(f"Duration: {df['timestamp'].max() - df['timestamp'].min():.2f} seconds")
    
    print(f"\nInput A range: {df['A'].min():.3f}V to {df['A'].max():.3f}V")
    print(f"Input B range: {df['B'].min():.3f}V to {df['B'].max():.3f}V")
    print(f"Output range: {df['OUT'].min():.3f}V to {df['OUT'].max():.3f}V")
    
    print(f"\nState distribution:")
    state_counts = df['state'].value_counts().sort_index()
    for state, count in state_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  State {state}: {count} points ({percentage:.1f}%)")

def main():
    """Main function to run the analysis"""
    # Load data (using the string data provided)
    #df = load_data_from_string(data_string)
    
    # Alternative: Load from file
    df = load_data_from_file('micro.csv')
    
    # Identify states
    df = identify_states(df)
    
    # Print summary
    print_summary(df)
    
    # Choose which type of plot you want:
    
    # Option 2: Simple continuous lines (cleaner look)
    print("\nCreating simple continuous plots...")
    fig2 = create_simple_continuous_graphs(df, save_plots=True)
    
    # Show the plots
    plt.show()
    
    return df, fig2

if __name__ == "__main__":
    df, fig1, fig2 = main()