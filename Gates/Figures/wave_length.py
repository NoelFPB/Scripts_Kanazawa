import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_expected_states(gate_name):
    """
    Define expected HIGH and LOW states for each gate based on truth tables
    Returns (high_indices, low_indices) where indices correspond to input combinations:
    0: A=0.01, B=0.01 (LOW, LOW)
    1: A=0.01, B=4.9  (LOW, HIGH) 
    2: A=4.9,  B=0.01 (HIGH, LOW)
    3: A=4.9,  B=4.9  (HIGH, HIGH)
    """
    truth_tables = {
        'AND':   ([3], [0, 1, 2]),           # Only A=1,B=1 is high
        'OR':    ([1, 2, 3], [0]),           # A=1 or B=1 is high
        'XOR':   ([1, 2], [0, 3]),           # A≠B is high
        'NAND':  ([0, 1, 2], [3]),           # NOT(A AND B)
        'NOR':   ([0], [1, 2, 3]),           # NOT(A OR B)
        'XNOR':  ([0, 3], [1, 2])            # A=B is high
    }
    return truth_tables.get(gate_name, ([3], [0, 1, 2]))  # Default to AND

def calculate_extinction_ratio_for_gate(gate_data, wavelength_cols, method='max_vs_second'):
    """
    Calculate extinction ratio for each wavelength for a specific gate
    """
    er_values = []
    gate_name = gate_data['Gate'].iloc[0]
    
    for wavelength in wavelength_cols:
        # Get all output values for this wavelength across all input combinations
        outputs = gate_data[wavelength].values
        
        if method == 'max_vs_second':
            # Sort outputs in descending order
            sorted_outputs = np.sort(outputs)[::-1]
            if len(sorted_outputs) >= 2:
                max_output = sorted_outputs[0]
                second_output = sorted_outputs[1]
                er_db = 10 * np.log10(max_output / second_output) if second_output > 0 else 0
            else:
                er_db = 0
                
        elif method == 'worst_case':
            # Truth table based worst-case analysis
            high_indices, low_indices = get_expected_states(gate_name)
            
            if len(high_indices) > 0 and len(low_indices) > 0:
                expected_highs = [outputs[i] for i in high_indices if i < len(outputs)]
                expected_lows = [outputs[i] for i in low_indices if i < len(outputs)]
                print(gate_data)
                print(expected_highs)
                if expected_highs and expected_lows:
                    min_high = min(expected_highs)
                    max_low = max(expected_lows)
                    er_db = 10 * np.log10(min_high / max_low) if max_low > 0 else 0
                else:
                    er_db = 0
            else:
                er_db = 0
                
        elif method == 'truth_table_average':
            # Truth table based average analysis
            high_indices, low_indices = get_expected_states(gate_name)
            
            if len(high_indices) > 0 and len(low_indices) > 0:
                expected_highs = [outputs[i] for i in high_indices if i < len(outputs)]
                expected_lows = [outputs[i] for i in low_indices if i < len(outputs)]
                
                if expected_highs and expected_lows:
                    avg_high = np.mean(expected_highs)
                    avg_low = np.mean(expected_lows)
                    er_db = 10 * np.log10(avg_high / avg_low) if avg_low > 0 else 0
                else:
                    er_db = 0
            else:
                er_db = 0
        
        er_values.append(er_db)
    
    return er_values

def plot_simple_extinction_ratios(file_path, method='worst_case', sheet_name=None, save_plots=True, show_plots=True):
    # Read the Excel file
    if sheet_name:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
    else:
        df = pd.read_excel(file_path)
    
    # Get wavelength columns (all numeric columns except Gate, A, B)
    wavelength_cols = [col for col in df.columns if isinstance(col, (int, float)) or 
                      (isinstance(col, str) and col.replace('.', '').replace('-', '').isdigit())]
    wavelengths = [float(col) for col in wavelength_cols]
    
    # Get unique gates
    gates = df['Gate'].unique()
    
    # Create 2x3 subplot grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for gate_idx, gate in enumerate(gates):
        ax = axes[gate_idx]
        gate_data = df[df['Gate'] == gate]
        
        # Calculate extinction ratios for this gate
        er_values = calculate_extinction_ratio_for_gate(gate_data, wavelength_cols, method)
        
        # Create bar chart
        bars = ax.bar(wavelengths, er_values, color='steelblue', alpha=0.8, width=0.15)
        
        # Customize the plot
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Extinction Ratio (dB)', fontsize=12)
        ax.set_title(f'Extinction Ratio vs Wavelength for {gate} Gate', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='both', labelsize=10)
        ax.tick_params(axis='x', rotation=45)
        
        # Set y-axis limits to handle negative values
        if er_values:
            y_min = min(er_values)
            y_max = max(er_values)
            margin = (y_max - y_min) * 0.1 if y_max != y_min else 1
            ax.set_ylim(y_min - margin, y_max + margin)
            
            # Add a horizontal line at y=0 for reference
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add value labels on top of bars (adjusted for negative values)
        for bar, value in zip(bars, er_values):
            # Position label above positive bars, below negative bars
            if value >= 0:
                va = 'bottom'
                y_pos = bar.get_height() + 0.05
            else:
                va = 'top' 
                y_pos = bar.get_height() - 0.05
                
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{value:.1f}', ha='center', va=va, fontsize=9)
    
    plt.tight_layout(pad=2.0)
    
    if save_plots:
        filename = f"extinction_ratio_all_gates_{method}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {filename}")
    
    if show_plots:
        plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "wv_clean.xlsx"
    
    # Option 1: Worst-case analysis (min_high vs max_low)
    plot_simple_extinction_ratios(file_path, method='worst_case')
    
    # Option 2: Truth table average analysis (avg_high vs avg_low)
    # plot_simple_extinction_ratios(file_path, method='truth_table_average')
    
    # Option 3: Maximum contrast (original method)
    # plot_simple_extinction_ratios(file_path, method='max_vs_second')
    
    print("Extinction ratios calculated and plotted!")