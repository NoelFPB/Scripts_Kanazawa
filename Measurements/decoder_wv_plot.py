import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import matplotlib.colors as mcolors

def load_latest_data():
    """Load the most recent Excel file"""
    excel_files = glob.glob("wavelength_sweep_4ch_*.xlsx")
    if not excel_files:
        print("No Excel files found!")
        return None
    
    latest_file = max(excel_files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading: {latest_file}")
    
    excel_file = pd.ExcelFile(latest_file)
    data = {}
    
    for sheet_name in excel_file.sheet_names:
        if sheet_name.startswith("Ch"):
            channel_num = int(sheet_name.split("_")[0].replace("Ch", ""))
            channel_label = sheet_name.split("_", 1)[1] if "_" in sheet_name else f"Ch{channel_num}"
            
            df = pd.read_excel(latest_file, sheet_name=sheet_name)
            data[channel_num] = {'label': channel_label, 'data': df}
    
    return data

# This mapping is CRITICAL and defines the decoder's truth table
# Key: Channel_Label (from sheet name), Value: Gate input that should make this channel HIGH
DECODER_TRUTH_MAP = {
    'Output_00': 'Gate_00', 
    'Output_01': 'Gate_01', 
    'Output_10': 'Gate_10', 
    'Output_11': 'Gate_11', 
    'Output_1': 'Gate_00', # Example: Ch1 is high for Gate_00 (ADJUST AS PER YOUR DESIGN)
    'Output_2': 'Gate_01', # Example: Ch2 is high for Gate_01 (ADJUST AS PER YOUR DESIGN)
    'Output_3': 'Gate_10', # Example: Ch3 is high for Gate_10 (ADJUST AS PER YOUR DESIGN)
    'Output_4': 'Gate_11'  # Example: Ch4 is high for Gate_11 (ADJUST AS PER YOUR DESIGN)
}

# This maps the Gate input (e.g. Gate_00) to the channel that should be HIGH for that input
# This is the INVERSE of DECODER_TRUTH_MAP, and useful for the new heatmap
GATE_TO_HIGH_CHANNEL_MAP = {v: k for k, v in DECODER_TRUTH_MAP.items()}


def calculate_channel_extinction_ratio_and_levels(df, wavelengths, channel_label):
    """
    Calculates ER and high/low levels for a specific CHANNEL.
    This remains largely the same as the previous version, correctly defining high/low for each channel.
    """
    extinction_ratios = {}
    high_levels = {}
    low_levels = {}
    
    intended_high_gate = DECODER_TRUTH_MAP.get(channel_label)
    
    if not intended_high_gate or intended_high_gate not in df['Gate'].unique():
        # print(f"Warning: Could not determine or find '{intended_high_gate}' as intended high gate for channel '{channel_label}'. Skipping ER calculation.")
        for wl in wavelengths:
            extinction_ratios[wl] = np.nan
            high_levels[wl] = np.nan
            low_levels[wl] = np.nan
        return extinction_ratios, high_levels, low_levels

    all_gate_types = df['Gate'].unique()
    intended_low_gates = [g for g in all_gate_types if g != intended_high_gate]

    for wavelength in wavelengths:
        wl_col = str(wavelength)
        if wl_col not in df.columns:
            extinction_ratios[wavelength] = np.nan
            high_levels[wavelength] = np.nan
            low_levels[wavelength] = np.nan
            continue
        
        high_state_voltages = df[df['Gate'] == intended_high_gate][wl_col].dropna().values
        low_state_voltages_all = df[df['Gate'].isin(intended_low_gates)][wl_col].dropna().values

        if len(high_state_voltages) > 0:
            max_high = np.max(high_state_voltages) 
        else:
            max_high = np.nan

        if len(low_state_voltages_all) > 0:
            max_low = np.max(low_state_voltages_all) 
        else:
            max_low = np.nan

        if np.isnan(max_high) or np.isnan(max_low):
            er_db = np.nan
        elif max_low == 0:
            er_db = 10 * np.log10(max_high / 1e-9) if max_high > 0 else np.nan
        elif max_high <= 0:
            er_db = np.nan
        else:
            er_db = 10 * np.log10(max_high / max_low)
            
        extinction_ratios[wavelength] = er_db
        high_levels[wavelength] = max_high
        low_levels[wavelength] = max_low
    
    return extinction_ratios, high_levels, low_levels

def calculate_input_extinction_ratio(all_channels_data, wavelengths, gate_type_to_analyze):
    """
    Calculates an 'Input Extinction Ratio' for a specific GATE_TYPE.
    This metric aims to quantify the separation quality when a particular input (Gate) is active.
    
    For a given `gate_type_to_analyze`:
    1. Identify the channel(s) that should be HIGH when this gate is active.
    2. Identify the channel(s) that should be LOW when this gate is active.
    3. Calculate ER: (Voltage of intended HIGH channel) / (Max Voltage of intended LOW channels).
    """
    input_ers = {}
    
    intended_high_channel_label = GATE_TO_HIGH_CHANNEL_MAP.get(gate_type_to_analyze)
    
    if not intended_high_channel_label:
        # print(f"Warning: No channel mapped as HIGH for input '{gate_type_to_analyze}'. Skipping Input ER calculation.")
        return {wl: np.nan for wl in wavelengths}

    intended_high_channel_num = None
    for ch_num, ch_info in all_channels_data.items():
        if ch_info['label'] == intended_high_channel_label:
            intended_high_channel_num = ch_num
            break
            
    if intended_high_channel_num is None:
        # print(f"Warning: Mapped channel label '{intended_high_channel_label}' not found in loaded data. Skipping Input ER calculation.")
        return {wl: np.nan for wl in wavelengths}

    df_high_channel = all_channels_data[intended_high_channel_num]['data']
    
    other_channels_dfs = {
        ch_num: ch_info['data'] 
        for ch_num, ch_info in all_channels_data.items() 
        if ch_num != intended_high_channel_num
    }

    for wavelength in wavelengths:
        wl_col = str(wavelength)
        
        # Get the voltage of the intended HIGH channel for this gate and wavelength
        high_voltage_at_gate = df_high_channel[df_high_channel['Gate'] == gate_type_to_analyze][wl_col].iloc[0] if wl_col in df_high_channel.columns and not df_high_channel[df_high_channel['Gate'] == gate_type_to_analyze].empty else np.nan
        
        # Get all voltages of the intended LOW channels for this gate and wavelength
        low_voltages_at_gate = []
        for ch_num, df_other in other_channels_dfs.items():
            if wl_col in df_other.columns and not df_other[df_other['Gate'] == gate_type_to_analyze].empty:
                val = df_other[df_other['Gate'] == gate_type_to_analyze][wl_col].iloc[0]
                if pd.notna(val):
                    low_voltages_at_gate.append(val)
        
        # Determine the WORST-CASE (highest) low voltage
        if len(low_voltages_at_gate) > 0:
            max_low_voltage_at_gate = np.max(low_voltages_at_gate)
        else:
            max_low_voltage_at_gate = np.nan

        # Calculate ER for this input
        if np.isnan(high_voltage_at_gate) or np.isnan(max_low_voltage_at_gate):
            er_db = np.nan
        elif max_low_voltage_at_gate == 0:
            er_db = 10 * np.log10(high_voltage_at_gate / 1e-9) if high_voltage_at_gate > 0 else np.nan
        elif high_voltage_at_gate <= 0:
            er_db = np.nan
        else:
            er_db = 10 * np.log10(high_voltage_at_gate / max_low_voltage_at_gate)
        
        input_ers[wavelength] = er_db
    
    return input_ers


def plot_decoder_analysis_innovative():
    """Main plotting function - innovative visualizations for comparison and performance."""
    
    data = load_latest_data()
    if not data:
        return
    
    first_channel_df = data[list(data.keys())[0]]['data']
    all_gate_types = sorted(first_channel_df['Gate'].unique())

    first_channel_num = list(data.keys())[0]
    df_sample = data[first_channel_num]['data']
    wavelength_columns = [col for col in df_sample.columns if col not in ['Gate', 'A', 'B']]
    wavelengths = sorted([float(col) for col in wavelength_columns])
    
    # --- Data Processing for Heatmaps and Plots ---
    
    # 1. Per-Channel ER Data (for individual channel plots and heatmap)
    channel_results = {}
    for channel_num, channel_info in data.items():
        er_data, high_data, low_data = calculate_channel_extinction_ratio_and_levels(
            channel_info['data'], wavelengths, channel_info['label']
        )
        channel_results[channel_num] = {
            'label': channel_info['label'],
            'extinction_ratios': er_data,
            'high_levels': high_data,
            'low_levels': low_data,
            'data': channel_info['data'] # Raw data for individual plots
        }
    
    # 2. Per-Input ER Data (for the new Input vs. Wavelength heatmap)
    input_er_df = pd.DataFrame(index=wavelengths)
    for gate_type in all_gate_types:
        input_er_data = calculate_input_extinction_ratio(data, wavelengths, gate_type)
        input_er_df[gate_type] = pd.Series(input_er_data)

    n_channels = len(channel_results) # This is 4 for your 4 channels.
    n_inputs = len(all_gate_types) # This is 4 for Gate_00, 01, 10, 11

    # --- Figure Setup ---
    # Total rows: 2 for heatmaps + ceil(n_inputs/2) for input-specific voltage plots
    # If 4 inputs, that's 2 rows for input-specific plots. So total 4 rows.
    # We maintain 2 columns.
    fig = plt.figure(figsize=(18, 14)) # Keep similar size, can adjust if plots are too squished
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.0, 1.0]) # 4 rows (2 for heatmaps, 2 for input plots)
    
    fig.suptitle('2-bit Decoder Performance Overview: Wavelength-Dependent Characterization', 
                 fontsize=18, fontweight='bold', y=0.98)

    # --- Colormap Definition (Define once for both heatmaps) ---
    vmin_er = -5.0 
    vmax_er = 5.0 
    norm_0db_pos = (-vmin_er / (vmax_er - vmin_er))
    colors = ["firebrick", "lightgray", "forestgreen"]
    nodes = [0.0, norm_0db_pos, 1.0] 
    cmap_er = mcolors.LinearSegmentedColormap.from_list("er_divergent_cmap", list(zip(nodes, colors)))

    # --- 1. Input Extinction Ratio Heatmap (Top Row, Spans Both Columns) ---
    ax_input_heatmap = fig.add_subplot(gs[0, :]) # First row, spans both columns
    
    im_input = ax_input_heatmap.imshow(input_er_df.T, cmap=cmap_er, aspect='auto', 
                                interpolation='nearest', vmin=vmin_er, vmax=vmax_er)
    
    ax_input_heatmap.set_yticks(np.arange(len(input_er_df.columns)))
    input_heatmap_yticklabels = [f"{gt} (High on {GATE_TO_HIGH_CHANNEL_MAP.get(gt, '?')})" for gt in input_er_df.columns]
    ax_input_heatmap.set_yticklabels(input_heatmap_yticklabels, fontsize=11)
    
    ax_input_heatmap.set_xticks(np.arange(len(input_er_df.index)))
    ax_input_heatmap.set_xticklabels([f'{int(wl)}' for wl in input_er_df.index], rotation=45, ha='right', fontsize=9)
    
    ax_input_heatmap.set_xlabel('Wavelength (nm)', fontsize=12)
    ax_input_heatmap.set_title('Input State Separation (ER in dB) Heatmap', fontsize=14)
    
    cbar_input = fig.colorbar(im_input, ax=ax_input_heatmap, orientation='vertical', pad=0.02)
    cbar_input.set_label('Extinction Ratio (dB)', fontsize=10)
    
    for i in range(len(input_er_df.columns)):
        for j in range(len(input_er_df.index)):
            er_val = input_er_df.iloc[j, i]
            if not np.isnan(er_val):
                normalized_val = (er_val - vmin_er) / (vmax_er - vmin_er)
                text_color = 'black' if (normalized_val > 0.4 and normalized_val < 0.8) or np.isclose(normalized_val, norm_0db_pos, atol=0.05) else 'white'
                ax_input_heatmap.text(j, i, f'{er_val:.1f}', ha='center', va='center', 
                                color=text_color, fontsize=8)


    # --- 2. Per-Channel Extinction Ratio Heatmap (Second Row, Spans Both Columns) ---
    ax_channel_heatmap = fig.add_subplot(gs[1, :]) # Second row, spans both columns
    
    channel_er_data_for_heatmap = pd.DataFrame(index=wavelengths)
    for channel_num, result in channel_results.items():
        channel_er_data_for_heatmap[f"Ch{channel_num} ({result['label']})"] = pd.Series(result['extinction_ratios'])

    im_channel = ax_channel_heatmap.imshow(channel_er_data_for_heatmap.T, cmap=cmap_er, aspect='auto', 
                                  interpolation='nearest', vmin=vmin_er, vmax=vmax_er)
    
    ax_channel_heatmap.set_yticks(np.arange(len(channel_er_data_for_heatmap.columns)))
    ax_channel_heatmap.set_yticklabels(channel_er_data_for_heatmap.columns, fontsize=11)
    
    ax_channel_heatmap.set_xticks(np.arange(len(channel_er_data_for_heatmap.index)))
    ax_channel_heatmap.set_xticklabels([f'{int(wl)}' for wl in channel_er_data_for_heatmap.index], rotation=45, ha='right', fontsize=9)
    
    ax_channel_heatmap.set_xlabel('Wavelength (nm)', fontsize=12)
    ax_channel_heatmap.set_title('Channel-Specific Extinction Ratio (dB) Heatmap', fontsize=14)
    
    cbar_channel = fig.colorbar(im_channel, ax=ax_channel_heatmap, orientation='vertical', pad=0.02)
    cbar_channel.set_label('Extinction Ratio (dB)', fontsize=10)
    
    for i in range(len(channel_er_data_for_heatmap.columns)):
        for j in range(len(channel_er_data_for_heatmap.index)):
            er_val = channel_er_data_for_heatmap.iloc[j, i]
            if not np.isnan(er_val):
                normalized_val = (er_val - vmin_er) / (vmax_er - vmin_er)
                text_color = 'black' if (normalized_val > 0.4 and normalized_val < 0.8) or np.isclose(normalized_val, norm_0db_pos, atol=0.05) else 'white'
                ax_channel_heatmap.text(j, i, f'{er_val:.1f}', ha='center', va='center', 
                                color=text_color, fontsize=8)


    # --- 3. Input-Specific Voltage Separation Plots (Bottom 2 Rows) ---
    # Now, these plots will show voltage levels for EACH INPUT (not each channel)
    subplot_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), 
                    fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])]
    
    # Define a set of distinct colors for the individual channels
    # These are distinct from the primary channel_colors used above, 
    # as we're now plotting ALL channels on ONE graph for each input
    output_channel_plotting_colors = {
        'Ch1': 'blue',
        'Ch2': 'orange',
        'Ch3': 'green',
        'Ch4': 'purple'
    }

    # Iterate through each GATE type (input combination)
    for i, gate_type in enumerate(all_gate_types):
        if i >= len(subplot_axes):
            break
            
        ax = subplot_axes[i]
        
        # Determine which channel should be HIGH for this specific gate_type
        intended_high_channel_label = GATE_TO_HIGH_CHANNEL_MAP.get(gate_type)
        intended_high_channel_num = None
        for ch_num, ch_info in data.items():
            if ch_info['label'] == intended_high_channel_label:
                intended_high_channel_num = ch_num
                break

        # Plot voltages for all channels when this 'gate_type' input is active
        for ch_num, ch_info in data.items():
            df_channel = ch_info['data']
            
            voltages = []
            wl_plot = []
            
            # Find the row corresponding to the current gate_type
            gate_data_row = df_channel[df_channel['Gate'] == gate_type]
            
            if not gate_data_row.empty:
                for wl in wavelengths:
                    wl_col = str(wl)
                    if wl_col in gate_data_row.columns and pd.notna(gate_data_row.iloc[0][wl_col]):
                        voltages.append(gate_data_row.iloc[0][wl_col])
                        wl_plot.append(wl)
            
            if voltages:
                label = f"Ch{ch_num}: {ch_info['label']}"
                line_color = output_channel_plotting_colors.get(f'Ch{ch_num}', 'black')
                line_style = '-' # Solid line for all
                line_width = 2.0 if ch_num == intended_high_channel_num else 1.0 # Thicker for intended HIGH
                marker = 'o' if ch_num == intended_high_channel_num else '' # Marker only for intended HIGH
                alpha_val = 1.0 if ch_num == intended_high_channel_num else 0.6 # Stronger for intended HIGH
                
                ax.plot(wl_plot, voltages, line_style, linewidth=line_width, markersize=5, 
                       color=line_color, alpha=alpha_val, marker=marker,
                       label=label + (" (Intended HIGH)" if ch_num == intended_high_channel_num else ""))
        
        ax.set_ylabel('Output Voltage (V)', fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=10)
        ax.tick_params(axis='x', labelsize=9)
        # Title reflects the input being analyzed
        ax.set_title(f"Input: {gate_type.replace('Gate_', '')} - All Output Voltages", fontsize=12)
        ax.legend(loc='upper right', fontsize=8, ncol=1) # One column for legend for clarity

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- Print Summary Statistics ---
    print(f"\n{'='*60}")
    print(f"DECODER ANALYSIS SUMMARY")
    print(f"{'='*60}")
    
    # Channel-specific ER summary
    print("\n--- Channel-Specific Extinction Ratios ---")
    for channel_num, result in channel_results.items():
        er_data = result['extinction_ratios']
        if er_data:
            er_values = [v for v in er_data.values() if not np.isnan(v)]
            if er_values:
                print(f"\nChannel {channel_num} ({result['label']}):")
                print(f"  Extinction Ratio (Mean ± Std): {np.mean(er_values):.2f} ± {np.std(er_values):.2f} dB")
                print(f"  Extinction Ratio Range: {np.min(er_values):.2f} to {np.max(er_values):.2f} dB")
                
                best_er_val = -np.inf
                best_wl = np.nan
                for wl, er in er_data.items():
                    if not np.isnan(er) and er > best_er_val:
                        best_er_val = er
                        best_wl = wl

                if best_er_val != -np.inf:
                    print(f"  Best Extinction Ratio at: {best_wl:.1f} nm ({best_er_val:.2f} dB)")
                else:
                    print(f"  No valid (non-NaN) Extinction Ratio found.")
            else:
                print(f"\nChannel {channel_num} ({result['label']}): No valid Extinction Ratio data.")
    
    # Input-specific ER summary
    print("\n--- Input-Specific Extinction Ratios ---")
    for gate_type in all_gate_types:
        input_er_values = [v for v in input_er_df[gate_type].values if not np.isnan(v)]
        if input_er_values:
            print(f"\nInput {gate_type} (High on {GATE_TO_HIGH_CHANNEL_MAP.get(gate_type, '?')}):")
            print(f"  Extinction Ratio (Mean ± Std): {np.mean(input_er_values):.2f} ± {np.std(input_er_values):.2f} dB")
            print(f"  Extinction Ratio Range: {np.min(input_er_values):.2f} to {np.max(input_er_values):.2f} dB")
            
            best_er_val = -np.inf
            best_wl = np.nan
            input_er_data = input_er_df[gate_type].to_dict() # Convert Series back to dict for easy iteration
            for wl, er in input_er_data.items():
                if not np.isnan(er) and er > best_er_val:
                    best_er_val = er
                    best_wl = wl
            
            if best_er_val != -np.inf:
                print(f"  Best Extinction Ratio at: {best_wl:.1f} nm ({best_er_val:.2f} dB)")
            else:
                print(f"  No valid (non-NaN) Extinction Ratio found.")
        else:
            print(f"\nInput {gate_type}: No valid Extinction Ratio data.")


    plt.show()

# Simple usage
if __name__ == "__main__":
    plot_decoder_analysis_innovative()