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

# Truth table mapping
DECODER_TRUTH_MAP = {
    'Output_00': 'Gate_00', 
    'Output_01': 'Gate_01', 
    'Output_10': 'Gate_10', 
    'Output_11': 'Gate_11', 
    'Output_1': 'Gate_00',
    'Output_2': 'Gate_01',
    'Output_3': 'Gate_10',
    'Output_4': 'Gate_11'
}

GATE_TO_HIGH_CHANNEL_MAP = {v: k for k, v in DECODER_TRUTH_MAP.items()}

def calculate_input_extinction_ratio(all_channels_data, wavelengths, gate_type_to_analyze):
    """Calculate Input Extinction Ratio for a specific gate type"""
    input_ers = {}
    
    intended_high_channel_label = GATE_TO_HIGH_CHANNEL_MAP.get(gate_type_to_analyze)
    
    if not intended_high_channel_label:
        return {wl: np.nan for wl in wavelengths}

    intended_high_channel_num = None
    for ch_num, ch_info in all_channels_data.items():
        if ch_info['label'] == intended_high_channel_label:
            intended_high_channel_num = ch_num
            break
            
    if intended_high_channel_num is None:
        return {wl: np.nan for wl in wavelengths}

    df_high_channel = all_channels_data[intended_high_channel_num]['data']
    
    other_channels_dfs = {
        ch_num: ch_info['data'] 
        for ch_num, ch_info in all_channels_data.items() 
        if ch_num != intended_high_channel_num
    }

    for wavelength in wavelengths:
        wl_col = str(wavelength)
        
        # Get voltage of intended HIGH channel
        high_voltage_at_gate = df_high_channel[df_high_channel['Gate'] == gate_type_to_analyze][wl_col].iloc[0] if wl_col in df_high_channel.columns and not df_high_channel[df_high_channel['Gate'] == gate_type_to_analyze].empty else np.nan
        
        # Get voltages of intended LOW channels
        low_voltages_at_gate = []
        for ch_num, df_other in other_channels_dfs.items():
            if wl_col in df_other.columns and not df_other[df_other['Gate'] == gate_type_to_analyze].empty:
                val = df_other[df_other['Gate'] == gate_type_to_analyze][wl_col].iloc[0]
                if pd.notna(val):
                    low_voltages_at_gate.append(val)
        
        # Get worst-case (highest) low voltage
        if len(low_voltages_at_gate) > 0:
            max_low_voltage_at_gate = np.max(low_voltages_at_gate)
        else:
            max_low_voltage_at_gate = np.nan

        # Calculate ER
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

def plot_heatmap_only():
    """Plot only the heatmap"""
    
    data = load_latest_data()
    if not data:
        return
    
    # Get gate types and wavelengths
    first_channel_df = data[list(data.keys())[0]]['data']
    all_gate_types = sorted(first_channel_df['Gate'].unique())

    first_channel_num = list(data.keys())[0]
    df_sample = data[first_channel_num]['data']
    wavelength_columns = [col for col in df_sample.columns if col not in ['Gate', 'A', 'B']]
    wavelengths = sorted([float(col) for col in wavelength_columns])
    
    # Calculate Input ER data
    input_er_df = pd.DataFrame(index=wavelengths)
    for gate_type in all_gate_types:
        input_er_data = calculate_input_extinction_ratio(data, wavelengths, gate_type)
        input_er_df[gate_type] = pd.Series(input_er_data)

    # Set publication-quality style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

    # Create figure with better proportions
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Enhanced colormap definition
    vmin_er = -5.0 
    vmax_er = 5.0 
    norm_0db_pos = (-vmin_er / (vmax_er - vmin_er))
    colors = ["#8B0000", "#F5F5F5", "#006400"]  # Dark red, light gray, dark green
    nodes = [0.0, norm_0db_pos, 1.0] 
    cmap_er = mcolors.LinearSegmentedColormap.from_list("er_divergent_cmap", list(zip(nodes, colors)))

    # Create heatmap with enhanced styling
    im = ax.imshow(input_er_df.T, cmap=cmap_er, aspect='auto', 
                   interpolation='nearest', vmin=vmin_er, vmax=vmax_er)
    
    # Enhanced grid
    ax.set_xticks(np.arange(len(input_er_df.index)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(input_er_df.columns)) + 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2, alpha=0.8)
    
    # Set labels with better formatting
    ax.set_yticks(np.arange(len(input_er_df.columns)))
    # Clean up gate labels for publication
    clean_labels = [gt.replace('Gate_', '') for gt in input_er_df.columns]
    ax.set_yticklabels(clean_labels, fontsize=13, fontweight='bold')
    
    # Set all wavelengths on x-axis
    ax.set_xticks(np.arange(len(input_er_df.index)))
    ax.set_xticklabels([f'{wl:.1f}' for wl in input_er_df.index], 
                      rotation=45, ha='right', fontsize=11)
    
    # Professional labels
    ax.set_xlabel('Wavelength (nm)', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Logic Input State', fontsize=15, fontweight='bold', labelpad=10)
    #ax.set_title('Wavelength-Dependent Extinction Ratio Performance\nof 2-bit Optical Decoder', 
    #            fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    cbar.set_label('Extinction Ratio (dB)', fontsize=14, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Add professional text annotations
    for i in range(len(input_er_df.columns)):
        for j in range(len(input_er_df.index)):
            er_val = input_er_df.iloc[j, i]
            if not np.isnan(er_val):
                normalized_val = (er_val - vmin_er) / (vmax_er - vmin_er)
                # Better text color logic
                if normalized_val < 0.3 or normalized_val > 0.7:
                    text_color = 'white'
                else:
                    text_color = 'black'
                
                ax.text(j, i, f'{er_val:.1f}', ha='center', va='center', 
                       color=text_color, fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='none', 
                               edgecolor='none', alpha=0.8))

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    plt.tight_layout()
    
    # Save option for publication
    # plt.savefig('decoder_heatmap.pdf', format='pdf', bbox_inches='tight', dpi=300)
    # plt.savefig('decoder_heatmap.png', format='png', bbox_inches='tight', dpi=300)
    
    plt.show()

# Run the function
if __name__ == "__main__":
    plot_heatmap_only()