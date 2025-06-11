# import matplotlib.pyplot as plt
# import pandas as pd

# # Updated data
# # data_corrected = {
# #     "Input_A_New": [0.1, 0.1, 4.9, 4.9],
# #     "Input_B_New": [0.1, 4.9, 0.1, 4.9],
# #     "Output1": [4.623, 1.986, 3.38, 1.692],
# #     "Output2": [1.598, 5.625, 1.273, 3.613],
# #     "Output3": [3.513, 3.553, 4.173, 3.368],
# #     "Output4": [3.402, 4.911, 3.583, 5.727],
# # }

# data_corrected = {
#     "Input_A_New": [0.1, 0.1, 4.9, 4.9],
#     "Input_B_New": [0.1, 4.9, 0.1, 4.9],
#     "Output1": [4.995, 3.302, 1.733, 1.71],
#     "Output2": [2.462, 4.354, 2.859, 4.468],
#     "Output3": [2.897, 3.603, 6.3, 4.033],
#     "Output4": [2.132, 3.479, 2.371, 6.01]
# }


# # Create DataFrame
# df_corrected = pd.DataFrame(data_corrected)

# # Define channel colors (based on the oscilloscope screenshot)
# channel_colors = ["#FFD700", "#00FFFF", "#FF00FF", "#00FF00"]  # Yellow, Cyan, Magenta, Green

# # Function to adapt font size and layout dynamically
# def dynamic_plot(fig_width, fig_height):
#     fig, axs = plt.subplots(2, 2, figsize=(fig_width, fig_height))  # Dynamic figure size
#     titles = [
#         f"Input A={a}, B={b}" for a, b in zip(df_corrected["Input_A_New"], df_corrected["Input_B_New"])
#     ]

#     # Adjust font size based on figure dimensions
#     base_fontsize = max(18, min(fig_width, fig_height) * 1.5)

#     for i, ax in enumerate(axs.flatten()):
#         outputs = [df_corrected.loc[i, "Output1"], df_corrected.loc[i, "Output2"], 
#                    df_corrected.loc[i, "Output3"], df_corrected.loc[i, "Output4"]]
#         ax.bar(["O1", "O2", "O3", "O4"], outputs, color=channel_colors, alpha=0.8)
#         ax.set_title(titles[i], fontsize=base_fontsize)
#         ax.set_ylabel("Output Value (V)", fontsize=base_fontsize * 0.8)
#         ax.set_xlabel("Outputs", fontsize=base_fontsize * 0.8)
#         ax.tick_params(axis='x', labelrotation=0, labelsize=base_fontsize * 0.7)
#         ax.tick_params(axis='y', labelsize=base_fontsize * 0.7)
#         ax.grid(axis="y", linestyle="--", alpha=0.7)

#     # Adjust layout dynamically
#     plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ensure labels and titles stay in position
#     #plt.suptitle("Output Values for Each Input Combination", fontsize=base_fontsize * 1.2)
    
#     plt.savefig('test.png',dpi=300)
#     plt.show()

# # Call the function with your desired figure size
# dynamic_plot(8, 6)  # Example small size


import matplotlib.pyplot as plt
import pandas as pd

# Updated data
data_corrected = {
    "Input_A_New": [0.1, 0.1, 4.9, 4.9],
    "Input_B_New": [0.1, 4.9, 0.1, 4.9],
    "Output1": [5.14, 1.51, 2.62, 1.5],
    "Output2": [2.33, 5.211, 3.04, 4.62],
    "Output3": [3.32, 2.46, 5.90, 3.66],
    "Output4": [3.89, 3.5, 2.14, 5.04]
}

# Create DataFrame
df_corrected = pd.DataFrame(data_corrected)

# Define channel colors
channel_colors = ["#FFD700", "#00FFFF", "#FF00FF", "#0080FF"]  # Yellow, Cyan, Magenta, Green

# Function to generate and save individual plots
def save_individual_plots(fig_width, fig_height):
  
    base_fontsize = max(25, min(fig_width, fig_height) * 1.5)

    for i in range(len(df_corrected)):
        outputs = [
            df_corrected.loc[i, "Output1"],
            df_corrected.loc[i, "Output2"],
            df_corrected.loc[i, "Output3"],
            df_corrected.loc[i, "Output4"]
        ]
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.bar(["0", "1", "2", "3"], outputs, color=channel_colors, alpha=0.8)
        #ax.set_title(titles[i], fontsize=base_fontsize)
        ax.set_ylabel("Voltage (V)", fontsize=base_fontsize)
        #ax.set_xlabel("Outputs", fontsize=base_fontsize)
        ax.tick_params(axis='x', labelrotation=0, labelsize=base_fontsize * 0.9)
        ax.tick_params(axis='y', labelsize=base_fontsize * 0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.savefig(f"result{i+1}.png", dpi=300)
        plt.close()  # Close the figure to save memory

# Call the function
save_individual_plots(5, 4)  # You can adjust the size as needed
