import matplotlib
matplotlib.use('TkAgg')  # For better font rendering

import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame with the input-output combinations
data = pd.DataFrame({
    'Input A (V)': [0.10, 0.10, 4.90, 4.90],
    'Input B (V)': [0.10, 4.90, 0.10, 4.90],
    'Output': [0.9803, 1.08, 0.79, 4.29]
})

# Create labels for each combination with A on top of B
data['Label'] = [
    'A=0.10V\nB=0.10V',
    'A=0.10V\nB=4.90V',
    'A=4.90V\nB=0.10V',
    'A=4.90V\nB=4.90V'
]

# Plot bar chart
plt.figure(figsize=(6, 5), dpi= 180)
plt.bar(data['Label'], data['Output'], color='skyblue')
plt.title('AND Gate', fontsize = 16)
plt.xlabel('Input Combinations',fontsize = 16)
plt.ylabel('Output (V)',fontsize = 16)
plt.xticks(rotation=0, fontsize = 14)
plt.tight_layout()
plt.savefig("and2.png", dpi=300, bbox_inches='tight')

plt.show()


