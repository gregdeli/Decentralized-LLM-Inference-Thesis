import pandas as pd
import matplotlib.pyplot as plt
import os
import ast

# Define the path to the results data
file_path = 'tests/quantization/throughput_jitter.csv'
output_dir = 'tests/quantization/'

# Load the CSV file
df = pd.read_csv(file_path)

model_str = df['Model'].iloc[0]

df["Tokens_Generated"] = df["Tokens_Generated"].apply(ast.literal_eval)
df["Timestamps"] = df["Timestamps"].apply(ast.literal_eval)

plt.figure(figsize=(10, 6))

no_quant_df = df[df['Quantize'] == 0]
quant_df = df[df['Quantize'] == 1]

# Plot non-quantized points (Blue)
plt.plot(
    no_quant_df.iloc[0]['Timestamps'], 
    no_quant_df.iloc[0]['Tokens_Generated'], 
    marker='o', 
    markersize=1,
    color='blue', 
    label='No Quantization'
)

# Plot quantized points (Orange)
plt.plot(
    quant_df.iloc[0]['Timestamps'], 
    quant_df.iloc[0]['Tokens_Generated'], 
    marker='o', 
    markersize=1,
    color='orange', 
    label='Quantized'
)

# Format the plot
plt.title(f"Throughput: Token Count vs. Time\nModel: {model_str.split('/')[-1]}")
plt.xlabel('Time (s)')
plt.ylabel('Token Count')

# Apply logarithmic scale to the X-axis for clearer point spacing
# plt.xscale('log')
# Set explicit tick markers to match the dataset
# plt.xticks([1, 2, 3, 10, 20, 30, 100, 1000], ['1', '2', '3', '10', '20', '30', '100', '1000'])

# Apply uniform Y-axis scale (0 to global maximum)
# plt.ylim(0, global_y_max)

plt.grid(True)
plt.legend()

# Automatically adjust layout and display the plot
plt.tight_layout()

# Generate a safe filename by extracting the model name after the slash
safe_model_name = model_str.split('/')[-1]
save_path = os.path.join(output_dir, f'{safe_model_name}_throughput_band_drop.png')

# Save the figure and close the plot
plt.savefig(save_path, format='png', dpi=300)
plt.close()