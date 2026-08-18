import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the path to the results data
file_path = 'tests/quantization/big_results.csv'
output_dir = 'tests/quantization/'

# Load the CSV file
df = pd.read_csv(file_path)

# Clean the 'Bandwidth' column (e.g., convert '1mbit' to 1) and cast to integer
# df['Bandwidth'] = df['Bandwidth'].str.replace('mbit', '').astype(int)

# Convert latency from seconds to milliseconds
df['Total_Decoding_Avg_Network_Latency_ms'] = df['Total_Decoding_Avg_Network_Latency'] * 1000

# Calculate the global maximum latency to ensure a uniform Y-axis scale across all plots.
# Adding a 10% buffer above the maximum value for visual clarity.
global_y_max = df['Total_Decoding_Avg_Network_Latency_ms'].max() * 1.1

models = df['Model'].unique()

for model in models:
    plt.figure(figsize=(10, 6))
    
    # Isolate data for the current model
    model_df = df[df['Model'] == model]

    no_quant_df = model_df[model_df['Quantize'] == 0].sort_values(by='Bandwidth')
    quant_df = model_df[model_df['Quantize'] == 1].sort_values(by='Bandwidth')

    # Plot non-quantized points (Blue)
    plt.plot(
        no_quant_df['Bandwidth'], 
        no_quant_df['Total_Decoding_Avg_Network_Latency_ms'], 
        marker='o', 
        color='blue', 
        label='No Quantization'
    )
    
    # Plot quantized points (Orange)
    plt.plot(
        quant_df['Bandwidth'], 
        quant_df['Total_Decoding_Avg_Network_Latency_ms'], 
        marker='o', 
        color='orange', 
        label='Quantized'
    )

    # Format the plot
    plt.title(f"Decoding Stage Network Latency vs Bandwidth\nModel: {model.split('/')[-1]}")
    plt.xlabel('Bandwidth (Mbps)')
    plt.ylabel('Decoding Stage Avg Network Latency (ms)')
    
    # Apply logarithmic scale to the X-axis for clearer point spacing
    plt.xscale('log')
    # Set explicit tick markers to match the dataset
    plt.xticks([1, 2, 3, 10, 20, 30, 100, 1000], ['1', '2', '3', '10', '20', '30', '100', '1000'])

    # Apply uniform Y-axis scale (0 to global maximum)
    plt.ylim(0, global_y_max)
    
    # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    # Automatically adjust layout and display the plot
    plt.tight_layout()

    # Generate a safe filename by extracting the model name after the slash
    safe_model_name = model.split('/')[-1]
    save_path = os.path.join(output_dir, f'{safe_model_name}_net_latency_plot.png')
    
    # Save the figure and close the plot
    plt.savefig(save_path, format='png', dpi=300)
    plt.close()