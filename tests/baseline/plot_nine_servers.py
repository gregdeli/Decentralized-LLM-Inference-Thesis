import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_worker_results(csv_path="worker_results.csv"):
    """Generates stacked bar plots for worker delays per allocation strategy."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    delay_columns = [
        'Avg Deserialization Delay',
        'Avg Inference Delay',
        'Avg Logit Sampling Delay',
        'Avg Serialization Delay',
        'Avg Transmission Delay'
    ]
    
    # Filter out columns that might not exist or ensure they are properly formatted
    existing_delay_columns = [col for col in delay_columns if col in df.columns]

    # for allocation in df['Allocation'].unique():
    # subset = df[df['Allocation'] == allocation].copy()
    # if subset.empty:
    #     continue
        
    # Create a combined label for the x-axis (e.g., "worker-C\n(0, 3)")
    df['X_Label'] = df['Worker'] + "\n" + df['Layers']
    
    # Set the new label as the index to make DataFrame.plot use it for the x-axis
    df.set_index('X_Label', inplace=True)
    
    # Create the stacked bar plot
    ax = df[existing_delay_columns].plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6),
        colormap='viridis'
    )
    
    plt.title(f"Worker Delays")
    plt.xlabel("Worker and Assigned Layers")
    plt.ylabel("Delay (ms)")
    
    # Move legend outside the plot for better visibility
    plt.legend(title="Delay Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"tests/baseline/nine_worker_delays.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_worker_results("tests/baseline/nine_worker_results.csv")