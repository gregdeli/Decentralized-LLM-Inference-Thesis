import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def plot_client_results(csv_path="client_results.csv"):
    """Generates a line plot for Token Count vs. Timestamp for each allocation strategy."""
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Iterate through unique allocations to plot a line for each
    for allocation in df['Allocation'].unique():
        subset = df[df['Allocation'] == allocation]
        if not subset.empty:
            # Parse the string representations of lists back into Python lists
            timestamps = ast.literal_eval(subset.iloc[0]['Timestamps'])
            tokens = ast.literal_eval(subset.iloc[0]['Tokens_Generated'])
            
            plt.plot(timestamps, tokens, label=allocation.title(), marker='o', markersize=2)

    plt.title("Token Count vs. Time per Allocation Strategy")
    plt.xlabel("Times (s)")
    plt.ylabel("Token Count")
    plt.legend(title="Allocation Strategy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tests/baseline/tokens_vs_time.png")
    plt.close()

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

    for allocation in df['Allocation'].unique():
        subset = df[df['Allocation'] == allocation].copy()
        if subset.empty:
            continue
            
        # Create a combined label for the x-axis (e.g., "worker-C\n(0, 3)")
        subset['X_Label'] = subset['Worker'] + "\n" + subset['Layers']
        
        # Set the new label as the index to make DataFrame.plot use it for the x-axis
        subset.set_index('X_Label', inplace=True)
        
        # Create the stacked bar plot
        ax = subset[existing_delay_columns].plot(
            kind='bar', 
            stacked=True, 
            figsize=(10, 6),
            colormap='viridis'
        )
        
        plt.title(f"Worker Delays - {allocation.title()} Allocation")
        plt.xlabel("Worker and Assigned Layers")
        plt.ylabel("Delay (s)")
        
        # Move legend outside the plot for better visibility
        plt.legend(title="Delay Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"tests/baseline/worker_delays_{allocation}.png")
        plt.close()

if __name__ == "__main__":
    plot_client_results("tests/baseline/client_results.csv")
    plot_worker_results("tests/baseline/worker_results.csv")