import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_throughput():
    input_file = 'tests/multi_client/throughput_results.csv'
    output_file = 'tests/multi_client/throughput_vs_clients.png'
    
    # Verify the input file exists
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found in the current directory.")
        return

    # Load the CSV data
    df = pd.read_csv(input_file)
    
    # Sort the dataframe by 'Num Clients' to ensure a continuous line
    df = df.sort_values(by='Num Clients')

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(df['Num Clients'], df['Throughput'], marker='o', linestyle='-', color='b', linewidth=2)
    
    # Apply labels, title, and formatting
    plt.title('Throughput vs Number of Concurrent Clients')
    plt.xlabel('Number of Concurrent Clients')
    plt.ylabel('System Throughput (tokens/s)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to match the discrete client counts
    plt.xticks(df['Num Clients'])
    # plt.yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22])
    plt.yticks([0, 4, 8, 12, 16, 20, 24])
    
    # Adjust layout to prevent label clipping
    plt.tight_layout()
    
    # Save and close the plot
    plt.savefig(output_file)
    plt.close()
    
    print(f"Plot successfully generated and saved to '{output_file}'.")

if __name__ == "__main__":
    plot_throughput()