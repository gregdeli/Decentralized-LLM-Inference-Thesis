import pandas as pd
import matplotlib.pyplot as plt
import ast
import os

def generate_plot():
    file_path = 'tests/multi_client/results.csv'
    
    # Load the results file
    df = pd.read_csv(file_path)
    
    # Convert string representations of lists into Python lists
    df['Tokens_Generated'] = df['Tokens_Generated'].apply(ast.literal_eval)
    df['Timestamps'] = df['Timestamps'].apply(ast.literal_eval)
    
    plt.figure(figsize=(10, 6))
    
    # Plot data for each benchmark run
    for index, row in df.iterrows():
        num_clients = row['Num Clients']
        tokens = row['Tokens_Generated']
        timestamps = row['Timestamps']
        
        plt.plot(timestamps, tokens,  label=f'{num_clients} Client')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Token Count')
    plt.title('Token Count vs Time by Number of Clients')
    plt.legend(title='Concurrent Clients')
    plt.grid(True, linestyle='--')
    
    output_path = 'tests/multi_client/tokens_vs_time.png'
    
    # Save the plot
    plt.savefig(output_path)
    print(f"Plot successfully saved to {output_path}")

if __name__ == "__main__":
    generate_plot()