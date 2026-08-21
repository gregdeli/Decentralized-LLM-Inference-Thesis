import pandas as pd
import matplotlib.pyplot as plt
import ast

def plot_takeover_metrics_to_png(csv_filepath, output_filename):
    # Load the data from the CSV file
    df = pd.read_csv(csv_filepath)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate through the DataFrame rows to plot each configuration
    for index, row in df.iterrows():
        takeover_enabled = bool(row['TAKEOVER Enabled'])
        
        # Parse the string representations of lists back into actual lists
        tokens = ast.literal_eval(row['Tokens_Generated'])
        tokens = tokens[:250]
        timestamps = ast.literal_eval(row['Timestamps'])
        timestamps = timestamps[:250]
        
        # Set the line label based on the takeover flag
        label = "Takeover Enabled" if takeover_enabled else "Takeover Disabled"
        
        # Plot the data
        plt.plot(timestamps, tokens, label=label, linewidth=2)

    # Format the plot
    plt.title(f"Token Count vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Token Count")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as a high-resolution PNG file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    # Specify your input CSV and desired output PNG filename
    plot_takeover_metrics_to_png('tests/takeover/results.csv', 'tests/takeover/tokens_vs_time.png')