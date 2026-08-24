import pandas as pd
import matplotlib.pyplot as plt
import ast

def plot_throughput_timeline_to_png(csv_filepath, output_filename):
    # Load the data from the CSV file
    df = pd.read_csv(csv_filepath)

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    for index, row in df.iterrows():
        backup_repair = bool(row["Backup_Repair"])
    
        tokens = ast.literal_eval(row["Tokens_Generated"])
        timestamps = ast.literal_eval(row["Timestamps"])

        label = "Repair with Backup" if backup_repair else "Repair with Predecessor"

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
    plot_throughput_timeline_to_png('tests/repair/results.csv', 'tests/repair/tokens_vs_time.png')