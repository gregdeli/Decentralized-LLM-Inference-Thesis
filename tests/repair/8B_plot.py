import pandas as pd
import matplotlib.pyplot as plt
import ast

def plot_throughput_timeline_to_png(csv_filepath, output_filename):
    # Load the data from the CSV file
    df = pd.read_csv(csv_filepath)

    # Initialize the plot
    plt.figure(figsize=(10, 6))


    tokens = ast.literal_eval(df["Tokens_Generated"].iloc[0])
    timestamps = ast.literal_eval(df["Timestamps"].iloc[0])

    plt.plot(timestamps, tokens, linewidth=2)

    # Format the plot
    plt.title(f"Token Count vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Token Count")
    # plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as a high-resolution PNG file
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot successfully saved to {output_filename}")

if __name__ == "__main__":
    # Specify your input CSV and desired output PNG filename
    plot_throughput_timeline_to_png('tests/repair/8B_results.csv', 'tests/repair/8B_tokens_vs_time.png')