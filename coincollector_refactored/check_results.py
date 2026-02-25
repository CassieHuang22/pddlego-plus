import pandas as pd
import sys

# Usage: python count_successes.py results.csv
csv_path = sys.argv[1]

columns = [
    "date",
    "model",
    "method",
    "seed",
    "success",
    "runtime_seconds",
    "last_step_idx",
    "last_loop_idx",
    "trial_record",
]

# Load CSV without header
df = pd.read_csv(csv_path, header=None, names=columns)

# Ensure success column is numeric / boolean
df["success"] = df["success"].astype(int)

# Count successes per method
success_counts = (
    df[df["success"] == 1]
    .groupby("method")
    .size()
    .sort_values(ascending=False)
)

print(success_counts)
