#!/usr/bin/env python3
import sys
import numpy as np

def usage():
    print("Usage: calculate_min_max.py <ds_file> <label_index> [use_log=0]")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        usage()

    # Read command line arguments
    ds_file = sys.argv[1]  # Path to your training data file (TSV)
    label_index = int(sys.argv[2])  # Column index of the label (affinity score)
    use_log = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # Whether to use log transformation (default: 0)

    # Initialize variables to compute all_min and all_max
    all_min, all_max = 99999999, -99999999
    score_labels = []

    # Read the data and compute the min and max
    with open(ds_file, 'r') as f:
        skip_head = True
        for line in f:
            if skip_head:
                skip_head = False  # Skip the header
                continue

            line = line.strip().split('\t')
            score = line[label_index]  # Get the affinity score from the specified column
            try:
                # If using log, apply the transformation
                if use_log:
                    score = np.log(float(score))
                else:
                    score = float(score)

                score_labels.append(score)

                # Update the min and max
                if score < all_min:
                    all_min = score
                if score > all_max:
                    all_max = score
            except ValueError:
                continue  # Skip malformed rows

    # Output the min and max values
    print(f"Calculated all_min: {all_min}, all_max: {all_max}")
