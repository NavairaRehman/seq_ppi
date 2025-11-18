#!/usr/bin/env python3
import sys
sys.path.append('/media/rapidsai/Data1/Navaira/seq_ppi/embeddings')


import numpy as np
import os
from tensorflow.keras.models import load_model
from seq2tensor import s2t
import scipy.stats
from tqdm import tqdm
import csv

"""
PPI Binding Affinity Inference Script:
- Loads the trained model, prepares the input sequence data, and generates predictions for protein pairs.
- Outputs the predicted binding affinity (and other metrics like MAE, MSE, and correlation) to a CSV file.
"""

def usage():
    print("Usage: inference_pipr_regression.py seq_tsv pairs_tsv model_checkpoint output_csv [threshold=0.01]")
    sys.exit(1)

def key(a, b):
    """Canonicalize protein pair key"""
    return tuple(sorted((a.strip(), b.strip())))

if __name__ == "__main__":
    if len(sys.argv) < 4:
        usage()

    seq_tsv_path = sys.argv[1]
    pairs_tsv_path = sys.argv[2]
    model_checkpoint_path = sys.argv[3]
    output_csv_path = sys.argv[4]
    threshold = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.01

    # Load sequences from the sequence file (same as training)
    print("Loading sequences...")
    id2seq = {}
    seqs = []
    with open(seq_tsv_path) as f:
        for line in f:
            line = line.strip().split('\t')
            id2seq[line[0]] = line[1]
            seqs.append(line[1])

    # Prepare protein pairs from the pairs file
    print("Loading protein pairs...")
    protein_pairs = []
    with open(pairs_tsv_path) as f:
        for line in f:
            line = line.strip().split('\t')
            protA, protB = line[0], line[1]
            protein_pairs.append((protA, protB))

    # Prepare embeddings (same as training)
    emb_files = ['../../../embeddings/string_vec5.txt', 
                 '../../../embeddings/CTCoding_onehot.txt', '../../../embeddings/vec5_CTC.txt','../../../embeddings/vec7_CTC.txt']
    use_emb = 2  # The embedding index used during training
    seq2t = s2t(emb_files[use_emb])

    seq_size = 2000  # Same seq_size as in training
    seq_tensor = np.array([seq2t.embed_normalized(seq, seq_size) for seq in tqdm(seqs)])

    # Load the model
    print("Loading model from checkpoint...")
    model = load_model(model_checkpoint_path)

    # Assuming you've already calculated these values during the inference run or from a previous run
    all_min = -21.4053
    all_max = -1.3635

    # Prepare output CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["protA", "protB", "predicted_affinity", "scaled_predicted_affinity"])

        total_mae, total_mse, num_total = 0, 0, 0

        # Lists to store all predicted values and the true affinity values for error metrics
        all_predictions = []
        all_true_values = []

        for protA, protB in tqdm(protein_pairs):
            seq1 = id2seq.get(protA, "X")
            seq2 = id2seq.get(protB, "X")
            seq1_embedded = seq2t.embed_normalized(seq1, seq_size)
            seq2_embedded = seq2t.embed_normalized(seq2, seq_size)

            # Predict the affinity score (regression output)
            pred = model.predict([seq1_embedded[np.newaxis], seq2_embedded[np.newaxis]])[0][0]

            # Scale back the result (same as in training)
            scaled_pred = pred * (all_max - all_min) + all_min

            # # Here, you need the true affinity values for this protein pair (e.g., from raw_data)
            # true_affinity = float(raw_data[raw_ids.index((protA, protB))][label_index])  # Assuming you have the true affinity in raw_data

            # # Compute the error metrics (MAE and MSE)
            # diff = abs(true_affinity - scaled_pred)
            # total_mae += diff
            # total_mse += diff ** 2
            # num_total += 1

            # Write the result to CSV
            writer.writerow([protA, protB, f"{scaled_pred:.6f}", f"{pred:.6f}"])

        # # Calculate and print overall metrics
        # mse = total_mse / num_total
        # mae = total_mae / num_total
        # print(f"MSE: {mse}, MAE: {mae}")

    print(f"Predictions saved to {output_csv_path}")
