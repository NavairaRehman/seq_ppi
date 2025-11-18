#!/usr/bin/env python3
# inference_pipr.py  (Py3.6 / TF1.x compatible)

import os, sys, csv, glob, argparse
import numpy as np

# ---- TensorFlow / Keras (TF1.x style) ----
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

from tensorflow.keras import models

# ---- embeddings ----
if '/media/rapidsai/Data3/Navaira/seq_ppi/embeddings' not in sys.path:
    sys.path.append('/media/rapidsai/Data3/Navaira/seq_ppi/embeddings')
from seq2tensor import s2t

# ---- typing for Py3.6 ----
from typing import Dict, List, Tuple, Optional, Iterable

def load_id2seq(seq_tsv):  # type: (str) -> Dict[str, str]
    """Expect a 2-col TSV: <protein_id>\t<sequence>"""
    id2seq = {}
    with open(seq_tsv, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            pid, seq = parts[0], parts[1]
            id2seq[pid] = seq
    return id2seq

def read_pairs(pairs_tsv, sid1_col, sid2_col, label_col, skip_header):
    # type: (str, int, int, Optional[int], bool) -> List[Tuple[str, str, Optional[str]]]
    rows = []
    with open(pairs_tsv, 'r', encoding='utf-8') as f:
        first = True
        for line in f:
            if first and skip_header:
                first = False
                continue
            line = line.rstrip('\n').rstrip('\r')
            if not line:
                continue
            parts = line.split('\t')
            try:
                p1 = parts[sid1_col]
                p2 = parts[sid2_col]
            except Exception:
                continue
            lab = None
            if (label_col is not None) and (label_col < len(parts)):
                lab = parts[label_col]
            rows.append((p1, p2, lab))
    return rows

def embed_batch(seq_list, seq2t_obj, seq_size):
    # type: (List[str], s2t, int) -> np.ndarray
    return np.stack([seq2t_obj.embed_normalized(s, seq_size) for s in seq_list], axis=0)

def batched(iterable, batch_size):
    # type: (Iterable, int) -> Iterable[List]
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    parser = argparse.ArgumentParser(description="PIPR binary inference → CSV")
    parser.add_argument("--seq_tsv", required=True)
    parser.add_argument("--pairs_tsv", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--emb_files", nargs="+", default=[
        "../../../embeddings/default_onehot.txt",
        "../../../embeddings/string_vec5.txt",
        "../../../embeddings/CTCoding_onehot.txt",
        "../../../embeddings/vec7_CTC.txt",
    ])
    parser.add_argument("--use_emb", type=int, default=0)
    parser.add_argument("--seq_size", type=int, default=600)
    parser.add_argument("--positive_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--sid1_col", type=int, default=0)
    parser.add_argument("--sid2_col", type=int, default=1)
    parser.add_argument("--label_col", type=int, default=-1)
    parser.add_argument("--skip_header", action="store_true")
    args = parser.parse_args()

    # expand checkpoint globs
    ckpts = []
    for pattern in args.checkpoints:
        matched = glob.glob(pattern)
        if matched:
            ckpts.extend(matched)
        elif os.path.isfile(pattern):
            ckpts.append(pattern)
    ckpts = sorted(set(ckpts))
    if not ckpts:
        print("No checkpoints found.", file=sys.stderr)
        sys.exit(1)
    print("Checkpoints:")
    for p in ckpts:
        print("  -", p)

    print("Loading sequences...")
    id2seq = load_id2seq(args.seq_tsv)
    print("Total sequences:", len(id2seq))

    print("Reading pairs...")
    label_col = None if args.label_col < 0 else args.label_col
    pairs = read_pairs(args.pairs_tsv, args.sid1_col, args.sid2_col, label_col, args.skip_header)
    print("Total pairs:", len(pairs))

    print("Preparing embeddings...")
    emb_path = args.emb_files[args.use_emb]
    seq2t_obj = s2t(emb_path)
    # dim = seq2t_obj.dim  # (not needed directly here)

    # load models
    models_list = []
    for ck in ckpts:
        print("Loading model:", ck)
        m = models.load_model(ck, compile=False)
        models_list.append(m)

    print("Predicting...")
    results = []
    pos_idx = args.positive_index
    neg_idx = 1 - pos_idx

    for batch in batched(pairs, args.batch_size):
        s1_list, s2_list, n1, n2 = [], [], [], []
        for p1, p2, _ in batch:
            n1.append(p1); n2.append(p2)
            s1_list.append(id2seq.get(p1, "X"))
            s2_list.append(id2seq.get(p2, "X"))
        X1 = embed_batch(s1_list, seq2t_obj, args.seq_size)
        X2 = embed_batch(s2_list, seq2t_obj, args.seq_size)

        probs_accum = None
        for m in models_list:
            probs = m.predict([X1, X2], verbose=0)  # [B, 2], softmax
            probs_accum = probs if probs_accum is None else (probs_accum + probs)
        probs_mean = probs_accum / float(len(models_list))

        pos = probs_mean[:, pos_idx]
        neg = probs_mean[:, neg_idx]

        for (p1, p2, lab), p_pos, p_neg in zip(batch, pos, neg):
            if lab is None:
                results.append([p1, p2, "{:.6f}".format(p_pos), "{:.6f}".format(p_neg)])
            else:
                results.append([p1, p2, "{:.6f}".format(p_pos), "{:.6f}".format(p_neg), lab])

    # write CSV
    header = ["protein1", "protein2", "prob_interact", "prob_noninteract"]
    if label_col is not None:
        header.append("label")
    out_dir = os.path.dirname(args.out_csv) or "."
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(results)
    print("Saved:", args.out_csv)

if __name__ == "__main__":
    main()