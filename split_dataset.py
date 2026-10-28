"""
Split official_python_corpus.json into reproducible Train (80%), Val (10%), and Test (10%) splits.
"""
import json
import random
import os

def create_splits(
    input_file="data/official_python_corpus.json",
    output_dir="data",
    train_ratio=0.80,
    val_ratio=0.10,
    seed=42
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deterministic shuffle
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)

    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_data = [data[i] for i in train_idx]
    val_data = [data[i] for i in val_idx]
    test_data = [data[i] for i in test_idx]

    os.makedirs(output_dir, exist_ok=True)

    def write_jsonl(filename, items):
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        print(f"[+] Saved {len(items):>5} samples -> {filepath}")

    write_jsonl("train.jsonl", train_data)
    write_jsonl("val.jsonl", val_data)
    write_jsonl("test.jsonl", test_data)

    print("\n" + "=" * 60)
    print(f"  DATASET SPLIT COMPLETE (Total: {n_total})")
    print(f"  - Training Set   : {len(train_data):>5} samples ({train_ratio*100:.0f}%)")
    print(f"  - Validation Set : {len(val_data):>5} samples ({val_ratio*100:.0f}%)")
    print(f"  - Held-out Test  : {len(test_data):>5} samples ({len(test_data)/n_total*100:.0f}%)")
    print("=" * 60)

if __name__ == "__main__":
    create_splits()
