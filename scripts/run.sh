#!/usr/bin/env bash

set -e

DATASETS=(
    "data/coauthorship/cora"
    "data/coauthorship/dblp"
    "data/cocitation/citeseer"
    "data/cocitation/cora"
    "data/cocitation/pubmed"
)

for DATA_DIR in "${DATASETS[@]}"; do
    echo "==========================================="
    echo " Processing: $DATA_DIR"
    echo "==========================================="

    python extract_data.py --data_dir "$DATA_DIR"
done

echo "All datasets processed successfully!"
