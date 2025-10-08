#!/usr/bin/env bash

set -e

DATASETS=(
    "data/coauthorship/cora"
    "data/coauthorship/dblp"
    "data/cocitation/citeseer"
    "data/cocitation/cora"
    "data/cocitation/pubmed"
    "data_vilLain/VilLain/data/trivago"
    "data_vilLain/VilLain/data/amazon"
)

for DATA_DIR in "${DATASETS[@]}"; do
    echo "==========================================="
    echo " Processing: $DATA_DIR"
    echo "==========================================="

    python extract_data.py --data_dir "$DATA_DIR"
done

echo "All datasets processed successfully!"
