import os
import pickle as pkl
import argparse
import numpy as np
import scipy.sparse as sp

def build_and_save_csr(data_dir: str):
    hypergraph_path = os.path.join(data_dir, "hypergraph.pickle")
    labels_path = os.path.join(data_dir, "labels.pickle")

    if not os.path.isfile(hypergraph_path):
        raise FileNotFoundError(f"File not found: {hypergraph_path}")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"File not found: {labels_path}")

    with open(hypergraph_path, "rb") as f:
        hypergraph = pkl.load(f)
    with open(labels_path, "rb") as f:
        labels = pkl.load(f)

    nodes = sorted(set(n for v in hypergraph.values() for n in v))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edge_to_idx = {e: i for i, e in enumerate(hypergraph.keys())}

    n_nodes, n_edges = len(nodes), len(hypergraph.keys())
    print(f"Loaded hypergraph with {n_nodes} nodes and {n_edges} edges")

    rows, cols = [], []
    for e, node_list in hypergraph.items():
        if not node_list:
            print(f"Warning: edge {e} is empty and will be skipped")
            continue
        e_idx = edge_to_idx[e]
        for n in node_list:
            rows.append(node_to_idx[n])
            cols.append(e_idx)

    data = np.ones(len(rows), dtype=np.float32)
    H_csr = sp.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_edges)).tocsr()
    print(f"CSR matrix built: shape={H_csr.shape}, nnz={H_csr.nnz}")

    if isinstance(labels, dict):
        labels_array = np.array([labels[n] for n in nodes], dtype=np.int32)
    else:
        labels = np.asarray(labels)
        try:
            labels_array = labels[np.array(nodes, dtype=np.int64)].astype(np.int32)
        except Exception as e:
            raise ValueError(
                f"Could not index labels using node IDs. "
                f"labels.shape={labels.shape}, first 5 nodes={nodes[:5]}"
            ) from e

    output_dir = os.path.join(data_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(os.path.join(output_dir, "H_data.txt"), H_csr.data.astype(np.float32), fmt='%f')
    np.savetxt(os.path.join(output_dir, "H_indices.txt"), H_csr.indices.astype(np.int64), fmt='%d')
    np.savetxt(os.path.join(output_dir, "H_indptr.txt"), H_csr.indptr.astype(np.int64), fmt='%d')
    np.savetxt(os.path.join(output_dir, "labels.txt"), labels_array, fmt='%d')

    with open(os.path.join(output_dir, "meta.txt"), "w") as f:
        f.write(f"{n_nodes} {n_edges}\n")

    print(f"Processed data saved to {output_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a hypergraph dataset to CSR format for C++ label propagation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(f"Directory '{args.data_dir}' does not exist.")
    build_and_save_csr(args.data_dir)

