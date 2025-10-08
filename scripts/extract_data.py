import os
import pickle as pkl
import argparse
import json
import numpy as np

def load_villain_format(data_dir):
    h_path = os.path.join(data_dir, "H.pickle")
    l_path = os.path.join(data_dir, "L.pickle")

    if not os.path.isfile(h_path):
        alt_path = os.path.join(data_dir, "H.pkl")
        if not os.path.isfile(alt_path):
            raise FileNotFoundError("H.pickle / H.pkl not found in Villain dataset")
        with open(alt_path, "rb") as f:
            center, _, hyperedges = pkl.load(f)
        hypergraph = {i: [int(v) for v in edge] for i, edge in enumerate(hyperedges)}
    else:
        with open(h_path, "rb") as f:
            H = pkl.load(f)
        if isinstance(H, dict):
            hypergraph = H
        else:
            V_idx, E_idx = H
            hypergraph = {}
            for v, e in zip(V_idx.tolist(), E_idx.tolist()):
                hypergraph.setdefault(e, []).append(v)

    labels = None
    if os.path.isfile(l_path):
        with open(l_path, "rb") as f:
            labels = pkl.load(f)

    return hypergraph, labels


def load_classic_format(data_dir):
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

    return hypergraph, labels


def convert_to_json(data_dir, dataset_name="hypergraph"):
    try:
        hypergraph, labels = load_classic_format(data_dir)
        print("Loaded classic format (hypergraph.pickle / labels.pickle)")
    except FileNotFoundError:
        hypergraph, labels = load_villain_format(data_dir)
        print("Loaded Villain format (H.pickle / L.pickle)")

    all_nodes = sorted(set(n for nodes in hypergraph.values() for n in nodes))
    print(f"Hypergraph loaded: {len(all_nodes)} nodes, {len(hypergraph)} edges")

    node_data = {}
    label_array = []

    if labels is not None:
        if isinstance(labels, dict):
            for n in all_nodes:
                node_info = {}
                lbl = labels.get(n, None)
                if lbl is not None:
                    if isinstance(lbl, dict):
                        node_info.update(lbl)
                        label_array.append(-1)
                    else:
                        node_info["label"] = str(lbl)
                        try:
                            label_array.append(int(lbl))
                        except:
                            label_array.append(-1)
                else:
                    label_array.append(-1)
                node_data[str(n)] = node_info
        else:
            labels = np.asarray(labels)
            for n in all_nodes:
                node_data[str(n)] = {}
                if n < len(labels):
                    try:
                        label_array.append(int(labels[n]))
                    except:
                        label_array.append(-1)
                else:
                    label_array.append(-1)
    else:
        for n in all_nodes:
            node_data[str(n)] = {}
            label_array.append(-1)

    edge_dict = {}
    for e_id, (edge_key, node_list) in enumerate(hypergraph.items()):
        edge_dict[str(e_id)] = [str(n) for n in node_list]

    output = {
        "hypergraph-data": {"name": dataset_name},
        "node-data": node_data,
        "edge-dict": edge_dict
    }

    output_path = os.path.join(data_dir, f"{dataset_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    labels_txt_path = os.path.join(data_dir, f"{dataset_name}_labels.txt")
    np.savetxt(labels_txt_path, np.array(label_array, dtype=np.int32), fmt="%d")

    print(f"File JSON saved in: {output_path}")
    print(f"Labels TXT saved in: {labels_txt_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert hypergraph dataset to JSON format (supports classic and Villain formats)")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--name", type=str, default="hypergraph", help="Dataset name in JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise NotADirectoryError(f"Directory '{args.data_dir}' does not exist.")
    convert_to_json(args.data_dir, args.name)
