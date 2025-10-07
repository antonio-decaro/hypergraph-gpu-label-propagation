# **Data Source:** [AllSet Repository](https://github.com/jianhao2016/AllSet/tree/main?tab=readme-ov-file)

---

## 1. Create a virtual environment

Clone the repository (or download only the `data` folder if you prefer):

```bash
git clone https://github.com/jianhao2016/AllSet.git
cd AllSet
```
Create a Python virtual environment:
```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows PowerShell
python -m venv venv
venv\Scripts\activate
```

## 2. Install packages

Upgrade pip and install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Unzip the dataset

Unzip the hypergraph datasets from the compressed folder:

```bash
unzip data.zip -d data
```

The final structure should look like this:

```bash
data/
└── cocitation/
    └── cora/
        ├── hypergraph.pickle
        ├── labels.pickle
        └── ...
```

## 4. Run extract_data.py

```bash
python extract_data.py --data_dir data/cocitation/cora
```

After execution, a ```processed/``` folder will be generated with the dataset in CSR format:

```bash
data/cocitation/cora/processed/
├── H_data.txt
├── H_indices.txt
├── H_indptr.txt
├── labels.txt
└── meta.txt
```

The files ```hypergraph.pickle``` and ```labels.pickle``` come from the AllSet repository.
