#!/usr/bin/env python3

import argparse
import gdown
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# This script downloads and extracts datasets from given URLs.
# The structure is a dictionary where keys are dataset names and values are tuples of (URL, is_google_drive_link).
datasets = {
  "arxiv-kaggle": ("https://drive.google.com/uc?export=download&id=1TOTNjQie892rmi-4hqmT1EAPm_A47zCR", True),
  "eventernote-places": ("https://zenodo.org/records/11263394/files/eventernote-places.json", False),
  "coauth-MAG-History": ("https://zenodo.org/records/13151009/files/coauth-MAG-History.json", False),
  "ndc-substances": ("https://zenodo.org/records/10929019/files/NDC-substances.json", False),
  "coauth-MAG-Geology": ("https://zenodo.org/records/10928443/files/coauth-MAG-Geology.json", False),
  "senate-committees": ("https://zenodo.org/records/10957699/files/senate-committees.json", False),
  "coauth-DBLP": ("https://zenodo.org/records/13203175/files/coauth-DBLP.json", False),
  "stack-overflow": ("https://zenodo.org/records/10373328/files/threads-stack-overflow.json", False),  
}

def download_file(url, output_path, is_google_drive=False):
  if is_google_drive:
    gdown.download(url, str(output_path), quiet=False)
  else:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error for bad responses
    total = int(response.headers.get("content-length", 0))
    has_total = total > 0
    chunk_size = 8192
    progress = tqdm(total=total if has_total else None,
                    unit='B',
                    unit_scale=True,
                    desc=output_path.name,
                    dynamic_ncols=True)

    with open(output_path, 'wb') as f:
      for chunk in response.iter_content(chunk_size=chunk_size):
        f.write(chunk)
        progress.update(len(chunk))

    progress.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Download selected datasets.")
  parser.add_argument(
    "--datasets",
    nargs="+",
    choices=list(datasets.keys()),
    default=list(datasets.keys()),
    help="Names of datasets to download (default: all available datasets)."
  )
  args = parser.parse_args()

  selected = {name: datasets[name] for name in args.datasets}

  for dataset_name, (url, is_google_drive) in selected.items():
    tmp_path = Path(f"./{dataset_name}.tmp")
    download_file(url, tmp_path, is_google_drive)

    if zipfile.is_zipfile(tmp_path):
      extract_dir = Path(dataset_name)
      extract_dir.mkdir(parents=True, exist_ok=True)
      with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    else:
      json_path = Path(f"./{dataset_name}.json")
      tmp_path.rename(json_path)
