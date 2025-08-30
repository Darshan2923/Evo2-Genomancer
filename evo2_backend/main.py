import sys
from compression import gzip
import modal

evo2_image=(
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.11"
    )
    .apt_install(
        ["build-essential","cmake","ninja-build","git","libcudnn8","gcc","g++","libcudnn8-dev"]
    )
    .pip_install("wheel", "setuptools>=42")
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++"
    })
    .run_commands("git clone --recurse-submodules https://github.com/arcinstitute/evo2 && cd evo2 && pip install .")
    .run_commands("pip uninstall -y transformer-engine transformer_engine")
    .run_commands("pip install 'transformer_engine[pytorch]==1.13' --no-build-isolation")
    .pip_install_from_requirements("requirements.txt")
)

app=modal.App("evo2-genomancer",image=evo2_image)

volume=modal.Volume.from_name("hf_cache",create_if_missing=True)
mount_path="root/.cache/huggingface"

@app.function(gpu="H100",volumes={mount_path:volume},timeout=1000)
def run_brca1_analysis():
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    from evo2 import Evo2

    WINDOW_SIZE=8192

    print("Loading evo2 model...")
    model=Evo2(evo2_model="evo2_7b")
    print("Model loaded.")

    brca1_df = pd.read_excel(
        '/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx',
        header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
        'function.score.sd', 'func.class.1', 'func.class.2', 'func.class.3', 'func.class.4',
    ]]

    # Rename columns
    brca1_df.rename(columns={
    'chromosome': 'chrom',
    'position (hg19)': 'pos',
    'reference': 'ref',
    'alt': 'alt',
    'function.score.mean': 'score',
    'func.class': 'class',
    }, inplace=True)

    # Convert to two-class system
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

    # Read the reference genome sequence of chromosome 17
    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break





@app.function()
def brca1_example():
    print("Running BRCA1 variant analysis with Evo2")

    #Run inference
    response=run_brca1_analysis.remote()

    #Show plt from returned data

@app.function(gpu="H100")
def test():
    print("Testing")

@app.local_entrypoint()
def main():
    test.remote()