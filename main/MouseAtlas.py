import os
os.getcwd()

import sys
sys.path.append('scvi')

import torch
import numpy as np
import pandas as pd
import scanpy as sc


save_path = "/lustre/scratch117/cellgen/team205/tpcg/backup/backup_20190401/sc_sclassification/CellTypist/data_repo/MouseAtlas/MouseAtlas.total.h5ad"
save_path2 = "/lustre/scratch117/cellgen/team205/tpcg/human_data/HumanAtlas.h5ad"
adata_mouse = sc.read_h5ad(save_path)
adata_human = sc.read_h5ad(save_path2)

# sc.read_10x_mtx
save_path = "/lustre/scratch117/cellgen/team205/zx3/pooled_2019-03-21"
adata = sc.read_10x_mtx(save_path)


import scvi

from dataset.anndata import AnnDataset
from inference import UnsupervisedTrainer
from models.vae import VAE
from typing import Tuple


import scvi

from models.modules import Encoder
