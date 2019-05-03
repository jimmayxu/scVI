import os
os.getcwd()

import sys
sys.path.append('scvi')

import torch
import numpy as np
import pandas as pd
import scanpy as sc


save_path = "/lustre/scratch117/cellgen/team205/tpcg/backup/backup_20190401/sc_sclassification/CellTypist/data_repo/MouseAtlas/"
adata = sc.read_10x_h5(os.path.join( save_path, "MouseAtlas.total.h5ad" ))


import scvi
from dataset.anndata import AnnDataset
from inference import UnsupervisedTrainer
from models.vae import VAE
from typing import Tuple
