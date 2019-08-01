%load_ext autoreload
%autoreload 2



n_epochs_all = None
save_path = 'data/'
show_plot = True

import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import numpy as np
import numpy.random as random
import pandas as pd
import scanpy as sc
import louvain

use_cuda = True
from scvi.dataset.dataset import GeneExpressionDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models import SCANVI, VAE


from umap import UMAP


from scvi.dataset import DownloadableRawAnnDataset


DATAFILE = '../data/thymus/'
all_dataset = DownloadableRawAnnDataset("A42.v01.yadult_raw.h5ad", save_path=DATAFILE)
