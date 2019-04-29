# Seed for reproducability
import torch
import numpy as np

import scanpy as sc
import os , sys

sys.path.append(os.path.abspath("../.."))
n_epochs_all = None
test_mode = False


def if_not_test_else(x, y):
    if not test_mode:
        return x
    else:
        return y
sys.path.append(os.path.abspath("../.."))
n_epochs_all = None
test_mode = False


def if_not_test_else(x, y):
    if not test_mode:
        return x
    else:
        return y


torch.manual_seed(0)
np.random.seed(0)



save_path = "data/"
filename = "../data/ica_bone_marrow_h5.h5"
adata = sc.read_10x_h5(filename)


adata.var_names_make_unique()
adata.obs_names_make_unique()

# adata.shape
# adata.obs_names
# adata.var_names

sc.pp.filter_cells(adata, min_genes= 200)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=1)

mito_genes = adata.var_names.str.startswith("MT-")


adata.obs["percent_mito"] = (
    np.sum(adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
)
adata.obs["n_counts"] = adata.X.sum(axis=1).A1

adata = adata[adata.obs["n_genes"] < 2500, :]
adata = adata[adata.obs["percent_mito"] < 0.05, :]


"""

Normalization and more filtering

We only keep highly variable genes

"""




adata_original = adata.copy()

sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
sc.pp.log1p(adata)




min_mean = if_not_test_else(0.0125, -np.inf)
max_mean = if_not_test_else(3, np.inf)
min_disp = if_not_test_else(0.5, -np.inf)
max_disp = if_not_test_else(None, np.inf)

sc.pp.highly_variable_genes(
    adata,
    min_mean=min_mean,
    max_mean=max_mean,
    min_disp=min_disp,
    max_disp=max_disp
    # n_top_genes=500
)

adata.raw = adata

highly_variable_genes = adata.var["highly_variable"]
adata = adata[:, highly_variable_genes]

sc.pp.regress_out(adata, ["n_counts", "percent_mito"])
sc.pp.scale(adata, max_value=10)

# Also filter the original adata genes
adata_original = adata_original[:, highly_variable_genes]
print(highly_variable_genes.sum())

# We also store adata_original into adata.raw
# (which was designed for this purpose but actually has limited functionnalities)
adata.raw = adata_original



"""
 Compute the scVI latent space
"""



import scvi
from scvi.dataset.anndata import AnnDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE
from typing import Tuple


def compute_scvi_latent(
    adata: sc.AnnData,
    n_latent: int = 5,
    n_epochs: int = 100,
    lr: float = 1e-3,
    use_batches: bool = False,
    use_cuda: bool = True,
) -> Tuple[scvi.inference.Posterior, np.ndarray]:
    """Train and return a scVI model and sample a latent space

    :param adata: sc.AnnData object non-normalized
    :param n_latent: dimension of the latent space
    :param n_epochs: number of training epochs
    :param lr: learning rate
    :param use_batches
    :param use_cuda
    :return: (scvi.Posterior, latent_space)
    """
    # Convert easily to scvi dataset
    scviDataset = AnnDataset(adata)

    # Train a model
    vae = VAE(
        scviDataset.nb_genes,
        n_batch=scviDataset.n_batches * use_batches,
        n_latent=n_latent,
    )
    trainer = UnsupervisedTrainer(vae, scviDataset, train_size=1.0, use_cuda=use_cuda)
    trainer.train(n_epochs=n_epochs, lr=lr)
    ####

    # Extract latent space
    posterior = trainer.create_posterior(
        trainer.model, scviDataset, indices=np.arange(len(scviDataset))
    ).sequential()

    latent, _, _ = posterior.get_latent()

    return posterior, latent



n_epochs = 10 if n_epochs_all is None else n_epochs_all

scvi_posterior, scvi_latent = compute_scvi_latent(
    adata_original, n_epochs=n_epochs, n_latent=6
)
adata.obsm["X_scvi"] = scvi_latent
