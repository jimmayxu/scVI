n_epochs_all = None
save_path = 'data/'
show_plot = True

import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, RetinaDataset
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch




## Correction for batch effects

gene_dataset = RetinaDataset(save_path=save_path)
n_epochs=50 if n_epochs_all is None else n_epochs_all
lr=1e-3
use_batches=True
use_cuda=True

### Train the model and output model likelihood every 5 epochs
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.9,
                              use_cuda=use_cuda,
                              frequency=5)
trainer.train(n_epochs=n_epochs, lr=lr)
# Plotting the likelihood change across the 50 epochs of training: blue for training error and orange for testing error.

ll_train = trainer.history["ll_train_set"]
ll_test = trainer.history["ll_test_set"]
x = np.linspace(0,50,(len(ll_train)))
plt.plot(x, ll_train)
plt.plot(x, ll_test)
plt.ylim(min(ll_train)-50, 1000)
plt.show()

full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
print("Entropy batch mixing :", full.entropy_batch_mixing())
#%% md
**Coloring by batch and cell type**
#%%
# obtaining latent space in the same order as the input data
n_samples_tsne = 1000
full.show_t_sne(n_samples=n_samples_tsne, color_by='batches and labels')
#%%
def allow_notebook_for_test():
    print("Testing the basic tutorial notebook")
#%%
