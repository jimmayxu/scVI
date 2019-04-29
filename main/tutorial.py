
import os
os.chdir("/Users/zx3/PycharmProjects/scVI/tests")

n_epochs_all = None
save_path = 'data/'
show_plot = True

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scvi.dataset import CortexDataset, RetinaDataset
from scvi.models import *
from scvi.inference import UnsupervisedTrainer
import torch

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

gene_dataset = CortexDataset(save_path=save_path)

n_epochs=400 if n_epochs_all is None else n_epochs_all
lr=1e-3
use_batches=False
use_cuda=True

vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.75,
                              use_cuda=use_cuda,
                              frequency=5,
                              verbose=True)

trainer.train(n_epochs=n_epochs, lr=lr)
torch.save(trainer.model.state_dict(), '%s/vae.pkl' % save_path)

ll_train_set = trainer.history["ll_train_set"]
ll_test_set = trainer.history["ll_test_set"]
x = np.linspace(0,500,(len(ll_train_set)))
plt.plot(x, ll_train_set)
plt.plot(x, ll_test_set)
plt.ylim(1000,2000)


from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI
from scvi.models.utils import one_hot

n_latent = 10
n_layers = 1
float = 0.1
n_hidden = 128
n_batch = 0
dropout_rate = 0.1

n_input = gene_dataset.nb_genes

z_encoder = Encoder(n_input, n_latent, n_layers=n_layers, n_hidden=n_hidden,
                         dropout_rate=dropout_rate)
l_encoder = Encoder(n_input, 1, n_layers=1, n_hidden=n_hidden, dropout_rate=dropout_rate)

decoder = DecoderSCVI(n_latent, n_input, n_cat_list=[n_batch], n_layers=n_layers, n_hidden=n_hidden)

y = None
x = torch.from_numpy(gene_dataset.X)
x_ = x
dispersion = "gene"
batch_index=None

qz_m, qz_v, z = z_encoder(x_, y)
ql_m, ql_v, library = l_encoder(x_)

px_scale, px_r, px_rate, px_dropout = decoder(dispersion, z, library, batch_index, y)
