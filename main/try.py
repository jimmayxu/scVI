%load_ext autoreload
%autoreload 2


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
from scvi.inference import UnsupervisedTrainer, auto_tune_scvi_model
import torch

from scvi.dataset import LoomDataset, CsvDataset, Dataset10X

## Correction for batch effects

gene_dataset = RetinaDataset(save_path=save_path)
#tenX_dataset = Dataset10X("neuron_9k", save_path=save_path)
n_epochs=50 if n_epochs_all is None else n_epochs_all
lr=1e-3
use_batches=True
use_cuda=True

### Train the model and output model likelihood every 5 epochs
from scvi.models.vae import VAE

vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)



n_batch = gene_dataset.n_batches * use_batches
n_input = gene_dataset.nb_genes
n_hidden = 128
n_latent = 10
n_layers= 1
dropout_rate = 0.1
dispersion = "gene"
log_variational = True
reconstruction_loss: str = "zinb"
x = torch.rand([n_batch,n_input])

px_scale, px_r, px_rate, px_dropout, qz_m, qz_v, z, ql_m, ql_v, library = vae.inference(x)
mean = torch.zeros_like(qz_m)
scale = torch.ones_like(qz_v)
from torch.distributions import Normal, kl_divergence as kl

kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)
kl_divergence_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var))).sum(dim=1)
kl_divergence = kl_divergence_z

"""
x_ = torch.log(1 + x)
qz_m, qz_v, z = vae.z_encoder(x_)
ql_m, ql_v, library = vae.l_encoder(x_)

px_r = vae.px_r
vae.decoder = DecoderSCVI(n_latent, n_input)

px_scale, px_r, px_rate, px_dropout = vae.decoder(vae.dispersion, z, library)

px = vae.decoder.px_decoder(z, [n_batch])
px_scale = vae.decoder.px_scale_decoder(px)
px_dropout = ave.px_dropout_decoder(px)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.9,
                              use_cuda=use_cuda,
                              frequency=5)
"""

# ========
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.9,
                              use_cuda=use_cuda,
                              frequency=5)
trainer.train(n_epochs=n_epochs, lr=lr)

ll_train = trainer.history["ll_train_set"]
ll_test = trainer.history["ll_test_set"]
x = np.linspace(0,50,(len(ll_train)))
plt.plot(x, ll_train)
plt.plot(x, ll_test)
plt.ylim(min(ll_train)-50, 1000)
plt.show()

full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
print("Entropy batch mixing :", full.entropy_batch_mixing())

# ========

from scvi.inference import Trainer

from scvi.inference.posterior import Posterior
from sklearn.model_selection._split import _validate_shuffle_split
trainerr = Trainer(vae,gene_dataset)

train_size=0.1

# Posterior
n = len(gene_dataset)
test_size = None
n_train, n_test = _validate_shuffle_split(n, test_size, train_size)
np.random.seed(seed=seed)
permutation = np.random.permutation(n)
indices_test = permutation[:n_test]
indices_train = permutation[n_test:(n_test + n_train)]
Post = trainer.create_posterior(model=vae, gene_dataset=gene_dataset, indices=indices_train, type_class=Posterior)


Post.clustering_scores(prediction_algorithm = "gmm")

# def get_latent(self, sample=False):
latent, _, labels = Post.get_latent()
latent = []
batch_indices = []
labels = []

sample=False
for tensors in iter(DataLoader(gene_dataset)):
    sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
    give_mean = not sample
    latent += [Post.model.sample_from_posterior_z(sample_batch, give_mean=give_mean).cpu()]
    batch_indices += [batch_index.cpu()]
    labels += [label.cpu()]

for i, tensors in enumerate(Post):
    sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
    print(label)

# Plotting the likelihood change across the 50 epochs of training: blue for training error and orange for testing error.

xx = map(Post.to_cuda, iter(Post.data_loader))
list(xx)

xx = iter(Post.data_loader)
l = next(xx)
ll = Post.to_cuda(l)
len(ll)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
#sampler = SequentialSampler(gene_dataset)
sampler = SubsetRandomSampler(indices_train)

kwargs = {'collate_fn': gene_dataset.collate_fn, 'sampler': sampler}
l = iter(Post.data_loader)

l = iter(DataLoader(gene_dataset,**kwargs))
for i, tensors in enumerate(l):
    sample_batch, local_l_mean, local_l_var, batch_index, label = tensors
    print(len(tensors))

#     def loss(self, tensors):
l = iter(DataLoader(gene_dataset,**kwargs))
l = iter(Post.data_loader)


for tensors in l:
    sample_batch, local_l_mean, local_l_var, batch_index, _ = tensors
    reconst_loss, kl_divergence = vae(sample_batch, local_l_mean, local_l_var, batch_index)
    print(kl_divergence)

l = iter(Post.data_loader)
trainer.on_epoch_begin()
for tensors in l:
    los = trainer.loss(tensors)
    print(los)

# def train(self, n_epochs=20, lr=1e-3, eps=0.01, params=None): Class trainer
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
trainer = UnsupervisedTrainer(vae,
                              gene_dataset,
                              train_size=0.9,
                              use_cuda=use_cuda,
                              frequency=5)
trainer.model.train()

trainer.compute_metrics_time = 0
trainer.n_epochs = n_epochs
trainer.compute_metrics()


n_epochs=20
lr=1e-3
eps=0.01
para = trainer.model.parameters()
params = filter(lambda p: p.requires_grad, para)
optimizer = torch.optim.Adam(params, lr=lr, eps=eps)

for epoch in range(n_epochs):
    trainer.on_epoch_begin()
    for tensors_list in trainer.data_loaders_loop():
        loss = trainer.loss(*tensors_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not trainer.on_epoch_end():
        break

import sys
from tqdm import trange
with trange(
    n_epochs,
    desc="my training",
    file=sys.stdout,
    disable=False
) as pbar:
    # We have to use tqdm this way so it works in Jupyter notebook.
    # See https://stackoverflow.com/questions/42212810/tqdm-in-jupyter-notebook
    for trainer.epoch in pbar:
        trainer.on_epoch_begin()
        pbar.update(1)
        for tensors_list in trainer.data_loaders_loop():
            loss = trainer.loss(*tensors_list)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if not trainer.on_epoch_end():
            break

l = trainer.model.parameters()
vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
ll = vae.parameters()


x = [torch.sum(i-ii.cuda()) for (i,ii) in zip(l,ll)]
x = [torch.sum(i-ii) for (i,ii) in zip(l,ll)]
len(x)
x
next(m)



from scvi.inference.posterior import Posterior

ll_train = trainer.history["ll_train_set"]
ll_test = trainer.history["ll_test_set"]



## Loading and Saving model

torch.save(trainer.model.state_dict() , "saved_model/ave_dict.pt")

vvae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
vvae.load_state_dict(torch.load("saved_model/ave_dict.pt"))

kwargs = {
    "batch_size": 128,
    "pin_memory": True
}
#full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
full = Posterior(vvae, gene_dataset, indices=np.arange(len(gene_dataset)), use_cuda=False,
                          data_loader_kwargs=kwargs)
full.clustering_scores(prediction_algorithm = "gmm")
full.show_t_sne()


##  Plot the Gene regulatory network graph

import scanpy.api as sc
sc.tl.umap(adata,n_components=3,copy=True)
