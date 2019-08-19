
import numpy as np

from scvi.dataset import CortexDataset
from scvi.inference import UnsupervisedTrainer
from scvi.models.vae import VAE

def compute_empirical_means_cov_mat(X):

    mean_emp = np.mean(X, axis=0)
    sigma_emp = np.dot( (X - mean_emp).T, (X - mean_emp)) / (X.shape[0] - 1)

    return mean_emp, sigma_emp


if __name__ == '__main__':

    dataset = CortexDataset()
    model = VAE(n_input=dataset.nb_genes)
    trainer = UnsupervisedTrainer(model=model, gene_dataset=dataset)

    trainer.train(n_epochs=200, lr=1e-2)

    full = trainer.create_posterior(trainer.model, trainer.gene_dataset,indices=np.arange(len(trainer.gene_dataset)))
    px_scale = full.get_sample_scale()
    _, cov_mat = compute_empirical_means_cov_mat(px_scale)

    print(cov_mat.shape)
    print(cov_mat)
