
sys.path.append('scvi')
from dataset import LoomDataset, CsvDataset, Dataset10X, AnnDataset
import urllib.request
import os
import numpy as np
from dataset import BrainLargeDataset, CortexDataset, PbmcDataset, RetinaDataset, HematoDataset, CbmcDataset, BrainSmallDataset, SmfishDataset
save_path = "data/"
pbmc_dataset = PbmcDataset(save_path=save_path)

cbmc_dataset = CbmcDataset(save_path=os.path.join(save_path, "citeSeq/"))

hemato_dataset = HematoDataset(save_path=os.path.join(save_path, 'HEMATO/'))

tenX_dataset = Dataset10X("pbmc4k", save_path=save_path)

retina_dataset = RetinaDataset(save_path=save_path)

dataset = cbmc_dataset

X = dataset.X.toarray().transpose()
X = dataset.X.transpose()
labels = dataset.labels
dataset.n_labels

X.shape

np.savetxt("../data/pbmc_X.csv", X, delimiter=",")
np.savetxt("../data/pbmc_label.csv", labels, delimiter=",")



