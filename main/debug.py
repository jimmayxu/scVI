%load_ext autoreload
%autoreload 2

DATAFILE = '/lustre/scratch117/cellgen/team205/zx3/HCA_data/'
DATAFILE = 'data/'
from scvi.dataset import DownloadableRawAnnDataset
# all_dataset = DownloadableRawAnnDataset("MouseAtlas.total.h5ad", save_path=DATAFILE)
all_dataset = DownloadableRawAnnDataset("PBMC.merged.h5ad", save_path=DATAFILE)


