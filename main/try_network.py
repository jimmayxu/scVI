
%load_ext autoreload
%autoreload 2

import numpy as np
import scipy as sp
import scanpy.api as sc
import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

sys.path.append('main')
from BBKNN_Network_analysis import *

sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=100)
sc.logging.print_version_and_date()





adata = sc.read("data/PBMC.merged.h5ad")



bdata = sc.tl.umap(adata,n_components=3,copy=True)



n_neighbor = 30
select = get_grid(bdata,scale=1,select_per_grid =10,n_neighbor = n_neighbor)

idata = impute_neighbor(bdata,n_neighbor=n_neighbor)
tfdata = new_exp_matrix(bdata,idata,select, n_min_exp_cell = 50, min_mean=0,
                        min_disp=3, ratio_expressed = 0.05,
                        example_gene='PGD',show_filter = None,
                        max_cutoff=0.1, tflist = None)

generate_gene_network(tfdata,n_neighbors=n_neighbor)
anno_key = "anno"
anno_uniq,anno_ratio = impute_anno(bdata,select,anno_key,n_neighbor=n_neighbor)
draw_graph(tfdata,  anno_uniq, anno_ratio,adjust=True)
