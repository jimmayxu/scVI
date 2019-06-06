

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




gene_dataset = RetinaDataset(save_path=save_path)


adata = sc.AnnData(counts.values, obs=cellinfo, var=geneinfo)
