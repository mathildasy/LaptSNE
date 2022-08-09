# import necessary tools
import os
from time import time
from visdom import Visdom

# import external library
import torch
# import numpy as np
# from scipy.spatial.distance import squareform

# import local pakages
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
from tsne_torch import TorchTSNE as torchTSNE
from load_data import *

# DEVICE = torch.device('cuda:')
DEVICE = 'cpu'

NUM_SAMPLES = -1
DATASET = {
    'Iris': iris,
    'PenDigits': digits,
    'COIL20': COIL20,
    # 'COIL100': load_data.COIL100,
    # 'MNIST':load_data.MNIST,
    # 'Fashion MNIST':load_data.Fashion_MNIST,
}

DATANAME = 'COIL20'
X, y = DATASET[DATANAME](num_samples = NUM_SAMPLES, seed = 0)
opt_perplexity = 25

t0 = time()
X = torch.Tensor(X).type(torch.float64)

# prepare hyperparameters
NUM_ITER = 500
LEARNING_RATE = 1e1
KLNUM = 0
BETA = 1e-2
NUM_EIGEN = 19
LAST_STAGE = 0
LASTCOEF = 1

# prepare visualization
ENV = f'{DATANAME}: tsneTorch {LEARNING_RATE}+{KLNUM}+{BETA}+{NUM_EIGEN}+{LAST_STAGE}+{LASTCOEF}'
vis = Visdom(env = ENV)

# prepare save path
out_dir = f'./output/torch/{DATANAME}_num{NUM_SAMPLES}/KL_num={KLNUM}/beta={BETA}/num_eigen={NUM_EIGEN}/lst_stage={LAST_STAGE}/lst_coef={LASTCOEF}'
if not os.path.exists(out_dir): os.makedirs(out_dir)
out_dir2 = os.path.join(out_dir, 'X_embedded')
if not os.path.exists(out_dir2): os.makedirs(out_dir2)
out_dir3 = os.path.join(out_dir2, 'image')
if not os.path.exists(out_dir3): os.makedirs(out_dir3)

print(ENV)

X_emb = torchTSNE(n_components=2, perplexity = opt_perplexity, n_iter=NUM_ITER, initial_dims = 0,
                verbose=True, kl_num = KLNUM, beta = BETA, 
                laststage = LAST_STAGE, lastcoef = LASTCOEF,
                num_eigen = NUM_EIGEN, lr = LEARNING_RATE, out_dir = out_dir,
                PCA_need = False, device = DEVICE).fit_transform(X, None, vis, y)  # returns shape (n_samples, n_components)
t1 = time()

# vis.save([ENV])
print(f'FINISH! with {t1- t0}s')

