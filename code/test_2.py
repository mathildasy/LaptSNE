#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda
@Email: 119020045@link.cuhk.edu.com
@file: test_2.py
@time: 2021/08/01

Notes:


"""
# !usr/bin/env python
# -*- coding:utf-8 -*-

import tSNE_v2 as ts
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1  # Random seed
MOMENTUM = 0.8
NUM_ITERS = 100
TSNE = True
TSNE_grad = True
NUM_PLOTS = 3

data = datasets.load_iris()
NUM_SAMPLE = 400
PERPLEXITY = 40
LEARNING_RATE = 1e1 # change with NUM_SAMPLE
EXAGGERATE_STAGE = 50 # change with NUM_SAMPLE
LAST_STAGE = 5 # change with NUM_SAMPLE
NUM_EIGEN = 4
RDSEED = 1
GK = 1
BETA = 1e-2  # change with delta Y

# data = datasets.load_digits()
# PERPLEXITY = 20
# LEARNING_RATE = 1e2  # change with NUM_SAMPLE
# EXAGGERATE_STAGE = 50  # change with NUM_SAMPLE
# LAST_STAGE = 5  # change with NUM_SAMPLE
# GK = 1
# BETA = 1e-2  # change with delta Y
# NUM_EIGEN = 12
# RDSEED = 1
# GK = 0
# BETA = 1e-5  # change with delta Y


NUM_SAMPLE = 400
X = data['data'][:NUM_SAMPLE]
y = data['target'][:NUM_SAMPLE]

# Obtain matrix of joint probabilities p_ij
P, sigmas = ts.p_joint(X, PERPLEXITY)

# Fit SNE or t-SNE
Y, lam_list = ts.estimate_sne(X, y, P,
                     num_iters=NUM_ITERS,
                     q_fn= ts.q_tsne if TSNE else ts.q_joint,
                     grad_fn= ts.tsne_grad if TSNE_grad else ts.yy_grad,
                     eigen_fn= ts.gk_grad if GK else ts.eigen_grad,
                     learning_rate1=LEARNING_RATE,
                     momentum=MOMENTUM,
                     beta=BETA,
                     num_eigen=NUM_EIGEN,
                     plot=NUM_PLOTS,
                     exa_stage=EXAGGERATE_STAGE,
                     lst_stage=LAST_STAGE,
                     rdseed=RDSEED,
                     sigmas=sigmas)


lam_df = pd.DataFrame(lam_list).T
print(lam_df.shape)

lam_df = pd.DataFrame(lam_list, columns= ['eigen value'+str(i+1) for i in range(NUM_EIGEN)])
lam_df.plot()
plt.show()



print('-----end-----')