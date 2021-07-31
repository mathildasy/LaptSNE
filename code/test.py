#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda 
@Email: 119020045@link.cuhk.edu.com
@file: test.py
@time: 2021/07/27

Notes:


"""

#import tSNE as ts
import tSNE_v2 as ts
from sklearn import datasets

PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.8
NUM_ITERS = 200
TSNE = True
TSNE_grad = True
NUM_PLOTS = 5
NUM_EIGEN = 10

# revised
LEARNING_RATE = (1e2, 1e0)
BETA = 1e-1
RHO = 1e-1
EXAGGERATE_STAGE = 20
LAST_STAGE = 10

# EXAGGERATE_STAGE = 0
# LAST_STAGE = 0



# original t-SNE
# LEARNING_RATE = (1e2, 0)
# BETA = 0
# RHO = 0
# EXAGGERATE_STAGE = 20
# # EXAGGERATE_STAGE = 0
# LAST_STAGE = 0

data = datasets.load_digits()
NUM_SAMPLE = 400
#data = datasets.load_wine()
X = data['data'][:NUM_SAMPLE]
y = data['target'][:NUM_SAMPLE]

# Obtain matrix of joint probabilities p_ij
P = ts.p_joint(X, PERPLEXITY)

# Fit SNE or t-SNE
Y = ts.estimate_sne(X, y, P,
                 num_iters=NUM_ITERS,
                 q_fn= ts.q_tsne if TSNE else ts.q_joint,
                 grad_fn= ts.tsne_grad if TSNE_grad else ts.yy_grad,
                 learning_rate1=LEARNING_RATE[0],
                 learning_rate2=LEARNING_RATE[1],
                 momentum=MOMENTUM,
                 beta=BETA,
                 rho=RHO,
                 num_eigen = NUM_EIGEN,
                 plot=NUM_PLOTS,
                 exa_stage = EXAGGERATE_STAGE,
                 lst_stage = LAST_STAGE)
