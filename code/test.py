#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda 
@Email: 119020045@link.cuhk.edu.com
@file: test.py.py
@time: 2021/07/27

Notes:


"""

import tSNE as ts
from sklearn import datasets

PERPLEXITY = 20
SEED = 1                    # Random seed
MOMENTUM = 0.8
LEARNING_RATE = (1e2, 1e1)
BETA = 1e-1
RHO = 1e-1
# LEARNING_RATE = (1e2, 0)
# BETA = 0
# RHO = 0
NUM_ITERS = 50
TSNE = True
TSNE_grad = True
NUM_PLOTS = 5
NUM_EIGEN = 2

#data = datasets.load_digits()

data = datasets.load_wine()
X = data['data']
y = data['target']

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
                 plot=NUM_PLOTS)
