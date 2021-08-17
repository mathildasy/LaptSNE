#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda 
@Email: 119020045@link.cuhk.edu.com
@file: test_3.py.py
@time: 2021/08/17

Notes:


"""

import tSNE_v3 as ts
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt

SEED = 1  # Random seed
MOMENTUM = 0.8
NUM_ITERS = 100
TSNE = True
TSNE_grad = True
EXA_RATIO = 4
NUM_PLOTS = 3

# data = datasets.load_iris()
# NUM_SAMPLE = 400
# PERPLEXITY = 40
# LEARNING_RATE = 1e1 # change with NUM_SAMPLE
# EXAGGERATE_STAGE = 50 # change with NUM_SAMPLE
# LAST_STAGE = 5 # change with NUM_SAMPLE
# NUM_EIGEN = 4
# RDSEED = 1
# GK = 1
# BETA = 1e-2  # change with delta Y

data = datasets.load_digits()
NUM_SAMPLE = 400
PERPLEXITY = 25
LEARNING_RATE = [1e2, 1e-1]  # change with NUM_SAMPLE
EXAGGERATE_STAGE = 50  # change with NUM_SAMPLE
LAST_STAGE = 5  # change with NUM_SAMPLE
GK = 1
BETA = 1e0  # change with delta Y
NUM_EIGEN = 12
RDSEED = 1
SIGMA = 1e0

# data = datasets.load_digits()
# NUM_SAMPLE = 800
# PERPLEXITY = 30
# LEARNING_RATE = [1e2, 1e0]  # change with NUM_SAMPLE
# EXAGGERATE_STAGE = 50  # change with NUM_SAMPLE（不需要太大）
# LAST_STAGE = 20  # change with NUM_SAMPLE （到底会不会过头呢？）
# GK = 1
# BETA = 3e0  # change with delta Y （较为敏感）
# NUM_EIGEN = 12
# RDSEED = 1

print(data['data'].shape)
X = data['data'][:NUM_SAMPLE]
y = data['target'][:NUM_SAMPLE]

# Obtain matrix of joint probabilities p_ij
P, _ = ts.p_joint(X, PERPLEXITY)
print(_)
# Fit SNE or t-SNE
Y, lam_list, sigmas_list = ts.estimate_sne(X, y, P,
                     num_iters=NUM_ITERS,
                     q_fn= ts.q_tsne if TSNE else ts.q_joint,
                     learning_rate1=LEARNING_RATE[0],
                     learning_rate2=LEARNING_RATE[1],
                     momentum=MOMENTUM,
                     beta=BETA,
                     num_eigen=NUM_EIGEN,
                     plot=NUM_PLOTS,
                     exa_stage=EXAGGERATE_STAGE,
                     lst_stage=LAST_STAGE,
                     rdseed=RDSEED,
                     sigmas = SIGMA,
                     exa_ratio = EXA_RATIO)


lam_df = pd.DataFrame(lam_list, columns= ['eigen value'+str(i+1) for i in range(NUM_EIGEN)])
lam_df.plot()
plt.savefig('lamda.png')
plt.show()

# sigma_df = pd.DataFrame(sigmas_list[-LAST_STAGE:])
sigma_df = pd.DataFrame(sigmas_list)
sigma_df.plot()
plt.title('sigmas')
plt.show()



print('-----end-----')