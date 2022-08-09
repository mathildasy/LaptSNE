#!usr/bin/env python
# -*- coding:utf-8 -*-


import time
import os
import load_data
from sklearn.manifold import TSNE
from visdom import Visdom

def append_log(vis,info):
    log_win = vis.text('', opts=dict(title='log'))
    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = "<b>[{0}] \n </b>: {1}".format(localtime, info)
    vis.text(line, win=log_win, append=True)


def laplacian_TSNE(KLnum, beta, num_eigen, dataname = 'COIL100', n_components = 2):

    NUM_ITERS = 200
    NUM_SAMPLES = -1
    LEARNING_RATE = 1e2 # update Y
    BETA = beta

    DATASET = {
        'PenDigits': load_data.digits,
        'COIL20': load_data.COIL20,
        # 'COIL100': load_data.COIL100,
        # 'MNIST':load_data.MNIST,
        # 'Fashion MNIST':load_data.Fashion_MNIST,
    }
    DATANAME = dataname
    X, y = DATASET[DATANAME](num_samples = NUM_SAMPLES, seed = 0)
    if X.shape[0] <= 10000:
        knn_type = 'small'
    else:
        knn_type = 'large'

    # EXAGGERATE_STAGE: momentum = 0.5; after that, momentum = 0.8; There is no effect on P.
    NUM_EIGEN = num_eigen; EXAGGERATE_STAGE = NUM_ITERS; EXA_RATIO = 1
    opt_perplexity = 25

    initial_y = 'spectral'
    NEW_OBJ = 'firstK'
    #KERNEL = 'Gaussian'
    sigman_type = 'constant'
    sigman_constant = 1e0

    KERNEL = 'Student t'

    info = {
        'Dataname': DATANAME,
        'optimal perplexity': opt_perplexity,
        'initial_y': initial_y,
        'BETA': BETA,
        'NUM_ITERS': NUM_ITERS,
        'NUM_SAMPLES': NUM_SAMPLES,
        'learning_rates': LEARNING_RATE,
        'num_eigen' : NUM_EIGEN,
        'exa_stage': EXAGGERATE_STAGE,
        'kernel': KERNEL,
        'EXA_RATIO': EXA_RATIO
    }

    # method = 'barnes_hut'
    method = 'exact'
    if KERNEL == 'Gaussian':
        info['sigman_type']= sigman_type
        info['sigman_constant'] = sigman_constant
        ENV = f'{DATANAME}({NUM_SAMPLES}_&_{n_components}_{opt_perplexity})_{NUM_EIGEN}_{NEW_OBJ}_lr({LEARNING_RATE})_beta({BETA})_{KERNEL}({sigman_type}-{sigman_constant}))_EXAG:{EXAGGERATE_STAGE}({EXA_RATIO}))_{method}'
    else:
        ENV = f'{DATANAME}({NUM_SAMPLES}_&_{n_components}_{opt_perplexity})_{initial_y}_{NUM_EIGEN}_{NEW_OBJ}_lr({LEARNING_RATE})_beta({BETA})_{KERNEL})_EXAG:{EXAGGERATE_STAGE}({EXA_RATIO})))_{method}_faster'

    vis = Visdom(env = ENV)
    append_log(vis, str(info))

    out_dir = f'./output/{DATANAME}(per{opt_perplexity})/beta={BETA}/num_eigen={NUM_EIGEN}/n_components={n_components}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_dir2 = os.path.join(out_dir, 'X_embedded')
    if not os.path.exists(out_dir2): os.makedirs(out_dir2)
    out_dir3 = os.path.join(out_dir2, 'image')
    if not os.path.exists(out_dir3): os.makedirs(out_dir3)

    X_embedded  = TSNE(n_components= n_components, perplexity = opt_perplexity, init = initial_y,
                    learning_rate = LEARNING_RATE, method = method,
                    early_exaggeration = EXA_RATIO, 
                    vis = vis, label = y, num_eigen= NUM_EIGEN, beta = BETA,
                    n_iter = NUM_ITERS, exag_stage = EXAGGERATE_STAGE, kernel = KERNEL, 
                    new_obj = NEW_OBJ, knn_type = knn_type, 
                    sigman_type = sigman_type, sigman_constant = sigman_constant,
                    out_dir = out_dir).fit_transform(X)

    vis.save([ENV])

    print('-----end-----')


if __name__ == '__main__':
    KLnum=0
    beta = 2e-2
    num_eigen = 19
    dataname = 'COIL20'
    print(KLnum, beta, num_eigen, dataname)
    laplacian_TSNE(KLnum, beta, num_eigen, dataname = dataname, n_components = 2)



