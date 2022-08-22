#!usr/bin/env python
# -*- coding:utf-8 -*-


import time
import os
import load_data
import numpy as np
from auto_perplexity import opt_numEigen
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from visdom import Visdom

# def append_log(vis,info):
#     log_win = vis.text('', opts=dict(title='log'))
#     localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#     line = "<b>[{0}] \n </b>: {1}".format(localtime, info)
#     vis.text(line, win=log_win, append=True)


def laplacian_TSNE(beta, batch_size, num_neighbors, opt_perplexity = 25, dataname = 'COIL100', n_components = 2, num_eigen = 10, proxy = False):

    NUM_ITERS = 100
    NUM_SAMPLES = 5000
    LEARNING_RATE = 1e2 # update Y
    BETA = beta

    DATASET = {
        'PenDigits': load_data.digits,
        'COIL20': load_data.COIL20,
        'HAR': load_data.HAR,
        'EEG': load_data.EEG,
        'WAV': load_data.WAV,
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
    if num_eigen > 0 :
        NUM_EIGEN = num_eigen
    else:
        NUM_EIGEN = opt_numEigen(X, maxNum=30, perplex=opt_perplexity)
    print('num of eigen value:', NUM_EIGEN)
    EXAGGERATE_STAGE = NUM_ITERS; EXA_RATIO = 1
    

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
        ENV = f'{DATANAME}({NUM_SAMPLES}_&_{proxy}__{opt_perplexity})_{NUM_EIGEN}_{NEW_OBJ}_lr({LEARNING_RATE})_beta({BETA})_{KERNEL}({sigman_type}-{sigman_constant}))_EXAG:{EXAGGERATE_STAGE}({EXA_RATIO}))_{method}'
    else:
        ENV = f'{DATANAME}({NUM_SAMPLES}_&_{proxy}__{opt_perplexity})_{initial_y}_{NUM_EIGEN}_{NEW_OBJ}_lr({LEARNING_RATE})_beta({BETA})_{KERNEL})_EXAG:{EXAGGERATE_STAGE}({EXA_RATIO})))_{method}_faster'

    vis = Visdom(env = ENV)
    # append_log(vis, str(info))

    out_dir = f'./output/{DATANAME}(per{opt_perplexity})/beta={BETA}/num_eigen={NUM_EIGEN}/n_components={n_components}'
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    out_dir2 = os.path.join(out_dir, 'X_embedded')
    if not os.path.exists(out_dir2): os.makedirs(out_dir2)
    out_dir3 = os.path.join(out_dir2, 'image')
    if not os.path.exists(out_dir3): os.makedirs(out_dir3)

    start = time.time()
    X_embedded, K_coef  = TSNE(n_components= n_components, perplexity = opt_perplexity, init = initial_y,
                    learning_rate = LEARNING_RATE, method = method,
                    early_exaggeration = EXA_RATIO, 
                    vis = vis, label = y, num_eigen= NUM_EIGEN, beta = BETA,
                    n_iter = NUM_ITERS, exag_stage = EXAGGERATE_STAGE, kernel = KERNEL, 
                    new_obj = NEW_OBJ, knn_type = knn_type, 
                    sigman_type = sigman_type, sigman_constant = sigman_constant,
                    out_dir = out_dir, proxy=proxy, batch_size=batch_size, num_neighbors=num_neighbors).fit_transform(X)
    end = time.time()
    duration = np.round(end-start,3)
    print(duration, 's')
    vis.scatter(X_embedded, opts=dict(markersize=4, title=f'{batch_size}-{num_neighbors}:{duration}s'))
    X_embedded = K_coef @ X_embedded

    score_list = []
    neighbors = [10, 20, 40, 80]
    for n_neighbor in neighbors:
        neigh = KNeighborsClassifier(n_neighbors=n_neighbor)
        neigh.fit(X_embedded, y.ravel())
        score_list.append(np.round(neigh.score(X_embedded, y.ravel()), 3))
    vis.scatter(X_embedded, y + 1, opts=dict(markersize=4, title=f'{batch_size}-{num_neighbors}:{score_list}'))
    
    kmeans_model = KMeans(n_clusters=num_eigen, n_init='auto').fit(X)
    labels = kmeans_model.labels_
    silhouette = metrics.silhouette_score(X,labels)
    calinski = metrics.calinski_harabasz_score(X, labels) # the higher the score the more well defined the clusters are.
    davies = metrics.davies_bouldin_score(X, labels) # this score measures the similarity of your clusters, meaning that the lower the score the better separation there is between your clusters.
    vis.text(f'{batch_size}-{num_neighbors} | silhouette: {silhouette}; calinski: {calinski}; davies: {davies}')
    vis.save([ENV])

    print('-----end-----')


if __name__ == '__main__':
    beta = 1e-1
    opt_perplexity = 30
    dataname = 'HAR'
    proxy=False
    batch_size = -1
    num_eigen = 10 # HAR: 28; EEG: 10; WAV: 19
    num_neighbors = 'tsne'
    laplacian_TSNE(beta, batch_size, num_neighbors, opt_perplexity, dataname = dataname, n_components = 2, num_eigen = num_eigen, proxy = proxy)
