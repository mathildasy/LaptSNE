#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:mathilda 
@Email: 119020045@link.cuhk.edu.com
@file: load_data.py
@time: 2021/08/21

Notes: a complication of data, return: X, y


"""

from sklearn import datasets
import numpy as np
import scipy.io

PATH = './data/'


# fmnist = sklearn.datasets.fetch_openml('Fashion-MNIST')

def iris(num_samples=-1, seed=0):
    data = datasets.load_iris()
    if num_samples >= 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(len(data)), size=num_samples)
        X = data['data'][select]
        y = data['target'][select]
    else:
        X, y = data['data'], data['target']

    if num_samples < 0: num_samples = X.shape[0]
    print(f'--------Finish loading iris dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y


def digits(num_samples=-1, seed=0):
    data = datasets.load_digits()
    if num_samples >= 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(len(data)), size=num_samples)
        X = data['data'][select]
        y = data['target'][select]
    else:
        X, y = data['data'], data['target']

    if num_samples < 0: num_samples = X.shape[0]
    print(f'--------Finish loading digits dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y


def MNIST(num_samples = -1, seed = 0):
    X,y = np.load(PATH + 'MNIST_X.npy'), np.load(PATH + 'MNIST_y.npy')
    # if num_samples >=0:
    #     np.random.seed(seed)
    #     select = np.random.choice(np.arange(X.shape[0]), size = num_samples)
    #     X = X[select]
    #     y = y[select]

    # if num_samples < 0: num_samples = X.shape[0]
    X, y = X[-10000:,:], y[-10000:]
    print(f'--------Finish loading MNIST dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y

def Fashion_MNIST(num_samples = -1, seed = 0):
    X,y = np.load(PATH + 'Fashion-MNIST_X.npy'), np.load(PATH + 'Fashion-MNIST_y.npy')
    # if num_samples >=0:
    #     np.random.seed(seed)
    #     select = np.random.choice(np.arange(X.shape[0]), size = num_samples)
    #     X = X[select]
    #     y = y[select]

    # if num_samples < 0: num_samples = X.shape[0]
    X, y = X[-10000:,:], y[-10000:]
    print(f'--------Finish loading Fashion MNIST dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y


def COIL20(num_samples=-1, seed=0):
    X, y = np.load(PATH + 'COIL20_X.npy'), np.load(PATH + 'COIL20_y.npy')
    if num_samples >= 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(X.shape[0]), size=num_samples)
        X = X[select]
        y = y[select]

    if num_samples < 0: num_samples = X.shape[0]
    print(f'--------Finish loading COIL20 dataset [size: {num_samples}, shape: {X.shape}]--------')
    # y = np.array([int(i) for i in y])
    return X, y


def COIL100(num_samples=-1, seed=0):
    data = scipy.io.loadmat(PATH + 'COIL100.mat')
    if num_samples > 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(data['Label'].shape[0]), size=num_samples)
        X = data['X'].T[select]
        y = data['Label'][select]

    else:
        X, y = data['X'].T, data['Label']

    if num_samples < 0: num_samples = X.shape[0]
    print(X.shape, y.shape)
    # print(np.unique(y)) # 100
    print(f'--------Finish loading COIL100 dataset [size: {num_samples}, shape: {X.shape}]--------')
    return X, y.ravel()

def CIFAR100(num_samples=-1, seed=0):
    X, y = np.load(PATH + 'CIFAR100_X.npy'), np.load(PATH + 'CIFAR100_y.npy')
    if num_samples >= 0:
        np.random.seed(seed)
        select = np.random.choice(np.arange(X.shape[0]), size=num_samples)
        X = X[select]
        y = y[select]

    if num_samples < 0: num_samples = X.shape[0]
    print(f'--------Finish loading CIFAR100 dataset [size: {num_samples}, shape: {X.shape}]--------')
    # y = np.array([int(i) for i in y])
    return X, y

