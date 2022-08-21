import numpy as np
import sklearn
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from numba import jit


@jit(fastmath=True, parallel=True)
def power_diag(D, power):
    D_new = np.diag(np.power(np.diag(D), power))
    return D_new

@jit(fastmath=True, parallel=True)
def get_Q_gaussian(eucdis, sigman):
    K = np.exp(-0.5 * eucdis / sigman)
    K = K - np.eye(eucdis.shape[0])  # turn the similarity matrix into affinity matrix
    Q = 0.5 * (K + K.T)
    return Q

@jit(fastmath=True, parallel=True)
def get_Q_tStudent(eucdis, degrees_of_freedom):
    eucdis /= degrees_of_freedom
    eucdis += 1.
    Q = np.power(eucdis, (degrees_of_freedom + 1.0) / -2.0)
    return Q


@jit(fastmath=True, parallel=True)
def cal_coef_first(eigenVectors, lam, new_obj, num_eigen):
    eig_V = eigenVectors[:, 1:]
    lam_first = (1 - lam[1:])[::-1]
    lam_coef = np.true_divide(1, lam_first)
    eig_V= np.multiply(np.power(lam_coef, 0.5), eig_V)
    return eig_V


def cal_coef(eigenVectors, lam, new_obj, num_eigen):

    if new_obj == 'gap':
        eigenGap = lam[-1] - lam[-2]
        lam_Kplus1 = lam[-1]
        eig_V = eigenVectors[:, 1]
        eig_V_kp1 = eigenVectors[:, 0]
        inv_trace1 = np.multiply(np.power(eigenGap,-1), (np.matmul(eig_V_kp1, eig_V_kp1.T) - np.matmul(eig_V, eig_V.T)))
        inv_trace2 = np.multiply(np.power(lam_Kplus1,-1), (np.matmul(eig_V_kp1, eig_V_kp1.T)))
        coef = (inv_trace1 - inv_trace2)
    elif new_obj == 'firstK':
        eig_V = eigenVectors[:, 1:]
        lam_first = 1 - lam[1:-1]
        lam_coef = 1 / lam_first
        eig_V[:, :-1] = np.multiply(np.power(lam_coef, 0.5), eig_V[:, :-1])
        coef = np.matmul(eig_V, eig_V.T)
    elif new_obj == 'ratio':
        eig_V = eigenVectors[:, 1:]
        eig_V_kp1 = eigenVectors[:, 0]
        inv_trace1 = np.multiply(np.power((num_eigen - 1) * sum(lam[1:]), -1), np.matmul(eig_V, eig_V.T))
        inv_trace2 =  np.multiply(np.power(lam_Kplus1,-1) * np.matmul(eig_V_kp1, eig_V_kp1.T))
        coef = (inv_trace2 - inv_trace1)
    else:
        raise ValueError("'new_obj' must be 'gap', 'ratio' or 'firstK'")

    return coef


def cal_gq_Q(Q, coef):
    '''
    @ partial derivative of Q w.r.t. q
    input:
        A: required shape: (n, n)
        coef: coefficient part of the entire gradient (derived from eigenvectors): (n,n)
    output:
        grad_Q: shape (n,n)
    '''

    n = Q.shape[0]
    D_diag = Q.sum(axis=0)  # column sum
    # D_05 = np.diag(np.power(D_diag, -0.5))
    # D_15 = np.diag(np.power(D_diag, -1.5))
    D_05 = np.power(D_diag, -0.5)
    D_15 = np.power(D_diag, -1.5)
    D_05_tile = np.tile(D_05.sum(axis=0), (n, 1))
    # U0 = -0.5 * np.matmul(np.matmul(D_15, Q), D_05) * coef
    # U1 = -0.5 * np.matmul(np.matmul(D_05, Q), D_15) * coef
    U0 = -0.5 * np.tile(D_15,(n,1)).T * Q * np.tile(D_05,(n,1)) * coef
    U1 = -0.5 * np.tile(D_05,(n,1)).T * Q * np.tile(D_15,(n,1)) * coef    ## change matrix multiplication to dot multiplication
    U0 = np.tile(U0.sum(axis=0), (n, 1)).T
    U1 = np.tile(U1.sum(axis=1), (n, 1))
    U2 = D_05_tile.T * D_05_tile * coef
    grad_Q = -(U0 + U1 + U2)
    return grad_Q


def cal_gy_Q_gaussian(gq_L, Y, Q):
    '''
    @ partial derivative of Q w.r.t. y under Gaussian kernel
    input:
        gq_L: coefficient from the previous parital devirative: (n, n)
        Y: required shape: (n, d)
        Q: weighted adjacency matrix in lower dimension: (n, n)
    output:
        g: shape (n,d)
    '''
    d = Y.shape[1]
    T = gq_L * Q  # (n,n)
    C = np.tile(np.sum(T, axis=0), (d, 1)).T
    g = np.matmul((T + T.T), Y) - Y * C  # (n,2)
    return g


def cal_gy_Q_tStudent(gq_L, Y, Q):
    '''
    @ partial derivative of Q w.r.t. Y under T-Student kernel
    input:
        gq_L: coefficient from the previous parital devirative: (n, n)
        Y: required shape: (n, d)
        Q: weighted adjacency matrix in lower dimension: (n, n)
    output:
        g: shape (n,d)
    '''

    Y = Y.T
    T = gq_L * (Q ** 2)
    C = np.tile(np.sum(T, axis=0), (len(Y), 1))
    g = 4 * (np.matmul(Y,T) - Y * C).T
    return g


def t_grad(Y, num_eigen, beta, new_obj = 'firstK', degrees_of_freedom=2, skip_decompose=False, eig_V=None):
    '''
    @ derivative of Laplacian eigentrace w.r.t. Y under T-Student kernel
    input:
        Y: required shape: (n, d) #TODO: reduce into mini-batch (m, d), batch_size=m
        num_eigen: hidden/required number of clusters
        beta: relative weight in the entire loss funciton
        P: 
    output:
        g: shape (n,d)
    '''

    eucdis = pdist(Y, 'sqeuclidean')
    Q = get_Q_tStudent(eucdis, degrees_of_freedom)
    Q = squareform(Q)
    print('###### the actual Q ######')
    print(Q)
    if not skip_decompose: 
        D_diag = Q.sum(axis=0)
        D_05 = np.diag(np.power(D_diag, -0.5))
        L = np.matmul(np.matmul(D_05, Q), D_05)
        eigenVectors, lam, _ = sklearn.utils.extmath.randomized_svd(L, n_components=num_eigen, random_state=0)
        # coef = cal_coef(eigenVectors, lam, new_obj, num_eigen)
        eig_V = cal_coef_first(eigenVectors, lam, new_obj, num_eigen)
    
    coef = np.matmul(eig_V, eig_V.T)
    coef *= beta
    
    grad_Q = cal_gq_Q(Q, coef)
    grad_Y = cal_gy_Q_tStudent(grad_Q, Y, Q)

    return grad_Y, eig_V

    
def g_grad(Y, num_eigen, beta, new_obj = 'firstK', n_neighbors = 10, sigman_type = 'constant', skip_decompose=False, eigenVectors=None, lam=None):
    '''
    @ derivative of Laplacian eigentrace w.r.t. Y under Gaussian kernel
    input:
        Y: required shape: (n, d)
        num_eigen: hidden/required number of clusters
        beta: relative weight in the entire loss funciton
        P: 
    output:
        g: shape (n,d)
    '''
    
    eucdis = sklearn.metrics.pairwise.pairwise_distances(Y) ** 2
    n = Y.shape[0]
    if sigman_type == 'constant':
        sigman = sigman
    else:
        M = kneighbors_graph(Y, n_neighbors, mode='connectivity', include_self=False).toarray()
        sigman = np.tile((eucdis * M).mean(axis=1), (n, 1)).T

    eucdis = sklearn.metrics.pairwise.pairwise_distances(Y) ** 2
    Q = get_Q_gaussian(eucdis, sigman)
    D = np.diag(Q.sum(axis=0))
    L = power_diag(D, -0.5) @ Q @ power_diag(D, -0.5)
    eigenVectors, lam, _ = sklearn.utils.extmath.randomized_svd(L, n_components=num_eigen, random_state=0)

    coef = cal_coef(eigenVectors, lam, new_obj, num_eigen)
    grad_Q = cal_gq_Q(Q, coef)
    grad_Y = cal_gy_Q_gaussian(grad_Q, Y, Q)
    error = beta * (num_eigen - sum(lam[1:]))

    return error, grad_Y, lam


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                            assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    return grp, L


def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _ = post_proC(C, K, d, ro)
    return y