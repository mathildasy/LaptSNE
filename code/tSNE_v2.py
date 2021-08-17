# -*- coding: utf-8 -*-
# @author Yan SUN
# @email 119020045@link.cuhk.edu.cn
# @create date 2021-07-25 16:18:29
# @modify date 2021-07-25 16:18:30
# @desc [Revised version 2:
#       Objective function: KL(P||Q) + beta * tr(v'Lv)
#       Q = t-sne Q]


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;
from scipy.spatial import distance as dist
from tqdm import tqdm
from scipy.linalg import eigh
sns.set()

################ visualization function ################
def categorical_scatter_2d(X2D, class_idxs, title, ms=3, ax=None, alpha=0.1, legend=True, figsize=None, show=True, savename=None):
    ## Plot a 2D matrix with corresponding class labels: each class diff colour
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    classes = list(np.unique(class_idxs))
    markers = 'os' * len(classes)
    colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(X2D[class_idxs==cls, 0], X2D[class_idxs==cls, 1], marker=mark,
            linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
            markeredgecolor='black', markeredgewidth=0.4)
        ax.title.set_text(title)
    if legend:
        ax.legend()

    if savename is not None:
        plt.tight_layout()
        plt.savefig(savename)

    if show:
        plt.show()

    return ax

################  definitions: ################
def perplexity(distances, sigmas):

    """Wrapper function for quick calculation of
    perplexity over a distance matrix."""
    return calc_perplexity(calc_prob_matrix(distances, sigmas))

def softmax(X, diag_zero=True):
    """Take softmax of each row of matrix X."""

    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

    # We usually want diagonal probailities to be 0.
    if diag_zero:
        np.fill_diagonal(e_x, 0.)

    # Add a tiny constant for stability of log we take later
    e_x = e_x + 1e-8  # numerical stability

    return e_x / e_x.sum(axis=1).reshape([-1, 1])

################  calculation functions ################
def cal_euclidean_dis(df):
    '''
    input:
            df - a two dim array, (num of samples, num of features)
    output:
            dis - a two dim array, (num of sample, num of sample)
    '''
    dis = dist.pdist(df,'sqeuclidean') # Computes the squared Euclidean distance
    dis = dist.squareform(dis)

    return dis

def calc_perplexity(prob_matrix):
    """Calculate the perplexity of each row
    of a matrix of probabilities."""
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity

def calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)

################  sub-algorithms ################
def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, lower=1e-20, upper=1000.):
    """Perform a binary search over input values to eval_fn.

    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess

def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)

################  generate probablity ################
def p_conditional_to_joint(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + P.T) / (2. * P.shape[0])

def p_joint(X, target_perplexity):
    """Given a data matrix X, gives joint probabilities matrix.

    # Arguments
        X: Input data matrix.
    # Returns:
        P: Matrix with entries p_ij = joint probabilities.
    """
    # Get the squared euclidian distances matrix for our data
    distances = cal_euclidean_dis(X)
    # Find optimal sigma for each row of this distances matrix
    sigmas = find_optimal_sigmas(- distances, target_perplexity)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(- distances, sigmas)
    # Go from conditional to joint probabilities matrix
    P = p_conditional_to_joint(p_conditional)
    return P, sigmas

def q_tsne(Y):
    """
    t-SNE: Given low-dimensional representations Y,
    compute matrix of joint probabilities with entries q_ij.
    """
    distances = cal_euclidean_dis(Y)
    inv_distances = np.power(1. + distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances

def q_joint(Y):
    dis = dist.pdist(Y)
    dis = dist.squareform(dis)
    np.fill_diagonal(dis, 1e10)
    Q = np.power(dis, -1)
    np.fill_diagonal(Q, 0.)
    return Q, None

################  calculate gradients ################
def expand(X,n,d=2):
    n0 = X.shape[0]
    n1 = X.shape[1]
    return np.tile(X.reshape(n0,n1,1,1),(n,d))

def reform(X, d):
    assert X.shape[0] == X.shape[1]
    n = X.shape[0]
    A = list()
    for i in range(n):
        zeros = np.zeros_like(X)
        zeros[i,:] = np.ones_like(zeros[i,:])
        zeros[:,i] = - np.ones_like(zeros[:,i])
        A.append((zeros * X).reshape(-1,d))
    A = np.array(A).transpose(1,0,2)
    A = A.reshape(n,n,n,d)
    return A

def cal_gy_Q(Q, Y, inv_distances):
    n, d = Y.shape[0], Y.shape[1]
    diff_Y = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)
    # numerator part
    inv_dis_Y = np.expand_dims(inv_distances,2).repeat(d,axis=2) * diff_Y
    gy_Q = - reform(inv_dis_Y, d)

    # demoninator part
    gy_Q2 = (expand(Q * inv_distances, 1, d) * diff_Y.reshape(n,n,1,d)).sum(axis = 1)
    gy_Q = gy_Q + np.kron(Q.reshape(n,n,1),gy_Q2.reshape(n,d)).reshape(n,n,n,d)

    return gy_Q

def power_diag(D,power):
    D_new = np.diag(np.power(np.diag(D),power))
    return D_new

def eigen_grad(Q, Y, inv_distances, num_eigen,sigmas):
    '''
    calculate extra part of gradient w.r.t. Y:
    '''
    global lam_list

    print(' ------------Start eigen_grad ---------------')
    gy_Q = cal_gy_Q(Q, Y, inv_distances)
    n, d = Y.shape[0], Y.shape[1]
    D = np.diag(Q.sum(axis = 0))
    L = np.eye(n) - power_diag(D,-0.5) @ Q @ power_diag(D,-0.5)
    # scipy的方式较快
    lam, eig_V = eigh(L, subset_by_index=[0, num_eigen-1])

    U0 = -0.5 * power_diag(D,-1.5) @ Q @ power_diag(D,-0.5)
    U1 = -0.5 * power_diag(D,-0.5) @ Q @ power_diag(D,-1.5)
    ones = np.ones((n,1))
    U0_ = expand(((U0 * (eig_V @ eig_V.T)) @ ones) @ ones.T, n, d) * gy_Q
    U1_ = expand(ones @ (ones.T @ (U1 * (eig_V @ eig_V.T))), n, d) * gy_Q
    D_ =  expand((eig_V @ eig_V.T) * power_diag(D, -1), n, d) * gy_Q
    grad_Y = sum(sum(U0_ + U1_ + D_))

    print(f'----------- the {num_eigen} smallest eigenvalues ---------')
    lam_list.append(lam[:num_eigen])
    print(lam[:num_eigen])
    return grad_Y



def gk_grad(Q, Y, inv_distances, num_eigen,sigmas):
    '''
    calculate extra part of gradient w.r.t. Y:
    '''
    global lam_list
    def get_K(Y, sigmas):
        # sigma2 = np.mean(sigmas)
        sigma2 = 1e0
        eucdis = cal_euclidean_dis(Y)
        K = np.exp(-0.5 * eucdis / sigma2)
        K = K - np.eye(Y.shape[0]) # turn the similarity matrix into affinity matrix
        return K, sigma2

    def cal_gy_K(gK_L, Y, Kval, sigma2):
        sigma2 = 1e0
        Y = Y.T
        T = gK_L * Kval;
        C = np.tile(sum(T), (len(Y), 1));
        g = 2 / sigma2 * (Y @ T - Y * C).T;
        return g

    print(' ------------Start gk_grad ---------------')

    Kval,sigma2 = get_K(Y,sigmas)
    Q = Kval #### 这里准备计算 eigenvalue的时候应该是用Kval的，而不是Q

    n, d = Y.shape[0], Y.shape[1]
    D = np.diag(Q.sum(axis = 0))
    L = np.eye(n) - power_diag(D,-0.5) @ Q @ power_diag(D,-0.5)
    # scipy的方式较快
    lam, eig_V = eigh(L, subset_by_index=[0, num_eigen-1])

    U0 = -0.5 * power_diag(D,-1.5) @ Q @ power_diag(D,-0.5)
    U1 = -0.5 * power_diag(D,-0.5) @ Q @ power_diag(D,-1.5)
    grad_LK = (U0+U1+power_diag(D, -1)) * (eig_V @ eig_V.T)
    grad_Y = cal_gy_K(grad_LK, Y, Kval, sigma2)


    # print(f'----------- the {num_eigen} smallest eigenvalues ---------')
    lam_list.append(lam[:num_eigen])
    # print(lam[:num_eigen])
    return grad_Y


def tsne_grad(P, Q, eigen_fn,Y,  inv_distances, beta, num_eigen,sigmas):
    """
    Estimate the gradient of t-SNE cost with respect to Y.
    """
    pq_diff = P - Q
    pq_expanded = np.expand_dims(pq_diff, 2)
    y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)

    # Expand our inv_distances matrix so can multiply by y_diffs
    distances_expanded = np.expand_dims(inv_distances, 2)

    # Multiply this by inverse distances matrix
    y_diffs_wt = y_diffs * distances_expanded

    # Multiply then sum over j's
    if beta != 0:
        grad_Y2 = eigen_fn(Q, Y, inv_distances, num_eigen,sigmas)
        grad_Y = 4. * (pq_expanded * y_diffs_wt).sum(1) + beta * grad_Y2
    else:
        grad_Y = 4. * (pq_expanded * y_diffs_wt).sum(1)
    print('-------gradient of Y -------')
    print(grad_Y[0])
    return grad_Y

################  embedding algorithms ################
def init_y(sample_num, sigma, low_dim = 2, random_state = 0):
    '''
    input:
            sample_num - a scalar, the number of data
            sigma - a scalar, standard deviation of the initial Gaussian distribution
            low_dim - (2 by defualt) the low dimension that the data are projected onto
            random_state - (0 by default) the random seed
    output:
            y0 - a array, (num of samples, low_dim)
    '''

    np.random.seed(random_state)
    y0 = np.random.normal(0, sigma, size=(sample_num,low_dim))

    return y0


def estimate_sne(X, y, P, num_iters, q_fn, grad_fn, eigen_fn, learning_rate1, momentum, beta, num_eigen, plot,
                 exa_stage, lst_stage, rdseed, sigmas, exa_ratio):
    """Estimates a t-SNE model.
    # Arguments
        X: Input data matrix.
        P: Matrix of joint probabilities.
        num_iters: Iterations to train for.
        q_fn: Function that takes Y and gives Q prob matrix.
        plot: How many times to plot during training.
    # Returns:
        Y: Matrix, low-dimensional representation of X.
    """
    global lam_list

    sample_num = X.shape[0]
    sigma = 1e-2
    # Initialise our 2D representation
    Y = init_y(sample_num, sigma)
    # prepare lambda_list
    lam_list = list()

    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()
    # calculate the size of each update
    Y_update_num = int(rdseed * Y.shape[0])
    # Start gradient descent loop
    for i in tqdm(range(num_iters)):
        if i < exa_stage:
            P2 = exa_ratio * P
            # if i < 20:
            #     categorical_scatter_2d(Y, y,
            #                            title=f'Exaggerate Stage (No.{i + 1}; learning_rate1:{learning_rate1}, momentum:{momentum}, beta:{beta}, num_eigen:{num_eigen}, exa_stage:{exa_stage}, lst_stage:{lst_stage}',
            #                            alpha=1.0, ms=6,
            #                            show=True, figsize=(9, 6))
        else:
            P2 = P
        if i < num_iters - lst_stage:
            beta2 = 0
            print('beta = 0')
        else:
            beta2 = beta
            print('beta =', beta)
            categorical_scatter_2d(Y, y,
                                   title=f'Plus grad_Y2 (No.{i + 1}; learning_rate1:{learning_rate1}, momentum:{momentum}, beta:{beta}, num_eigen:{num_eigen}, exa_stage:{exa_stage}, lst_stage:{lst_stage}',
                                   alpha=1.0, ms=6,
                                   show=True, figsize=(9, 6))
        # choose part if Y for update according to rdseed
        choice = np.random.choice(range(Y.shape[0]), size=(Y_update_num,), replace=False)
        Y_update = Y[choice, :]
        P2 = P2[np.ix_(choice, choice)]
        Q, distances = q_fn(Y_update)
        grads_Y = grad_fn(P2, Q, eigen_fn, Y_update, distances, beta2, num_eigen, sigmas)

        # Update Y
        Y_update = Y_update - learning_rate1 * grads_Y
        # convert back to n*d
        Y[choice, :] = Y_update
        if momentum:  # Add momentum
            Y += momentum * (Y_m1 - Y_m2)
            # Update previous Y's for momentum
            Y_m2 = Y_m1.copy()
            Y_m1 = Y.copy()
        if plot and i % (num_iters / plot) == 0:
            categorical_scatter_2d(Y, y,
                                   title=f'No.{i + 1}; learning_rate1:{learning_rate1}, momentum:{momentum}, beta:{beta}, num_eigen:{num_eigen}, exa_stage:{exa_stage}, lst_stage:{lst_stage}',
                                   alpha=1.0, ms=6,
                                   show=True, figsize=(9, 6))
    categorical_scatter_2d(Y, y,
                           title=f'No.{i + 1} (end); learning_rate1:{learning_rate1}, momentum:{momentum}, beta:{beta}, num_eigen:{num_eigen}, exa_stage:{exa_stage}, lst_stage:{lst_stage}',
                           alpha=1.0, ms=6,
                           show=True, figsize=(9, 6))


    return Y, lam_list



