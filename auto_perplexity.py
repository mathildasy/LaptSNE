import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy.linalg import eigh
from tqdm import tqdm
import pandas as pd


EPSILON_DBL = 1e-8

def _h_beta_np(D, beta: float = 1.0, exa_ratio = 1):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    if sumP == 0.0:
        sumP = np.array([EPSILON_DBL]).sum()
        
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

# noinspection PyPep8Naming
def _x2p_np(X, tol: float = 1e-5, perplexity: float = 30.0, verbose: bool = False):
    """Binary search for sigmas of conditional Gaussians.

    Parameters
    ----------
    X: shape(n_samples, n_features)

    tol: float
        tolerance for optimization

    perplexity: float
        desired perplexity (2^entropy) of the conditional Gaussians
    
    verbose: bool
        verbosity level
    
    Returns
    ------
    P : array, shape (n_samples, n_samples)
        Probabilities of conditional Gaussian distributions p_i|j
        
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape
    D = pairwise_distances(X, metric='euclidean', squared=True)
    P = np.zeros((n, n), dtype = np.float64)
    beta = np.ones((n, 1),dtype = np.float64)
    logU = np.log(np.array([perplexity]), dtype = np.float64)
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    bar = range(n)
    if verbose:
        bar = tqdm(bar)

    for i in bar:
        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]
        (H, thisP) = _h_beta_np(Di, beta[i])
        # print(f'H is {H}')
        # Evaluate whether the perplexity is within tolerance
        h_diff = H - logU
        
        tries = 0
        while np.abs(h_diff) > tol and tries < 100:
            # If not, increase or decrease precision
            if h_diff > 0:
                betamin = beta[i].copy()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = _h_beta_np(Di, beta[i])

            h_diff = H - logU
            tries += 1
        # print(f'Total tries = {tries} with h_diff {h_diff}')
        
        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def opt_numEigen(X, maxNum = 30, perplex=30, verbose=True):
    P = _x2p_np(X, 1e-3, perplex, verbose)
    print(P.shape)

    D = np.diag(np.power(P.sum(axis = 1),-0.5))
    n = P.shape[0]
    L = np.eye(n) - D @ P @ D
    w, v = eigh(L, subset_by_index=[1, maxNum])
    eigenRatio = np.diff(w)/w[1:]
    if verbose: print('-- cal eigen ratio --')
    num_eigen = np.argmax(eigenRatio)+1

    # if verbose:
    #     k = len(eigenRatio)
    #     plt.plot(np.arange(1, k+1), eigenRatio)
    #     plt.vlines(x=num_eigen,
    #             ymin=0, ymax=np.max(eigenRatio), colors='red', linestyles='dashed')
    #     # plt.title(f'Optimal Perplexity is {opt_perplexity}')
    #     plt.show()

    return num_eigen

if __name__ == '__main__':
    import load_data
    X, y = load_data.EEG(-1,0)
    print(X.shape, y.shape)
    num_eigen = opt_numEigen(X, 30, 30)
    print(num_eigen)