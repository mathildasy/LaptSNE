import sys
import csv
import os
from typing import Union
import numpy as np
import torch
from time import time
from torch import nn
from scipy import linalg
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier,kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from visdom import Visdom
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import seaborn as sns


EPSILON_DBL = 1e-8

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
# noinspection PyPep8Naming

def _h_beta_torch(D: torch.Tensor, beta: float = 1.0, exa_ratio = 1):
    P = torch.exp(-D * beta)
    sumP = torch.sum(P)
    if sumP == 0.0:
        sumP = torch.Tensor([EPSILON_DBL]).sum()
        
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    return H.type(torch.float64), P.type(torch.float64)

# noinspection PyPep8Naming
def _x2p_torch(X: torch.Tensor, tol: float = 1e-5, perplexity: float = 30.0, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose: bool = False):
    """Binary search for sigmas of conditional Gaussians.

    Parameters
    ----------
    X: tensor-like, shape(n_samples, n_features)

    tol: float
        tolerance for optimization

    perplexity: float
        desired perplexity (2^entropy) of the conditional Gaussians

    device: torch.device
    
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

    sum_X = torch.sum(X*X, 1)
    # D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
    D = torch.cdist(X, X, p=2.0)

    P = torch.zeros(n, n, device=device, dtype = torch.float64)
    beta = torch.ones(n, 1, device=device, dtype = torch.float64)
    logU = torch.log(torch.tensor([perplexity], device=device, dtype = torch.float64))
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
        (H, thisP) = _h_beta_torch(Di, beta[i])
        # print(f'H is {H}')
        # Evaluate whether the perplexity is within tolerance
        h_diff = H - logU
        
        tries = 0
        while torch.abs(h_diff) > tol and tries < 100:
            # If not, increase or decrease precision
            if h_diff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = _h_beta_torch(Di, beta[i])

            h_diff = H - logU
            tries += 1
        # print(f'Total tries = {tries} with h_diff {h_diff}')
        
        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def _pca_torch(X, no_dims=50):
    # print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.linalg.eig(torch.mm(X.t(), X))
    l, M = l.double(), M.double()
    # split M real
    for i in range(d-1):
        if l[i] != 0:
            M[:, i+1] = M[:, i]
            i += 1
            
    Y = torch.mm(X, M[:, 0:no_dims])
    return Y

def _t_grad_torch(Y, num_eigen, beta, Kval, new_obj, lambda_list, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    def power_diag(D, power):
        D_new = torch.diag(torch.diag(D).pow(power))
        return D_new

    def cal_gk_L(A, coef):
        n = A.shape[0]
        D = torch.diag(A.sum(axis = 0))
        U0 = -0.5 * power_diag(D, -1.5) @ A @ power_diag(D, -0.5) * coef
        U1 = -0.5 * power_diag(D, -0.5) @ A @ power_diag(D, -1.5) * coef
        U0 = torch.tile(U0.sum(axis=0), (n, 1)).t()
        U1 = torch.tile(U1.sum(axis=1), (n, 1))
        U2 = torch.tile(power_diag(D, -0.5).sum(axis=0), (n, 1)).t() * torch.tile(power_diag(D, -0.5).sum(axis=1), (n, 1)) * coef
        grad_LK = -(U0 + U1 + U2)
        return grad_LK

    def cal_gy_K(gK_L, Y, Kval):
        Y = Y.t()
        T = gK_L * (Kval ** 2)
        C = torch.tile(T.sum(axis = 0), (len(Y), 1))
        g = 4 * (Y @ T - Y * C).t()
        return g

    # print(' ------------Start t_grad ---------------')

    n, d  = Y.shape
    D = torch.diag(Kval.sum(axis = 0))
    K_ = power_diag(D, -0.5) @ Kval @ power_diag(D, -0.5)
    K_ = K_.cpu().numpy()
    # eigenVectors,lam, _ = randomized_svd(K_, n_components = num_eigen, random_state= 0)
    lam, eigenVectors = linalg.eigh(K_, subset_by_index=[n - num_eigen - 1, n - 1])
    eigenVectors, lam = torch.from_numpy(eigenVectors).to(device), torch.from_numpy(lam).to(device)
    eigenGap = lam[-1] - lam[-2]; lam_Kplus1 = lam[-1]
    eigenGapRatio = eigenGap / lam_Kplus1
    if new_obj == 'firstK':
        eig_V = eigenVectors[:,1:]; eig_V_kp1 = eigenVectors[:, 0]
        lam_first = 1 - lam[1:-1]
        lam_coef = 1 / lam_first
        eig_V[:,:-1] = lam_coef ** (1/2) * eig_V[:,:-1]
        coef = beta * (eig_V @ eig_V.T)
    elif new_obj == 'gap':
        eig_V = eigenVectors[:,1]; eig_V_kp1 = eigenVectors[:,0]
        inv_trace1 = eigenGap ** (-1) * (eig_V_kp1 @ eig_V_kp1.T - eig_V @ eig_V.T)
        inv_trace2 = lam_Kplus1 ** (-1) * (eig_V_kp1 @ eig_V_kp1.T)
        coef = beta * (inv_trace1 - inv_trace2)
    elif new_obj == 'ratio':
        eig_V = eigenVectors[:,1:]; eig_V_kp1 = eigenVectors[:,0]
        inv_trace1 = (num_eigen - 1) * sum(lam[1:]) ** (-1) * (eig_V @ eig_V.T)
        inv_trace2 = lam_Kplus1 ** (-1) * (eig_V_kp1 @ eig_V_kp1.T)
        coef = beta * (inv_trace2 - inv_trace1)
    else:
        raise ValueError("'new_obj' must be 'first K', 'gap' or 'ratio")

    grad_LK = cal_gk_L(Kval, coef)
    grad_Y = cal_gy_K(grad_LK, Y, Kval).ravel()
    lambda_list.append(lam.cpu().numpy())
    error = beta * (num_eigen - sum(lam[1:]))
    return error, grad_Y, lambda_list

# noinspection PyPep8Naming




class TorchTSNE:
    def __init__(
            self,
            perplexity: float = 30.0,
            n_iter: int = 1000,
            n_components: int = 2,
            initial_dims: int = 50,
            verbose: bool = False,
            kl_num = 0,
            beta = 1e-2,
            laststage = 20,
            lastcoef = 2,
            num_eigen = 10,
            lr = 1e2,
            out_dir = None,
            PCA_need = False,
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.n_components = n_components
        self.initial_dims = initial_dims
        self.verbose = verbose
        self.beta = beta
        self.num_eigen = num_eigen
        self.lr = lr
        self.kl_num = kl_num
        self.laststage = laststage
        self.lastcoef = lastcoef
        self.out_dir = out_dir
        self.device = device
        self.PCA_need = PCA_need

    def _tsne(self, X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray], 
        vis, P_precomputed: torch.Tensor, no_dims: int = 2, initial_dims: int = 50, 
        perplexity: float = 30.0,max_iter: int = 1000, verbose: bool = False, 
        beta: float = 1e-2, num_eigen: int = 10, PCA_need: bool = False, 
        lr: float = 1e2, kl_num = 20, laststage = 20, lastcoef = 2, 
        n_iter = 1000, knn_type = 'small', output_dir = None, 
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
            Runs t-SNE on the dataset in the NxD array X to reduce its
            dimensionality to no_dims dimensions. The syntaxis of the function is
            `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
        """

        if not isinstance(no_dims, int) or no_dims <= 0:
            raise ValueError("dims must be positive integer")
        if isinstance(X, np.ndarray):
            X = torch.tensor(X)

        if verbose:
            print(f"using {device}", file=sys.stderr)
        X = X.to(device)

        if verbose:
            print("initializing...", file=sys.stderr)
        # Initialize variables
        if (initial_dims < X.shape[1]) & PCA_need:
            print('Start PCA ====>')
            X = _pca_torch(X, initial_dims)
        elif verbose:
            print("skipping PCA", file=sys.stderr)
        (n, d) = X.shape
        print(n,d)
        initial_momentum = 0.5
        final_momentum = 0.8
        eta = lr
        min_gain = 0.01

        # P = P_precomputed.to(device)
        # Compute P-values
        P = _x2p_torch(X, 1e-5, perplexity, device = device, verbose = True)
        P = P + P.t()
        P = P / torch.sum(P)
        P = P * 4.    # early exaggeration
        print("get P shape", P.shape)
        P = torch.max(P, torch.tensor([1e-21]).to(device))

        def power_diag(D, power):
            D_new = torch.diag(torch.diag(D).pow(power))
            return D_new
        D = torch.diag(P.sum(axis=1))
        L = torch.eye(D.shape[0]).to(device) - power_diag(D, -0.5) @ P @ power_diag(D, -0.5)
        _, X_embedded = linalg.eigh(L.cpu().numpy(), subset_by_index=[1, no_dims])

        #Y = torch.randn(n, no_dims, device=device)
        Y = torch.from_numpy(X_embedded).to(device)

        # print('--------Calculate K-Means Score-------')
        score = 0
        num_class = len(np.unique(y.ravel))
        for t in range(5):
            y_pred = KMeans(n_clusters=num_class, random_state=t, n_init='auto').fit_predict(X_embedded)
            score += acc(y.ravel(), y_pred)
        score = score/5
        kMeans_report = ['initial', score]
        output2_report = os.path.join(output_dir, f'kMeans_report.csv')
        with open(output2_report, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(kMeans_report)

        
        # Spectral initialization
        iY = torch.zeros(n, no_dims, device=device)
        gains = torch.ones(n, no_dims, device=device)


        if verbose:
            print("fitting...", file=sys.stderr)

        bar = range(max_iter)

        if verbose:
            bar = tqdm(bar)
        lambda_list = list()
        time_output = list()
        tic = time()
        for it in bar:
            # Compute pairwise affinities
            sum_Y = torch.sum(Y * Y, dim=1)
            num = -2. * torch.mm(Y, Y.t())  # (N, N)
            num = 1. / (1. + (num + sum_Y).t() + sum_Y)
            num.fill_diagonal_(0)
            Q = num / torch.sum(num)
            Q = torch.max(Q, torch.tensor(1e-12, device=Q.device))
            # Compute gradient
            PQ = P - Q
            # ((N, 1, N).repeat -> (N, no_dims, N).permute -> (N, N, no_dims)) *
            # ((N, no_dims) - (N, 1, no_dims) -> (N, N, no_dims))
            # -> (N, N, no_dims).sum[1] -> (N, no_dims)
            dY = torch.sum(
                (PQ * num).unsqueeze(1).repeat(1, no_dims, 1).transpose(2, 1) * (Y.unsqueeze(1) - Y),
                dim=1
            )
            
            ######### New gradient ##########
            if (it > kl_num):
                if beta > 0:
                    error, dY2, lambda_list = _t_grad_torch(Y, num_eigen, beta, num, 'firstK', lambda_list, device = device)
                    dY2 = dY2.reshape(-1, no_dims)
                    if (it + laststage >= n_iter):
                        # print('-------------- last stage coef adjusted! -----------------')
                        dY += lastcoef * dY2
                    else:
                        dY += dY2
                    vis.line(np.array([lambda_list[-1]]), np.array([it]), win='Lambda', update='append',opts=dict(title='Lambda Curve'))
                    

            # print(f'Iter {it}, error: {error}')
            # Perform the update
            # if it < 20:
            #     momentum = initial_momentum
            # else:
            #     momentum = final_momentum
            momentum = initial_momentum
            gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).float() + (gains + 0.8) * ((dY > 0.) == (iY > 0.)).float()
            gains[gains < min_gain] = min_gain
            iY = momentum * iY - eta * (gains * dY)
            Y = Y + iY
            Y = Y - torch.mean(Y, 0)

            # Stop lying about P-values
            # if it == 100:
            #     P = P / 4.
            X_embedded = Y.reshape(-1, no_dims).cpu().numpy()
            if (it % 20 == 0) | (it + 1 >= n_iter):
                # KNN score
                # print('--------Calculate K-NN Score-------')
                score_list = []
                if knn_type == 'small':
                    neighbors = [10, 20, 40, 80, 160]
                else:
                    neighbors = [100, 200, 400, 800, 1600]
                for n_neighbor in neighbors:
                    neigh = KNeighborsClassifier(n_neighbors = n_neighbor)

                    neigh.fit(X_embedded, y.ravel())
                    score_list.append(neigh.score(X_embedded, y.ravel()))
                knn_report = [it,]+ score_list
                '''
                example:
                10, 0.9123, 0.8923, 0.8452, 0.8213
                20, 0.9321, 0.9023, 0.8762, 0.8533
                '''
                output_report = os.path.join(output_dir, f'knn_report.csv')
                with open(output_report, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(knn_report)

                # print('--------Calculate K-Means Score-------')
                score = 0
                for t in range(5):
                    y_pred = KMeans(n_clusters=num_eigen, random_state=t, n_init='auto').fit_predict(X_embedded)
                    score += acc(y.ravel(), y_pred)
                score = score/5
                kMeans_report = [it, score]
                output2_report = os.path.join(output_dir, f'kMeans_report.csv')
                with open(output2_report, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(kMeans_report)

                vis.line(np.array([score_list]), np.array([it]), win='KNN-Score',update='append',opts=dict(title='Score Curve'))
                vis.line(np.array([score]), np.array([it]), win='KMeans-Score',update='append',opts=dict(title='Score Curve'))
            
            # Compute current value of cost function
            if verbose:
                bar.set_description(f"KMeans score: {score:.3f}")
            
            if (it % 10 == 0) & (no_dims < 3):
                vis.scatter(X_embedded, y + 1, opts=dict(markersize=4, title=f'T-SNE at step{it}'))
                # save .npy
                embedding_file = os.path.join(output_dir, f'X_embedded/embedding_{it}.npy')
                np.save(embedding_file, X_embedded)
                # save .eps
                image_file = os.path.join(output_dir, f'X_embedded/image/embedding_{it}.eps')
                fig = plt.figure(dpi=1200)
                sns.scatterplot(x = X_embedded[:,0], y = X_embedded[:,1], c = y + 1)
                plt.title(f'T-SNE at step {it}')
                fig.savefig(image_file, format='eps', dpi=1200)
                plt.clf()
                
            toc = time()
            duration = toc - tic
            tic = toc
            vis.line(np.array([duration]), np.array([it]), win='Duration_i',update='append',opts=dict(title='Time Curve'))
            time_output.append(duration)


        # savings 
        # lambda_list
        time_report = os.path.join(output_dir, f'time_report.csv')
        torch.save(time_output, time_report)
        lambda_report = os.path.join(output_dir, f'lambda_report.csv')
        torch.save(lambda_list, lambda_report)
        # Return solution
        return Y.detach().cpu().numpy()

    # noinspection PyPep8Naming,PyUnusedLocal
    def fit_transform(self, X, P_precomputed, vis, y=None):
        """
        Learns the t-stochastic neighbor embedding of the given data.

        :param X: ndarray or torch tensor (n_samples, *)
        :param y: ignored
        :param P_precomputed: torch tensor (n_samples, n_samples)
        :return: ndarray (n_samples, n_components)
        """
        if X.shape[0] <= 10000:
            knn_type = 'small'
        else:
            knn_type = 'large'

        with torch.no_grad():
            return self._tsne(
                X,
                y,
                vis,
                P_precomputed = P_precomputed,
                no_dims=self.n_components,
                initial_dims=self.initial_dims,
                perplexity=self.perplexity,
                verbose=self.verbose,
                max_iter=self.n_iter,
                beta = self.beta,
                num_eigen= self.num_eigen,
                lr = self.lr,
                kl_num= self.kl_num,
                laststage = self.laststage,
                lastcoef = 2,
                n_iter = self.n_iter,
                knn_type = knn_type,
                output_dir = self.out_dir,
                device = self.device,
                PCA_need = self.PCA_need
            )
