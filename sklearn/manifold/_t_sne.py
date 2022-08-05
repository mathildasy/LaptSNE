# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf


import csv
import os
import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from ..neighbors import NearestNeighbors, KNeighborsClassifier, kneighbors_graph
from ..cluster import KMeans
from .. import metrics
from ..base import BaseEstimator
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.validation import check_non_negative
from ..utils.validation import _deprecate_positional_args
from ..decomposition import PCA
from ..metrics.pairwise import pairwise_distances
# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from . import _utils  # type: ignore
# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from . import _barnes_hut_tsne  # type: ignore
from scipy.linalg import eigh
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from ..utils.extmath import randomized_svd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.
    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose)
    # P = squareform(conditional_P + conditional_P.T) * 0.5
    P = (conditional_P + conditional_P.T)
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    print(f'P is normalized with shape: {P.shape}')
    return P


def _joint_probabilities_nn(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances using just nearest
    neighbors.
    This method is approximately equal to _joint_probabilities. The latter
    is O(N), but limiting the joint probability to nearest neighbors improves
    this substantially to O(uN).
    Parameters
    ----------
    distances : sparse matrix of shape (n_samples, n_samples)
        Distances of samples to its n_neighbors nearest neighbors. All other
        distances are left to zero (and are not materialized in memory).
        Matrix should be of CSR format.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.
    Returns
    -------
    P : sparse matrix of shape (n_samples, n_samples)
        Condensed joint probability matrix with only nearest neighbors. Matrix
        will be of CSR format.
    """
    t0 = time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    distances.sort_indices()
    n_samples = distances.shape[0]
    distances_data = distances.data.reshape(n_samples, -1)
    distances_data = distances_data.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances_data, desired_perplexity, verbose)
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    # Symmetrize the joint probability distribution using sparse operations
    P = csr_matrix((conditional_P.ravel(), distances.indices,
                    distances.indptr),
                   shape=(n_samples, n_samples))
    P = P + P.T

    # Normalize the joint probability distribution
    sum_P = np.maximum(P.sum(), MACHINE_EPSILON)
    P /= sum_P

    assert np.all(np.abs(P.data) <= 1.0)
    if verbose >= 2:
        duration = time() - t0
        print("[t-SNE] Computed conditional probabilities in {:.3f}s"
              .format(duration))
    return P


def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components,
                   skip_num_points=0, compute_error=True):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.
    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    # Q is a heavy-tailed distribution: Student's t-distribution
    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Optimization trick below: np.dot(x, y) is faster than
    # np.sum(x * y) because it calls BLAS

    # Objective: C (Kullback-Leibler divergence of P and Q)
    kl_divergence = 2.0 * np.dot(
        P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    # pdist always returns double precision distances. Thus we need to take
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c
    # print(f'======== KL divergence is {kl_divergence} ========')
    return kl_divergence, grad, dist


@_deprecate_positional_args
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric='euclidean'):
    r"""Expresses to what extent the local structure is retained.
    The trustworthiness is within [0, 1]. It is defined as
    .. math::
        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.
    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.
    X_embedded : ndarray of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.
    n_neighbors : int, default=5
        Number of neighbors k that will be considered.
    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, see the
        documentation of argument metric in sklearn.pairwise.pairwise_distances
        for a list of available metrics.
        .. versionadded:: 0.20
    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.
    """
    dist_X = pairwise_distances(X, metric=metric)
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = NearestNeighbors(n_neighbors=n_neighbors).fit(
        X_embedded).kneighbors(return_distance=False)

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    n_samples = X.shape[0]
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis],
                   ind_X] = ordered_indices[1:]
    ranks = inverted_index[ordered_indices[:-1, np.newaxis],
                           ind_X_embedded] - n_neighbors
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (2.0 / (n_samples * n_neighbors *
                          (2.0 * n_samples - 3.0 * n_neighbors - 1.0)))
    return t


def categorical_scatter_2d(vis, X2D, class_idxs, title):
    vis.scatter(X2D, class_idxs + 1, opts=dict(markersize=4, title=title))


def _kl_divergence_bh(params, P, degrees_of_freedom, n_samples, n_components,
                      angle=0.5, skip_num_points=0, verbose=False,
                      compute_error=True, num_threads=1):
    """t-SNE objective function: KL divergence of p_ijs and q_ijs.
    Uses Barnes-Hut tree methods to calculate the gradient that
    runs in O(NlogN) instead of O(N^2).
    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.
    P : sparse matrix of shape (n_samples, n_sample)
        Sparse approximate joint probability matrix, computed only for the
        k nearest-neighbors and symmetrized. Matrix should be of CSR format.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    angle : float, default=0.5
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.
    verbose : int, default=False
        Verbosity level.
    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.
    num_threads : int, default=1
        Number of threads used to compute the gradient. This is set here to
        avoid calling _openmp_effective_n_threads for each gradient step.
    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    params = params.astype(np.float32, copy=False)
    X_embedded = params.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)

    grad = np.zeros(X_embedded.shape, dtype=np.float32)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c

    return error, grad


class TSNE(BaseEstimator):
    """t-distributed Stochastic Neighbor Embedding.
    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.
    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].
    Read more in the :ref:`User Guide <t_sne>`.
    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.
    init : {'random', 'pca'} or ndarray of shape (n_samples, n_components), \
            default='random'
        Initialization of embedding. Possible options are 'random', 'pca',
        and a numpy array of shape (n_samples, n_components).
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term: `Glossary <random_state>`.
    method : str, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.
        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.
    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.
    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
        .. versionadded:: 0.22
    square_distances : True or 'legacy', default='legacy'
        Whether TSNE should square the distance values. ``'legacy'`` means
        that distance values are squared only when ``metric="euclidean"``.
        ``True`` means that distance values are squared for all metrics.
        .. versionadded:: 0.24
           Added to provide backward compatibility during deprecation of
           legacy squaring behavior.
        .. deprecated:: 0.24
           Legacy squaring behavior was deprecated in 0.24. The ``'legacy'``
           value will be removed in 1.1 (renaming of 0.26), at which point the
           default value will change to ``True``.
    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        Stores the embedding vectors.
    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    References
    ----------
    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/
    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf
    """
    # Control the number of exploration iterations with early_exaggeration on

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    @_deprecate_positional_args
    def __init__(self, n_components=2, *, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0, kernel='Gaussian',
                 random_state=None, method='exact', angle=0.5,
                 n_jobs=None, square_distances='legacy', vis=None, label=None,
                 lst_stage=5, num_eigen=0, beta=1e0, exag_stage=250,
                 kl_num=20, new_obj='firstK', obj_F=None, lastcoef=2,
                 opt_method='momentum', knn_type='small', sigman_type='constant', sigman_constant=1e1,
                 P_eigen=False, proxy=False, num_proxies=2000, out_dir=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.method = method
        self.angle = angle
        self.n_jobs = n_jobs
        self.square_distances = square_distances
        self.vis = vis
        self.y = label
        self.lst_stage = lst_stage
        self.num_eigen = num_eigen
        self.beta = beta
        self._EXPLORATION_N_ITER = exag_stage
        self.kernel = kernel
        self.lastcoef = lastcoef
        self.kl_num = kl_num
        self.new_obj = new_obj
        self.obj_F = obj_F
        self.opt_method = opt_method
        self.knn_type = knn_type
        self.sigman_type = sigman_type
        self.sigman = sigman_constant
        self.P_eigen = P_eigen
        self.proxy = proxy,
        self.num_proxies = num_proxies
        self.output_dir = out_dir

    def _gradient_descent(self, objective, p0, it, n_iter, X=None,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
        """Batch gradient descent with momentum and individual gains.
        Parameters
        ----------
        objective : callable
            Should return a tuple of cost and gradient for a given parameter
            vector. When expensive to compute, the cost can optionally
            be None and can be computed every n_iter_check steps using
            the objective_error function.
        p0 : array-like of shape (n_params,)
            Initial parameter vector.
        it : int
            Current number of iterations (this function will be called more than
            once during the optimization).
        n_iter : int
            Maximum number of gradient descent iterations.
        n_iter_check : int, default=1
            Number of iterations before evaluating the global error. If the error
            is sufficiently low, we abort the optimization.
        n_iter_without_progress : int, default=300
            Maximum number of iterations without progress before we abort the
            optimization.
        momentum : float within (0.0, 1.0), default=0.8
            The momentum generates a weight for previous gradients that decays
            exponentially.
        learning_rate : float, default=200.0
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers.
        min_gain : float, default=0.01
            Minimum individual gain for each parameter.
        min_grad_norm : float, default=1e-7
            If the gradient norm is below this threshold, the optimization will
            be aborted.
        verbose : int, default=0
            Verbosity level.
        args : sequence, default=None
            Arguments to pass to objective function.
        kwargs : dict, default=None
            Keyword arguments to pass to objective function.
        Returns
        -------
        p : ndarray of shape (n_params,)
            Optimum parameters.
        error : float
            Optimum.
        i : int
            Last iteration.
        """
        global adam_m, adam_v, K_coef

        def gk_grad(Y, num_eigen, lambda_list, new_obj, P, min_k=False, UPDATE_SIGMA=False):
            '''
            calculate extra part of gradient w.r.t. Y:
            '''
            global lam_list

            def power_diag(D, power):
                D_new = np.diag(np.power(np.diag(D), power))
                return D_new

            def get_Q(Y, sigman):
                eucdis = pairwise_distances(Y) ** 2
                K = np.exp(-0.5 * eucdis / sigman)
                K = K - np.eye(Y.shape[0])  # turn the similarity matrix into affinity matrix
                Q = 0.5 * (K + K.T)
                return Q

            def cal_gy_K(gK_L, Y, Q, sigman):
                '''
                input:
                    Y: required shape: (n, d)
                output:
                    g: shape (n,d)
                '''
                n, d = Y.shape[0], Y.shape[1]
                Kval = Q / np.tile(sigman, (n, 1))
                T = gK_L * Kval  # (n,n)
                C = np.tile(sum(T), (d, 1)).T
                g = (T + T.T) @ Y - Y * C  # (n,2)
                return g

            def cal_gk_L(A, coef, M):
                n = A.shape[0]
                D = np.diag(A.sum(axis=0))  # column sum
                U0 = -0.5 * power_diag(D, -1.5) @ A @ power_diag(D, -0.5) * coef;
                U0 = np.tile(U0.sum(axis=0), (n, 1)).T
                U1 = -0.5 * power_diag(D, -0.5) @ A @ power_diag(D, -1.5) * coef;
                U1 = np.tile(U1.sum(axis=1), (n, 1))
                U2 = np.tile(power_diag(D, -0.5).sum(axis=0), (n, 1)).T * np.tile(power_diag(D, -0.5).sum(axis=1),
                                                                                  (n, 1)) * coef
                grad_LK = -(U0 + U1 + U2) * M
                return grad_LK

            print(' ------------Start gk_grad ---------------')
            M = kneighbors_graph(Y, self.kneigh, mode='connectivity', include_self=False)
            M = M.toarray()
            # eucdis = pairwise_distances(Y) ** 2
            n = Y.shape[0]

            # TODO: Search for asymmtric sigma supports
            if self.sigman_type == 'constant':
                sigman = self.sigman
            else:
                sigman = np.tile((eucdis * M).mean(axis=1), (n, 1)).T
            Q = get_Q(Y, sigman)
            D = np.diag(Q.sum(axis=0))
            A_normalized = power_diag(D, -0.5) @ Q @ power_diag(D, -0.5)

            lam, eigenVectors = linalg.eigh(A_normalized, subset_by_index=[n - num_eigen - 1, n - 1])
            eigenGap = lam[1] - lam[0];
            lam_Kplus1 = 1 - lam[0]
            eigenGapRatio = eigenGap / lam_Kplus1
            print('lam_Kplus1 is', lam_Kplus1)
            print('Current eigenGapRatio is ', eigenGapRatio)

            eig_V = eigenVectors[:, 1:]
            lam_first = 1 - lam[1:-1]
            lam_coef = 1 / lam_first
            eig_V[:, :-1] = lam_coef ** (1 / 2) * eig_V[:, :-1]
            if self.P_eigen:
                D_P = np.diag(P.sum(axis=0))
                K_P = power_diag(D_P, -0.5) @ P @ power_diag(D_P, -0.5)
                lam_P, eigenVectors_P = linalg.eigh(K_P, subset_by_index=[n - num_eigen - 1, n - 1])
                eig_V_P = eigenVectors_P[:, 1:]
                lam_first_P = 1 - lam_P[1:-1]
                lam_coef_P = 1 / lam_first_P
                eig_V_P[:, :-1] = lam_coef_P ** (1 / 2) * eig_V_P[:, :-1]
                coef = self.beta * ((eig_V - eig_V_P) @ (eig_V - eig_V_P).T)
            else:
                coef = self.beta * (eig_V @ eig_V.T)

            grad_LK = cal_gk_L(Q, coef, M)
            grad_Y = cal_gy_K(grad_LK, Y, Q, sigman).ravel()
            lambda_list.append(lam)

            error = self.beta * (num_eigen - sum(lam[1:]))

            return error, grad_Y, lam

        def gk_grad_2(Y, num_eigen, lambda_list, new_obj, min_k=False, UPDATE_SIGMA=False):

            '''
            calculate extra part of gradient w.r.t. Y:
            '''
            global lam_list

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

            def calc_prob_matrix(distances, sigmas=None):
                """Convert a distances matrix to a matrix of probabilities."""
                if sigmas is not None:
                    two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
                    return softmax(distances / two_sig_sq)
                else:
                    return softmax(distances)

            def cal_degree(Dist, sigmas):

                """Wrapper function for quick calculation of
                perplexity over a distance matrix."""
                return calc_prob_matrix(Dist, sigmas).sum(axis=1)

            def find_optimal_sigmas(distances, target_k=self.kneigh):
                """For each row of distances matrix, find sigma that results
                in target perplexity for that role."""
                sigmas = []
                # For each row of the matrix (each point in our dataset)
                for i in range(distances.shape[0]):
                    # Make fn that returns perplexity of this row given sigma
                    eval_fn = lambda sigma: \
                        cal_degree(distances[i:i + 1, :], np.array(sigma))
                    # Binary search over sigmas to achieve target perplexity
                    correct_sigma = binary_search(eval_fn, np.log(target_k))
                    # Append the resulting sigma to our output array
                    sigmas.append(correct_sigma)
                return np.array(sigmas)

            def power_diag(D, power):
                D_new = np.diag(np.power(np.diag(D), power))
                return D_new

            def get_K(Y):
                n = Y.shape[0]
                Dist = pairwise_distances(Y) ** 2
                M = kneighbors_graph(Y, self.kneigh, mode='connectivity', include_self=False)
                M = M.toarray()
                Dist = (Dist * M) - (Dist * M).min(axis=1)
                np.fill_diagonal(Dist, 0)
                print('------ Starting Search for Sigma -------')
                # sigma2 = find_optimal_sigmas(Dist, self.kneigh)
                # sigma2_ = np.tile((Dist * M).mean(axis = 1), (2,1)).T
                # sigma2 = np.tile((Dist * M).mean(axis = 1), (n,1)).T
                sigma2_ = sigma2 = 1
                K = np.exp(- 0.5 * Dist / sigma2)
                print('------ Finish Preparing K -------')
                return K, sigma2_

            def cal_gy_K(gK_L, Y, Kval, sigma2):
                Y = Y.T
                T = gK_L * Kval
                C = np.tile(sum(T), (len(Y), 1))
                g = 2 / sigma2 * (Y @ T - Y * C).T

                return g

            def cal_gk_L(A, coef):
                n = A.shape[0]
                D = np.diag(A.sum(axis=0))  # column sum
                U0 = -0.5 * power_diag(D, -1.5) @ A @ power_diag(D, -0.5) * coef
                U1 = -0.5 * power_diag(D, -0.5) @ A @ power_diag(D, -1.5) * coef
                U0 = np.tile(U0.sum(axis=0), (n, 1)).T
                U1 = np.tile(U1.sum(axis=1), (n, 1))
                U2 = np.tile(power_diag(D, -0.5).sum(axis=0), (n, 1)).T * np.tile(power_diag(D, -0.5).sum(axis=1),
                                                                                  (n, 1)) * coef
                grad_LK = -(U0 + U1 + U2)
                return grad_LK

            print(' ------------Start gk_grad ---------------')

            Kval, sigma2 = get_K(Y)
            n = Y.shape[0]
            D = np.diag(Kval.sum(axis=0))
            K_ = power_diag(D, -0.5) @ Kval @ power_diag(D, -0.5)

            lam, eigenVectors = linalg.eigh(K_, subset_by_index=[n - num_eigen - 1, n - 1])
            eigenGap = lam[1] - lam[0];
            lam_Kplus1 = 1 - lam[0]
            eigenGapRatio = eigenGap / lam_Kplus1
            print('lam_Kplus1 is', lam_Kplus1)
            print('Current eigenGapRatio is ', eigenGapRatio)

            eig_V = eigenVectors[:, 1:]
            # lam_first = 1 - lam[1:]
            # lam_first[-1] = (lam_first[-2]**2 / lam_first[-3])+ 1e-6
            # lam_coef = 1 / lam_first
            # eig_V = (lam_coef ** (1/2)) * eig_V
            coef = self.beta * (eig_V @ eig_V.T)  # num_eigen * num_sample

            grad_LK = cal_gk_L(Kval, coef)
            grad_Y = cal_gy_K(grad_LK, Y, Kval, sigma2).ravel()
            lambda_list.append(lam)

            error = self.beta * (num_eigen - sum(lam[1:]))

            return error, grad_Y, lambda_list

        def t_grad(Y, num_eigen, dist, lambda_list, new_obj, P, min_k=False):

            def power_diag(D, power):
                D_new = np.diag(np.power(np.diag(D), power))
                return D_new

            def cal_gk_L(A, coef):
                n = A.shape[0]
                D = np.diag(A.sum(axis=0))  # column sum
                U0 = -0.5 * power_diag(D, -1.5) @ A @ power_diag(D, -0.5) * coef
                U1 = -0.5 * power_diag(D, -0.5) @ A @ power_diag(D, -1.5) * coef
                U0 = np.tile(U0.sum(axis=0), (n, 1)).T
                U1 = np.tile(U1.sum(axis=1), (n, 1))
                U2 = np.tile(power_diag(D, -0.5).sum(axis=0), (n, 1)).T * np.tile(power_diag(D, -0.5).sum(axis=1),
                                                                                  (n, 1)) * coef
                grad_LK = -(U0 + U1 + U2)  # * M
                return grad_LK

            def cal_gy_K(gK_L, Y, Kval):
                Y = Y.T
                T = gK_L * (Kval ** 2)
                C = np.tile(sum(T), (len(Y), 1))
                g = 4 * (Y @ T - Y * C).T
                return g

            # print(' ------------Start t_grad ---------------')
            Kval = squareform(dist)
            n, d = Y.shape[0], Y.shape[1]
            D = np.diag(Kval.sum(axis=0))
            K_ = power_diag(D, -0.5) @ Kval @ power_diag(D, -0.5)
            L = np.eye(n) - K_
            # lam, eigenVectors = linalg.eigh(K_, subset_by_index=[n - num_eigen - 1, n - 1])
            eigenVectors, lam, _ = randomized_svd(K_, n_components=num_eigen, random_state=0)
            eigenGap = lam[-1] - lam[-2];
            lam_Kplus1 = lam[-1]
            eigenGapRatio = eigenGap / lam_Kplus1
            # print('lam_Kplus1 is', lam_Kplus1)
            # print('Current eigenGapRatio is ', eigenGapRatio)

            if new_obj == 'gap':
                eig_V = eigenVectors[:, 1];
                eig_V_kp1 = eigenVectors[:, 0]
                inv_trace1 = eigenGap ** (-1) * (eig_V_kp1 @ eig_V_kp1.T - eig_V @ eig_V.T)
                inv_trace2 = lam_Kplus1 ** (-1) * (eig_V_kp1 @ eig_V_kp1.T)
                coef = self.beta * (inv_trace1 - inv_trace2)
            elif new_obj == 'firstK':
                eig_V = eigenVectors[:, 1:];
                eig_V_kp1 = eigenVectors[:, 0]
                lam_first = 1 - lam[1:-1]
                lam_coef = 1 / lam_first
                eig_V[:, :-1] = lam_coef ** (1 / 2) * eig_V[:, :-1]
                if self.P_eigen:
                    D_P = np.diag(P.sum(axis=0))
                    K_P = power_diag(D_P, -0.5) @ P @ power_diag(D_P, -0.5)
                    eigenVectors_P, lam_P, _ = randomized_svd(K_P, n_components=num_eigen, random_state=0)
                    eig_V_P = eigenVectors_P[:, 1:]
                    lam_first_P = 1 - lam_P[1:-1]
                    lam_coef_P = 1 / lam_first_P
                    eig_V_P[:, :-1] = lam_coef_P ** (1 / 2) * eig_V_P[:, :-1]
                    coef = self.beta * ((eig_V - eig_V_P) @ (eig_V - eig_V_P).T)
                else:
                    coef = self.beta * (eig_V @ eig_V.T)
            elif new_obj == 'ratio':
                eig_V = eigenVectors[:, 1:];
                eig_V_kp1 = eigenVectors[:, 0]
                inv_trace1 = (self.num_eigen - 1) * sum(lam[1:]) ** (-1) * (eig_V @ eig_V.T)
                inv_trace2 = lam_Kplus1 ** (-1) * (eig_V_kp1 @ eig_V_kp1.T)
                coef = self.beta * (inv_trace2 - inv_trace1)
            else:
                raise ValueError("'new_obj' must be 'gap', 'ratio' or 'firstK'")

            grad_LK = cal_gk_L(Kval, coef)
            grad_Y = cal_gy_K(grad_LK, Y, Kval).ravel()
            lambda_list.append(lam)

            error = self.beta * (num_eigen - sum(lam[1:]))

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
            spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                                  assign_labels='discretize')
            spectral.fit(L)
            grp = spectral.fit_predict(L)
            return grp, L

        def spectral_clustering(C, K, d, alpha, ro):
            C = thrC(C, alpha)
            y, _ = post_proC(C, K, d, ro)
            return y

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = it
        lambda_list = []
        tic = time()

        for i in tqdm(range(it, n_iter), colour='green'):
            check_convergence = (i + 1) % n_iter_check == 0
            # only compute the error when needed
            kwargs['compute_error'] = check_convergence or i == n_iter - 1
            X_embedded = p.reshape(-1, self.n_components)
            error, grad, dist = objective(p, *args, **kwargs)

            if (i > self.kl_num):
                P = squareform(args[0])
                # print(f'-------- new objective: {self.new_obj} --------')
                if self.kernel == 'Gaussian':
                    error_2, grads, lam = gk_grad(X_embedded, self.num_eigen, lambda_list, self.new_obj, P)
                elif self.kernel == 'Student t':
                    error_2, grads, lam = t_grad(X_embedded, self.num_eigen, dist, lambda_list, self.new_obj, P)
                # print(f'gradient: {grads[:2]}, shape: {grads.shape}')
                self.vis.line(np.array([lam]), np.array([i]), win='Lambda', update='append',
                              opts=dict(title='Lambda Curve'))
                # print('-------------- Save eigenvalues -----------------')
                lam_report = os.path.join(self.output_dir, f'eigenvalue_report.csv')
                eigen_output = [i, ] + lam
                with open(lam_report, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(eigen_output)

                if (i + self.lst_stage >= n_iter) & (n_iter == self.n_iter):
                    # print('-------------- last stage coef adjusted! -----------------')
                    grad += self.lastcoef * grads
                    error += error_2
                else:
                    # print('-------------- middle stage coef only beta! -----------------')
                    grad += grads
                    error += error_2

            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains

            if self.opt_method == 'momentum':
                update = momentum * update - learning_rate * grad
                p += update
            elif self.opt_method == 'adam':
                adam_m = 0.9 * adam_m + (1.0 - 0.9) * grad
                adam_v = 0.999 * adam_v + (1.0 - 0.999) * grad ** 2
                mhat = adam_m / (1.0 - 0.9 ** (i + 1))
                vhat = adam_v / (1.0 - 0.999 ** (i + 1))
                p -= 0.01 * mhat / (np.sqrt(vhat) + 1e-8)

            toc = time()
            duration = toc - tic
            tic = toc

            self.vis.line(np.array([duration]), np.array([i]), win='Duration_i', update='append',
                          opts=dict(title='Time Curve'))
            time_output = [i, duration]
            time_report = os.path.join(self.output_dir, f'time_report.csv')
            with open(time_report, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(time_output)

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

            X_embedded_all = K_coef @ X_embedded
            if (i % 10 == 0):
                if (self.proxy is True):
                    self.vis.scatter(X_embedded_all, self.y + 1,
                                     opts=dict(markersize=4, title=f'T-SNE (Exact Revised) step{i}'))
                    self.vis.scatter(X_embedded, None, opts=dict(markersize=4, title=f'T-SNE (Pseudo) step{i}'))
                    # save .npy
                    embedding_file = os.path.join(self.output_dir, f'X_embedded/embedding_{i}.npy')
                    np.save(embedding_file, X_embedded_all)
                    # save .eps
                    image_file = os.path.join(self.output_dir, f'X_embedded/image/embedding_{i}.eps')
                    fig = plt.figure(dpi=1200)
                    sns.scatterplot(x=X_embedded_all[:, 0], y=X_embedded_all[:, 1], c=self.y + 1)
                    plt.title(f'T-SNE with proxy at step {i}')
                    fig.savefig(image_file, format='eps', dpi=1200)
                else:
                    self.vis.scatter(X_embedded, self.y + 1,
                                     opts=dict(markersize=4, title=f'T-SNE (Exact Revised) step{i}'))
                    # save .npy
                    embedding_file = os.path.join(self.output_dir, f'X_embedded/embedding_{i}.npy')
                    np.save(embedding_file, X_embedded)
                    # save .eps
                    image_file = os.path.join(self.output_dir, f'X_embedded/image/embedding_{i}.eps')
                    fig = plt.figure(dpi=1200)
                    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], c=self.y + 1)
                    plt.title(f'T-SNE at step {i}')
                    fig.savefig(image_file, format='eps', dpi=1200)

            if (i % 20 == 0) | (i + 1 >= self.n_iter):
                # KNN score
                # print('--------Calculate K-NN Score-------')
                score_list = []
                if self.knn_type == 'small':
                    neighbors = [10, 20, 40, 80, 160]
                else:
                    neighbors = [100, 200, 400, 800, 1600]
                for n_neighbor in neighbors:
                    neigh = KNeighborsClassifier(n_neighbors=n_neighbor)

                    neigh.fit(X_embedded_all, self.y.ravel())
                    score_list.append(neigh.score(X_embedded, self.y.ravel()))
                knn_report = [i, ] + score_list
                '''
                example:
                10, 0.9123, 0.8923, 0.8452, 0.8213
                20, 0.9321, 0.9023, 0.8762, 0.8533
                '''
                output_report = os.path.join(self.output_dir, f'knn_report.csv')
                with open(output_report, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(knn_report)

                # print('--------Calculate K-Means Score-------')
                score = 0
                for t in range(5):
                    y_pred = KMeans(n_clusters=self.num_eigen, random_state=t).fit_predict(X_embedded)
                    score += acc(self.y.ravel(), y_pred)
                score = score / 5
                kMeans_report = [i, score]
                output2_report = os.path.join(self.output_dir, f'kMeans_report.csv')
                with open(output2_report, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(kMeans_report)

                self.vis.line(np.array([score_list]), np.array([i]), win='KNN-Score', update='append',
                              opts=dict(title='Score Curve'))
                self.vis.line(np.array([error]), np.array([i]), win='Objective value', update='append',
                              opts=dict(title='Objective value'))

                # print(f'K-NN Score at {i} is {dict(zip(neighbors,score_list))}')

        return p, error, i

    def _fit(self, X, skip_num_points=0):
        global K_coef
        """Private function to fit the model using X as training data."""

        if self.method not in ['barnes_hut', 'exact']:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.square_distances not in [True, 'legacy']:
            raise ValueError("'square_distances' must be True or 'legacy'.")
        if self.metric != "euclidean" and self.square_distances is not True:
            warnings.warn(
                "'square_distances' has been introduced in 0.24 to help phase "
                "out legacy squaring behavior. The 'legacy' setting will be "
                "removed in 1.1 (renaming of 0.26), and the default setting "
                "will be changed to True. In 1.3, 'square_distances' will be "
                "removed altogether, and distances will be squared by "
                "default. Set 'square_distances'=True to silence this "
                "warning.",
                FutureWarning
            )
        if self.method == 'barnes_hut':
            X = self._validate_data(X, accept_sparse=['csr'],
                                    ensure_min_samples=2,
                                    dtype=[np.float32, np.float64])
        else:
            X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'],
                                    dtype=[np.float32, np.float64])
        if self.metric == "precomputed":
            if isinstance(self.init, str) and self.init == 'pca':
                raise ValueError("The parameter init=\"pca\" cannot be "
                                 "used with metric=\"precomputed\".")
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(X, "TSNE.fit(). With metric='precomputed', X "
                                  "should contain positive distances.")

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    'or provide the dense distance matrix.')

        if self.method == 'barnes_hut' and self.n_components > 3:
            raise ValueError("'n_components' should be inferior to 4 for the "
                             "barnes_hut algorithm as it relies on "
                             "quad-tree or oct-tree.")
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError("early_exaggeration must be at least 1, but is {}"
                             .format(self.early_exaggeration))

        if self.n_iter < self._EXPLORATION_N_ITER:
            raise ValueError(f"n_iter should be at least {self._EXPLORATION_N_ITER}")

        n_samples = X.shape[0]

        if self.proxy is True:
            print(f'======= X shape is {X.shape} ==========')
            n_component = X.shape[1] // 20
            pca = PCA(n_components=n_component, svd_solver='randomized', random_state=random_state)
            X_low = pca.fit_transform(X).astype(np.float64, copy=False)
            print(f'==== Finish PCA with X_low ({X_low.shape})=====')
            X = KMeans(n_clusters=self.num_proxies, random_state=0, init='k-means++').fit(
                X_low).cluster_centers_  # (2000, X.shape[1]//10)
            print('X is \n', X)
            print('==== Finish KMeans =====')

            def Epan_kernel(X_low, X):
                X_diff = pairwise_distances(X_low, X,
                                            metric='l1')  # (70,000, X.shape[1]//10) - (2,000, X.shape[1]//10) -> (70,000, 2,000)
                print('X_diff 1 is \n', X_diff[0])
                ker_lambda = np.mean(X_diff, axis=1) * 1
                t = X_diff / np.tile(ker_lambda, (X_diff.shape[1], 1)).T
                t[t > 1] = 1
                print('t matrix has zero values:', (t == 0).sum())
                print('t 1 is \n', t[0])
                K_lam = 3 / 4 * (1 - t ** 2)
                K_coef = K_lam / np.tile(np.sum(K_lam, axis=1), (K_lam.shape[1], 1)).T
                return K_coef

            def tricube_kernel(X_low, X):
                X_diff = pairwise_distances(X_low, X,
                                            metric='l1')  # (70,000, X.shape[1]//10) - (2,000, X.shape[1]//10) -> (70,000, 2,000)
                print('X_diff 1 is \n', X_diff[0])
                ker_lambda = np.mean(X_diff, axis=1) * 1
                t = X_diff / np.tile(ker_lambda, (X_diff.shape[1], 1)).T
                t[t > 1] = 1
                print('t matrix has zero values:', (t == 0).sum())
                print('t 1 is \n', t[0])
                K_lam = (1 - t ** 3) ** 3
                K_coef = K_lam / np.tile(np.sum(K_lam, axis=1), (K_lam.shape[1], 1)).T
                return K_coef

            # K_coef = Epan_kernel(X_low, X)
            K_coef = tricube_kernel(X_low, X)
            print(f'K_coef matrix has shape {K_coef.shape}')
            print('K_coef 1 is \n', K_coef[0])
        else:
            K_coef = np.eye(n_samples)

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric,
                                                   squared=True)
                else:
                    distances = pairwise_distances(X, metric=self.metric,
                                                   n_jobs=self.n_jobs)

            if np.any(distances < 0):
                raise ValueError("All distances should be positive, the "
                                 "metric given is not correct")

            if self.metric != "euclidean" and self.square_distances is True:
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            # assert np.all(P <= 1), ("All probabilities should be less "
            # "or then equal to one")

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3. * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors..."
                      .format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = NearestNeighbors(algorithm='auto',
                                   n_jobs=self.n_jobs,
                                   n_neighbors=n_neighbors,
                                   metric=self.metric)
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                    n_samples, duration))

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode='distance')
            duration = time() - t0
            if self.verbose:
                print("[t-SNE] Computed neighbors for {} samples "
                      "in {:.3f}s...".format(n_samples, duration))

            # Free the memory used by the ball_tree
            del knn

            if self.square_distances is True or self.metric == "euclidean":
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity,
                                        self.verbose)

        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == 'pca':
            pca = PCA(n_components=self.n_components, svd_solver='randomized',
                      random_state=random_state)
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
        elif self.init == 'random':
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(
                n_samples, self.n_components).astype(np.float32)
        elif self.init == 'spectral':

            def power_diag(D, power):
                D_new = np.diag(np.power(np.diag(D), power))
                return D_new

            D = np.diag(squareform(P).sum(axis=1))
            _, X_embedded = linalg.eigh(np.eye(D.shape[0]) - power_diag(D, -0.5) @ squareform(P) @ power_diag(D, -0.5),
                                        subset_by_index=[1, self.n_components])
            # X_embedded, _, _ = randomized_svd(power_diag(D, -0.5) @ squareform(P) @ power_diag(D, -0.5), n_components = self.n_components, random_state= 0)

        else:
            raise ValueError("'init' must be 'pca', 'random', or "
                             "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._tsne(P, degrees_of_freedom, n_samples, X=X,
                          X_embedded=X_embedded,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points
                          )

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded, X,
              neighbors=None, skip_num_points=0):
        global adam_m, adam_v
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8
        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [P, degrees_of_freedom, n_samples, self.n_components],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }
        if self.method == 'barnes_hut':
            obj_func = _kl_divergence_bh
            opt_args['kwargs']['angle'] = self.angle
            # Repeat verbose argument for _kl_divergence_bh
            opt_args['kwargs']['verbose'] = self.verbose
            # Get the number of threads for gradient computation here to
            # avoid recomputing it at each iteration.
            opt_args['kwargs']['num_threads'] = _openmp_effective_n_threads()
        elif self.method == 'exact':
            obj_func = _kl_divergence

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter

        P *= self.early_exaggeration
        adam_m = adam_v = np.zeros(len(params))
        params, kl_divergence, it = self._gradient_descent(obj_func, params, X=X,
                                                           **opt_args)
        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations with early "
                  "exaggeration: %f" % (it + 1, kl_divergence))

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args['n_iter'] = self.n_iter
            opt_args['it'] = it + 1
            opt_args['momentum'] = 0.8
            opt_args['n_iter_without_progress'] = self.n_iter_without_progress
            params, kl_divergence, it = self._gradient_descent(obj_func, params, X=X,
                                                               **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print("[t-SNE] KL divergence after %d iterations: %f"
                  % (it + 1, kl_divergence))

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed
        output.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, y=None):
        """Fit X into an embedded space.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.
        y : Ignored
        """
        self.fit_transform(X)
        return self

    def get_P(self, X, desired_perplexity, verbose=True):
        distances = pairwise_distances(X, metric=self.metric, squared=True)
        P = _joint_probabilities(distances, desired_perplexity, verbose)
        return P