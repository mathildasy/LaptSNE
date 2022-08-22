# Author: Alexander Fabisch  -- <afabisch@informatik.uni-bremen.de>
# Author: Christopher Moody <chrisemoody@gmail.com>
# Author: Nick Travers <nickt@squareup.com>
# License: BSD 3 clause (C) 2014

# This is the exact and Barnes-Hut t-SNE implementation. There are other
# modifications of the algorithm:
# * Fast Optimization for t-SNE:
#   https://cseweb.ucsd.edu/~lvdmaaten/workshops/nips2010/papers/vandermaaten.pdf

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import os
import warnings
from time import time
import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.sparse import csr_matrix, issparse
from numbers import Integral, Real
from ..neighbors import NearestNeighbors, KNeighborsClassifier
from ..base import BaseEstimator
from ..utils import check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils.validation import check_non_negative
from ..utils._param_validation import Interval, StrOptions, Hidden
from ..decomposition import PCA
from ..metrics.pairwise import pairwise_distances, _VALID_METRICS
from laplacian import g_grad, t_grad, power_diag
import dask.array as da

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
from . import _utils  # type: ignore

# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from . import _barnes_hut_tsne  # type: ignore

MACHINE_EPSILON = np.finfo(np.double).eps
NUM_PROXY = 1000


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
    return kl_divergence, grad



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

    def __init__(self, n_components=2, *, perplexity=30.0,
                 early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
                 n_iter_without_progress=300, min_grad_norm=1e-7,
                 metric="euclidean", init="random", verbose=0, kernel='Gaussian',
                 random_state=None, method='exact', angle=0.5,
                 n_jobs=None, square_distances='legacy', 
                 vis=None, label=None, num_eigen=0, beta=1e0, exag_stage=250,
                 new_obj='firstK', knn_type='small', sigman_type='constant', sigman_constant=1e1,
                 out_dir=None, proxy=False, batch_size = 100, num_neighbors=10):
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
        self.num_eigen = num_eigen
        self.beta = beta
        self._EXPLORATION_N_ITER = exag_stage
        self.kernel = kernel
        self.new_obj = new_obj
        self.knn_type = knn_type
        self.sigman_type = sigman_type
        self.sigman = sigman_constant
        self.output_dir = out_dir
        self.proxy = proxy
        self.batch_size = batch_size
        self.mini_batch = True if batch_size > 0 else False
        self.num_neighbors = num_neighbors

    def _gradient_descent(self, objective, p0, it, n_iter, X=None,
                          n_iter_check=1, n_iter_without_progress=300,
                          momentum=0.8, learning_rate=200.0, min_gain=0.01,
                          min_grad_norm=1e-7, verbose=0, K_coef = None, args=None, kwargs=None):
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
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        if self.kernel == 'Gaussian':
            objective_2 = g_grad
        else:
            objective_2 = t_grad

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = it

        if self.mini_batch & (self.num_neighbors != 'tsne'):
            P = squareform(args[0]) # (n_samples, n_samples)
            P_dis = 1 - P
            neigh = NearestNeighbors(n_neighbors=self.num_neighbors, metric='precomputed').fit(P_dis)
            A = neigh.kneighbors_graph(P_dis).toarray()

        pbar = tqdm(range(it, n_iter), colour='green')
        for i in pbar:

            check_convergence = (i + 1) % n_iter_check == 0
            # only compute the error when needed
            kwargs['compute_error'] = check_convergence or i == n_iter - 1
            error, grad = objective(p, *args, **kwargs)

<<<<<<< HEAD
            if self.num_neighbors != 'tsne': 
                X_embedded = p.reshape(-1, self.n_components)
                if i % 10 == 0:
                    pbar.set_description('Processing %s samples'%(len(X_embedded)))
                    grads, eig_V = objective_2(X_embedded, self.num_eigen, 
                                                            self.beta, self.new_obj, 
                                                            skip_decompose=False)
                elif self.mini_batch:
                    grads = np.zeros_like(X_embedded)
                    #TODO: for mini-batched method, how to randomly select?
                    selected_sample = np.random.choice(np.arange(len(X_embedded)), size=self.batch_size, replace=False)
                    A_list = A[selected_sample]
                    K_index = np.argwhere(np.sum(A_list, axis=0) != 0).ravel() # relevant Nk samples 
                    X_embedded2 = X_embedded[K_index]
                    eig_V2 = eig_V[K_index]
                    pbar.set_description('Processing %s samples'%(len(K_index)))
                    grads2, eig_V2 = objective_2(X_embedded2, self.num_eigen, 
                                                            self.beta, self.new_obj, 
                                                            skip_decompose=True, 
                                                            eig_V=eig_V2)
                    grads[K_index] += grads2
                else:
                    pbar.set_description('Processing %s samples'%(len(X_embedded)))
                    grads, eig_V = objective_2(X_embedded, self.num_eigen, 
=======
            X_embedded = p.reshape(-1, self.n_components)

            ## just test whether relationship between non-neighbors are close to zero
            def get_Q_tStudent(eucdis, degrees_of_freedom):
                eucdis /= degrees_of_freedom
                eucdis += 1.
                Q = np.power(eucdis, (degrees_of_freedom + 1.0) / -2.0)
                return Q

            if i % 10 == 0:
                pbar.set_description('Processing %s samples'%(len(X_embedded)))
                grads, eig_V = objective_2(X_embedded, self.num_eigen, 
                                                        self.beta, self.new_obj, 
                                                        skip_decompose=False)
                
            
            elif self.mini_batch:
                grads = np.zeros_like(X_embedded)
                #TODO: for mini-batched method, how to randomly select?
                selected_sample = np.random.choice(np.arange(len(X_embedded)), size=self.batch_size, replace=False)
                A_list = A[selected_sample]
                K_index = np.argwhere(np.sum(A_list, axis=0) != 0).ravel() # relevant Nk samples 
                X_embedded2 = X_embedded[K_index]
                eig_V2 = eig_V[K_index]
                eucdis_1 = pdist(X_embedded, 'sqeuclidean')
                Q_1 = get_Q_tStudent(eucdis_1, degrees_of_freedom = 2)
                print('########### the Q except for nearest neighbors ########')
                print(Q_1[-K_index, -K_index])
                pbar.set_description('Processing %s samples'%(len(K_index)))
                grads2, eig_V2 = objective_2(X_embedded2, self.num_eigen, 
>>>>>>> 65499cee6f560c8db9f2aa933a14c86c8107518a
                                                        self.beta, self.new_obj, 
                                                        skip_decompose=True, 
                                                        eig_V = eig_V)

                grads = grads.ravel()
                grad += grads

            grad_norm = linalg.norm(grad)
            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains

            update = momentum * update - learning_rate * grad
            p += update

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check))

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
            
            X_embedded = p.reshape(-1, self.n_components)

        return p, error, i


    def _fit(self, X, skip_num_points=0):
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

        if self.proxy is True:
            from ..cluster import KMeans
            print(f'======= X shape is {X.shape} ==========')
            n_component = 30 #X.shape[1] // 20
            pca = PCA(n_components=n_component, svd_solver='randomized', random_state=random_state)
            X_low = pca.fit_transform(X).astype(np.float64, copy=False)
            print(f'==== Finish PCA with X_low ({X_low.shape})=====')
            X = KMeans(n_clusters=NUM_PROXY, random_state=0, init='k-means++', n_init ='auto').fit(
                X_low).cluster_centers_ 

            n_samples = X.shape[0]
            print('==== Finish KMeans =====')

            def Epan_kernel(X_low, X):
                X_diff = pairwise_distances(X_low, X,
                                            metric='l1')  # (70,000, X.shape[1]//10) - (2,000, X.shape[1]//10) -> (70,000, 2,000)
                ker_lambda = np.mean(X_diff, axis=1) * 1
                t = X_diff / np.tile(ker_lambda, (X_diff.shape[1], 1)).T
                t[t > 1] = 1
                K_lam = 3 / 4 * (1 - t ** 2)
                K_coef = K_lam / np.tile(np.sum(K_lam, axis=1), (K_lam.shape[1], 1)).T
                return K_coef

            def tricube_kernel(X_low, X):
                X_diff = pairwise_distances(X_low, X,
                                            metric='l1')  # (70,000, X.shape[1]//10) - (2,000, X.shape[1]//10) -> (70,000, 2,000)
                ker_lambda = np.mean(X_diff, axis=1) * 1
                t = X_diff / np.tile(ker_lambda, (X_diff.shape[1], 1)).T
                t[t > 1] = 1
                K_lam = (1 - t ** 3) ** 3
                K_coef = K_lam / np.tile(np.sum(K_lam, axis=1), (K_lam.shape[1], 1)).T
                return K_coef

            K_coef = Epan_kernel(X_low, X)
            # K_coef = tricube_kernel(X_low, X)
            print(f'K_coef matrix has shape {K_coef.shape}')
            # K_coef = K_coef**0.5
        else:
            n_samples = X.shape[0]
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
            # FIXME: optimize speed
            # D = np.diag(squareform(P).sum(axis=1))
            # _, X_embedded = linalg.eigh(np.eye(D.shape[0]) - power_diag(D, -0.5) @ squareform(P) @ power_diag(D, -0.5),
            #                             subset_by_index=[1, self.n_components])

            D_diag = squareform(P).sum(axis=0)
            D_05 = np.diag(np.power(D_diag, -0.5))
            L = np.matmul(np.matmul(D_05, squareform(P)), D_05)
            L_da = da.from_array(L, asarray=True)
            X_embedded, _, _ = da.linalg.svd_compressed(L_da, k=self.n_components, compute=True)
            X_embedded = X_embedded.compute()
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
                          X_embedded=X_embedded, K_coef=K_coef,
                          neighbors=neighbors_nn,
                          skip_num_points=skip_num_points
                          ), K_coef

    def _tsne(self, P, degrees_of_freedom, n_samples, X_embedded, X, K_coef,
              neighbors=None, skip_num_points=0):
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
            "K_coef": K_coef,
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
        embedding, K_coef = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_, K_coef

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