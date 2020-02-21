from dask import delayed, compute
from dask_ml.utils import svd_flip
#from dask_ml._utils import draw_seed
from sklearn.decomposition import PCA
import sklearn.utils.extmath as skm
import numbers
from sklearn.utils.validation import check_random_state
import dask.array as da
import xarray as xr
import numpy as np
import time
import os

def draw_seed(random_state, low, high=None, size=None, dtype=None, chunks=None):
    kwargs = {"size": size}
    if chunks is not None:
        kwargs["chunks"] = chunks

    seed = random_state.randint(low, high, **kwargs)
    if dtype is not None and isinstance(seed, (da.Array, np.ndarray)):
        seed = seed.astype(dtype)

    return seed

def _incremental_mean_and_var(X, last_mean, last_variance, last_sample_count):
    last_sum = last_mean * last_sample_count
    new_sum = da.nansum(X, axis=0)

    new_sample_count = X.shape[1] # Can't handle nan value appropriately for same reason
    #new_sample_count = da.sum(~da.isnan(X), axis=0)
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = (da.nanvar(X, axis=0) * new_sample_count)
        last_unnormalized_variance = last_variance * last_sample_count

        last_over_new_count = last_sample_count / new_sample_count
        updated_unnormalized_variance = (
            last_unnormalized_variance + new_unnormalized_variance +
            last_over_new_count / updated_sample_count *
            (last_sum / last_over_new_count - new_sum) ** 2)

        updated_unnormalized_variance = \
            da.where(last_sample_count==0, updated_unnormalized_variance, 
                     new_unnormalized_variance) 
        updated_variance = updated_unnormalized_variance / updated_sample_count

    updated_sample_count = updated_sample_count.compute()

    return updated_mean, updated_variance, updated_sample_count

_incremental_mean_and_var.__doc__ = skm._incremental_mean_and_var.__doc__

class IncrementalPCA:
    """
    """

    def __init__(
        self, 
        n_components=None, 
        whiten=False, 
        copy=True,
        svd_solver="auto",
        batch_size=None,
        iterated_power=0,
        random_state=None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.svd_solver = svd_solver
        self.batch_size = batch_size
        self.iterated_power = iterated_power
        self.random_state = random_state

    def partial_fit(self, X, y=None):
        """
        """

        solvers = {"full", "auto", "tsqr", "randomized"}
        solver = self.svd_solver

        if solver not in solvers:
            raise ValueError(
                "Invalid solver '{}'. Must be one of {}".format(solver, solvers)
            )

        n_samples, n_features = X.shape

        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features)
            else:
                self.n_components_ = self.components_.shape[0]
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to "
                             "the batch number of samples "
                             "%d." % (self.n_components, n_samples))
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] !=
                                               self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." %
                             (self.components_.shape[0], self.n_components_))

        if solver == "auto":
            # Small problem, just call full PCA
            if not _known_shape(X.shape):
                raise ValueError(
                    "Cannot automatically choose PCA solver with unknown "
                    "shapes. To clear this error,\n\n"
                    "    * pass X.to_dask_array(lengths=True)  "
                    "# for Dask DataFrame (dask >= 0.19)\n"
                    "    * pass X.compute_chunk_sizes()  "
                    "# for Dask Array X (dask >= 2.4)\n"
                    "    * Use a specific SVD solver "
                    "(e.g., ensure `svd_solver in ['randomized', 'tsqr', 'full']`)"
                )
            if max(n_samples, n_features) <= 500:
                solver = "full"
            elif n_components >= 1 and n_components < 0.8 * min(n_samples, n_features):
                solver = "randomized"
            # This is also the case of n_components in (0,1)
            else:
                solver = "full"

        # This is the first partial_fit
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            self.mean_ = .0
            self.var_  = .0

        # Update stats - they are 0 if this is the first step
        last_sample_count = da.from_array( np.repeat(self.n_samples_seen_, X.shape[1]) )
        col_mean, col_var, n_total_samples = _incremental_mean_and_var(X, self.mean_, self.var_, last_sample_count)
        n_total_samples = n_total_samples[0]

        # Whitening
        if self.n_samples_seen_ == 0:
            # If it is the first step, simply whiten X
            X -= col_mean
        else:
            col_batch_mean = da.mean(X, axis=0)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = \
                np.sqrt((self.n_samples_seen_ * n_samples) /
                        n_total_samples) * (self.mean_ - col_batch_mean)

            X = da.concatenate([self.singular_values_.reshape((-1,1)) * 
                               self.components_, X, mean_correction.reshape((1,-1))], axis=0)

        if solver in {"full", "tsqr"}:
            U, S, V = da.linalg.svd(X)
        else:
            # randomized
            random_state = check_random_state(self.random_state)
            seed = draw_seed(random_state, np.iinfo("int32").max)
            n_power_iter = self.iterated_power
            U, S, V = da.linalg.svd_compressed(
                X, self.n_components_, n_power_iter=n_power_iter, seed=seed
            )
        U, V = svd_flip(U, V)

        explained_variance = (S ** 2) / (n_total_samples -1)
        components, singular_values = V, S

        if solver == "randomized":
            total_var = X.var(ddof=1, axis=0).sum()
        else:
            total_var = explained_variance.sum()

        explained_variance_ratio = explained_variance / total_var

        self.n_samples_seen_ = n_total_samples
        self.mean_ = col_mean
        self.var_  = col_var

        if self.n_components_ < min(n_features, n_samples):
            if solver == "randomized":
                noise_variance = (total_var.sum() - explained_variance.sum()) / (
                    min(n_features, n_samples) - n_components
                )
            else:
                noise_variance = \
                    da.mean(explained_variance[self.n_components_:])
        else:
            noise_variance = 0.

        try:
            (
                self.n_samples_,
                self.n_features_,
                self.components_,
                self.explained_variance_,
                self.explained_variance_ratio_,
                self.singular_values_,
                self.noise_variance_,
            ) = compute(
                n_samples,
                n_features,
                components,
                explained_variance,
                explained_variance_ratio,
                singular_values,
                noise_variance,
            )
        except ValueError as e:
            if np.isnan([n_samples, n_features]).any():
                msg = (
                    "Computation of the SVD raised an error. It is possible "
                    "n_components is too large. i.e., "
                    "`n_components > np.nanmin(X.shape) = "
                    "np.nanmin({})`\n\n"
                    "A possible resolution to this error is to ensure that "
                    "n_components <= min(n_samples, n_features)"
                )
                raise ValueError(msg.format(X.shape)) from e
            raise e

        self.components_ = self.components_[:self.n_components_]
        self.explained_variance_ = self.explained_variance_[:self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ratio_[:self.n_components_]
        self.singular_values_ = self.singular_values_[:self.n_components_]

        if len(self.singular_values_) < self.n_components_:
            self.n_components_ = len(self.singular_values_)
            msg = (
                "n_components={n} is larger than the number of singular values"
                " ({s}) (note: PCA has attributes as if n_components == {s})"
            )
            raise ValueError(msg.format(n=self.n_components_, s=len(self.singular_values_)))

        return self

def _known_shape(shape):
    return all(isinstance(x, numbers.Integral) for x in shape)

if __name__ == '__main__':
    rng = np.random.RandomState(1)
    X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T
    # Cast to dask array
    X = da.from_array(X, chunks=2)
    batch_size = 4
    ipca = IncrementalPCA(n_components=2, batch_size=batch_size)
    ipca.partial_fit(X)
