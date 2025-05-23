###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math
import sys
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from torch.quasirandom import SobolEngine

from .gp import train_gp
from .utils import from_unit_cube, latin_hypercube, to_unit_cube


class Turbo1:
    """The TuRBO-1 algorithm.

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        turbo1.optimize()  # Run optimization
        X, fX = turbo1.X, turbo1.fX  # Evaluated points
    """

    def __init__(
        self,
        f,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        direction="max"
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        assert direction == 'max' or direction == 'min'
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"

        # Save function information
        self.f = f
        self.dim = len(lb)
        self.lb = lb
        self.ub = ub

        # Settings
        self.n_init = n_init
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps

        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))

        # Tolerances and counters
        self.n_cand = min(100 * self.dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, self.dim / batch_size]))
        self.succtol = 3
        self.n_evals = 0

        # Trust region sizes
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8

        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))

        # Device and dtype for GPyTorch
        self.min_cuda = min_cuda
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()

        if direction == 'max':
            self.np_best = lambda f1, f2: np.max(f1) > np.max(f2)
            self.best_obj = lambda f1: f1.max()
            self.np_arg_best = lambda f1: f1.argmax()
            self.is_improved = lambda fX: np.max(fX) > np.max(self._fX) + 1e-3 * math.fabs(np.max(self._fX))
            self.worst_val = np.NINF
        else:
            self.np_best = lambda f1, f2: np.min(f1) < np.min(f2)
            self.best_obj = lambda f1: f1.min()
            self.np_arg_best = lambda f1: f1.argmin()
            self.is_improved = lambda fX: np.min(fX) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX))
            self.worst_val = np.inf

        self.gp = None
        self.mu = 0
        self.sigma = 0

        # Initialize parameters
        self._restart()

    def __del__(self):
        del self.gp

    def _restart(self):
        self._X = []
        self._fX = []
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def _adjust_length(self, fX_next):
        if self.is_improved(fX_next):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        if self.succcount == self.succtol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:  # Shrink trust region
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X, fX, length, n_training_steps, hypers):
        """Generate candidates assuming X has been scaled to [0,1]^d."""
        # Pick the center as the point with the smallest function values
        # NOTE: This may not be robust to noise, in which case the posterior mean of the GP can be used instead
        assert X.min() >= 0.0 and X.max() <= 1.0

        # Standardize function values.
        self.mu, self.sigma = np.median(fX), fX.std()
        self.sigma = 1.0 if self.sigma < 1e-6 else self.sigma
        fX = (deepcopy(fX) - self.mu) / self.sigma

        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use CG + Lanczos for training if we have enough data
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            self.gp = train_gp(
                train_x=X_torch, train_y=y_torch, use_ard=self.use_ard, num_steps=n_training_steps, hypers=hypers
            )

            # Save state dict
            hypers = self.gp.state_dict()

        # Create the trust region boundaries
        x_center = X[self.np_arg_best(fX).item(), :][None, :]
        weights = self.gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()  # This will make the next line more stable
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # We now have weights.prod() = 1
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)

        # Draw a Sobolev sequence in [lb, ub]
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1

        # Create candidate points
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]

        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We may have to move the GP to a new device
        self.gp = self.gp.to(dtype=dtype, device=device)

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = self.gp.likelihood(self.gp(X_cand_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch, y_torch, X_cand_torch

        # De-standardize the sampled values
        y_cand = self.mu + self.sigma * y_cand

        return X_cand, y_cand, hypers

    def _select_candidates(self, X_cand, y_cand):
        """Select candidates."""
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            # Pick the best point and make sure we never pick it again
            indbest = self.np_arg_best(y_cand[:, i])
            X_next[i, :] = deepcopy(X_cand[indbest, :])
            y_cand[indbest, :] = self.worst_val
        return X_next

    def _record_all_visited_sol_and_obj(self, X, fX):
        """
        记录仿真平台采样后的结果
        X: 表示实际采样的解，numpy.array, shape为m行self.dim列，其中m是解的数量，self.dim是解的维度
        fX: 表示解的目标函数值，numpy.array，shape为m行1列，
        """
        self.X = np.vstack((self.X, deepcopy(X)))
        self.fX = np.vstack((self.fX, deepcopy(fX)))

    def is_trust_region_big_enough(self):
        return self.length >= self.length_min

    def initial_suggested_sampling(self):
        """
        返回建议采样的numpy.array，shape为self.n_init行self.dim列，前者表示解的数量，后者表示解的维度
        这个函数返回的采样点是完全随机的，
        可以最开始的时候调用一次，以及is_trust_region_big_enough返回false的时候再调用，
        也可以SAT那边生成随机的初始解，而不使用这个。
        """
        if len(self._fX) > 0 and self.verbose:
            n_evals, fbest = self.n_evals, self.best_obj(self._fX)
            print(f"{n_evals}) Restarting with fbest = {fbest:.4}")
            sys.stdout.flush()
        # Initialize parameters
        self._restart()
        X_init = latin_hypercube(self.n_init, self.dim)
        return from_unit_cube(X_init, self.lb, self.ub)

    def initial_solutions_of_trust_region(self, X, fX):
        """
        确定置信区间的初始解，
        在原始算法中，每次是先随机生成一些点，然后根据这些点获得新的采样点，直到置信空间太小或者采样次数用完
        X: 表示实际采样的解，numpy.array, shape为m行self.dim列，其中m是初始解的数量，self.dim是解的维度
        fX: 表示解的目标函数值，numpy.array，shape为m行1列，
        """
        self.n_evals += len(X)
        self._X = deepcopy(X)
        self._fX = deepcopy(fX)
        # Append data to the global history
        self._record_all_visited_sol_and_obj(X, fX)
        if self.verbose:
            fbest = self.best_obj(self._fX)
            print(f"Starting from fbest = {fbest:.4}")
            sys.stdout.flush()

    def suggested_sampling(self):
        """
        返回建议采样的numpy.array，shape为self.batch_size行，self.dim列，前者为一次推荐的采样的点数量，后者为解的维度
        需要is_trust_region_big_enough返回true的情形下调用才比较合适
        """
        # Warp inputs
        X = to_unit_cube(deepcopy(self._X), self.lb, self.ub)

        # Standardize values
        fX = deepcopy(self._fX).ravel()
        # Create th next batch
        del self.gp
        X_cand, y_cand, _ = self._create_candidates(
            X, fX, length=self.length, n_training_steps=self.n_training_steps, hypers={}
        )
        X_next = self._select_candidates(X_cand, y_cand)

        # Undo the warping
        return from_unit_cube(X_next, self.lb, self.ub)

    def select_from_candidates(self, X_cand):
        """
        从X_cand中选择self.batch_size个解
        X_cand：numpy.array, shape为x行，self.dim列，x表示候选集中解的数量，self.dim是解的维度
        TODO: 目前为了只训练一次模型，导致suggested_sampling和select_from_candidates返回的解的个数都是self.batch_size，如果有需要的话可以训练两个模型
        """
        # print(X_cand)
        # Figure out what device we are running on
        if len(X_cand) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype

        # We use Lanczos for sampling if we have enough data
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = self.gp.likelihood(self.gp(X_torch)).sample(torch.Size([self.batch_size])).t().cpu().detach().numpy()

        # Remove the torch variables
        del X_torch

        # De-standardize the sampled values
        y_cand = self.mu + self.sigma * y_cand

        return self._select_candidates(X_cand, y_cand)

    def solutions_and_obj_after_sampling(self, X, fX):
        """
        使用真实采样的解和目标函数值更新
        X: 表示实际采样的解，numpy.array, shape为m行self.dim列，其中m是解的数量，self.dim是解的维度
        fX: 表示解的目标函数值，numpy.array，shape为m行1列，
        与原始算法的差异在于原始算法采样的点就是suggested_sampling得到的点，而现在采样的点是SAT算出来的点
        """
        # Update trust region
        self._adjust_length(fX)

        # Update budget and append data
        self.n_evals += len(X)
        self._X = np.vstack((self._X, X))
        self._fX = np.vstack((self._fX, fX))

        if self.verbose and self.np_best(fX, self.fX):
            n_evals, fbest = self.n_evals, self.best_obj(fX)
            print(f"{n_evals}) New best: {fbest:.4}")
            sys.stdout.flush()

        # Append data to the global history
        self._record_all_visited_sol_and_obj(X, fX)

    def best_sol_and_obj(self):
        ind_best = self.np_arg_best(self.fX)
        return self.X[ind_best, :], self.fX[ind_best]

    def optimize(self):
        """Run the full optimization process."""
        while self.n_evals < self.max_evals:
            # Generate and evalute initial design points
            X_init = self.initial_suggested_sampling()
            # print(X_init)
            fX_init = np.array([[self.f(x)] for x in X_init])
            # print(fX_init)
            # Update budget and set as initial data for this TR
            self.initial_solutions_of_trust_region(X_init, fX_init)
            # print(self.n_evals) 
            # print(self.max_evals)
            # Thompson sample to get next suggestions
            while self.n_evals < self.max_evals and self.is_trust_region_big_enough():
                # print("sample")
                X_next = self.suggested_sampling()
                
                # print(X_next)
                # Evaluate batch
                fX_next = np.array([[self.f(x)] for x in X_next])
                # print("select")
                self.select_from_candidates(X_next)
                self.solutions_and_obj_after_sampling(X_next, fX_next)
