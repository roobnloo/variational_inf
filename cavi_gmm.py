import numpy as np
import math
from dataclasses import dataclass
from scipy.stats import norm, uniform


@dataclass
class CaviGmmResult:
    elbo: list
    mean_mat: np.ndarray
    s2_mat: np.ndarray
    phi_arr: np.ndarray


def elbo(kcat: int, xobs: np.ndarray, sigma, m: np.ndarray, s2: np.ndarray,
         phi: np.ndarray):
    n = len(xobs)
    t1 = sum(-math.log(2 * math.pi * sigma**2)/2 - mk**2/(2 * sigma**2)
             for mk in m)
    t2 = n*(-math.log(kcat) - math.log(2 * math.pi)/2)\
        - sum(phi[i, k] * (s2[k] + (m[k] - xobs[i])**2)
              for k in range(kcat) for i in range(n)) / 2
    t3 = np.sum(phi * np.log(phi))
    t3 = np.where(np.isnan(t3), 0, t3)
    t4 = sum(-math.log(2 * math.pi * s2k)/2 - 1/2 for s2k in s2)
    return t1 + t2 - t3 - t4


def update_cluster_assgn(kcat: int, m: np.ndarray, s2: np.ndarray, xi):
    result = np.array([math.exp(m[k] * xi - (s2[k] + m[k]**2) / 2)
                       for k in range(kcat)])
    return result / sum(result)


def update_center(sigma, phi: np.ndarray, xobs: np.ndarray, k):
    muk = np.dot(phi[:, k], xobs) / (1/(sigma**2) + np.sum(phi[:, k]))
    s2k = 1 / (1/(sigma**2) + np.sum(phi[:, k]))
    return muk, s2k


def initialize(kcat, n, rng: np.random.Generator, max_iter):
    phi_init = np.reshape(
        uniform.rvs(size=n * kcat, random_state=rng),
        (n, kcat))
    phi_sum = phi_init.sum(axis=1)
    phi_arr = np.zeros((n, kcat, max_iter))
    phi_arr[:, :, 0] = phi_init / phi_sum[:, np.newaxis]
    mean_mat = np.zeros((kcat, max_iter))
    mean_mat[:, 0] = norm.rvs(scale=2, size=kcat, random_state=rng)
    var_mat = np.zeros((kcat, max_iter))
    var_mat[:, 0] = np.ones(kcat)
    return mean_mat, var_mat, phi_arr


def cavi(xobs: np.ndarray, kcat, sigma,
         rng: np.random.Generator = None, tol=1e-4, max_iter=1_000):
    if rng is None:
        rng = np.random.default_rng()
    n = len(xobs)
    mean_mat, s2_mat, phi_arr = initialize(kcat, n, rng, max_iter)
    elbo_list = [0] * max_iter
    elbo_list[0] = -math.inf
    iter = 1
    while True:
        for i in range(n):
            phi_arr[i, :, iter] = update_cluster_assgn(
                kcat, mean_mat[:, iter-1], s2_mat[:, iter-1], xobs[i])
        for k in range(kcat):
            mean_mat[k, iter], s2_mat[k, iter] = update_center(
                sigma, phi_arr[:, :, iter], xobs, k)
        elbo_list[iter] = elbo(kcat, xobs, sigma, mean_mat[:, iter],
                               s2_mat[:, iter], phi_arr[:, :, iter])
        if abs(elbo_list[iter] - elbo_list[iter - 1]) < tol:
            break
        iter += 1

    return CaviGmmResult(elbo_list[1:iter+1], mean_mat[:, 0:iter],
                         s2_mat[:, 0:iter], phi_arr[:, :, 0:iter])
