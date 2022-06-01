import numpy as np
import math
from scipy.stats import norm, uniform

def elbo(kcat:int, xobs:np.ndarray, sigma, m:np.ndarray, s2:np.ndarray, phi:np.ndarray):
    n = len(xobs)
    t1 = sum(-math.log(2 * math.pi * sigma**2)/2 - mk**2/(2 * sigma**2) for mk in m)
    t2 = n*(-math.log(kcat) - math.log(2 * math.pi)/2)\
        - sum(phi[i, k] * (s2[k] + (m[k] - xobs[i])**2) for k in range(kcat) for i in range(n)) / 2
    t3 = np.sum(phi * np.log(phi))
    t3 = np.where(np.isnan(t3), 0, t3)
    t4 = sum(-math.log(2 * math.pi * s2k)/2 - 1/2 for s2k in s2)
    return t1 + t2 - t3 - t4

def update_cluster_assgn(kcat:int, m:np.ndarray, s2:np.ndarray, xi):
    result = np.array([math.exp(m[k] * xi - (s2[k] + m[k]**2) / 2) for k in range(kcat)])
    return result / sum(result)

def update_center(sigma, phi:np.ndarray, xobs:np.ndarray, k):
    muk = np.dot(phi[:, k], xobs) / (1/(sigma**2) + np.sum(phi[:, k]))
    s2k = 1 / (1/(sigma**2) + np.sum(phi[:, k]))
    return muk, s2k

def initialize(kcat, n, rng:np.random.Generator):
    phi_init = np.reshape(uniform.rvs(size=n * kcat, random_state=rng), (n, kcat))
    phi_sum = phi_init.sum(axis=1)
    return norm.rvs(scale=2, size=kcat, random_state=rng),\
            np.ones(kcat), phi_init / phi_sum[:, np.newaxis]

def cavi(xobs:np.ndarray, kcat, sigma, rng:np.random.Generator=None, tol=1e-4):
    if rng is None:
        rng = np.random.default_rng()
    n = len(xobs)
    m, s2, phi = initialize(kcat, n, rng)
    elbo_list = [0] * 1000
    elbo_list[0] = -math.inf
    iter = 1
    while True:
        for i in range(n):
            phi[i,:] = update_cluster_assgn(kcat, m, s2, xobs[i])
        for k in range(kcat):
            m[k], s2[k] = update_center(sigma, phi, xobs, k)
        elbo_list[iter] = elbo(kcat, xobs, sigma, m, s2, phi)
        if abs(elbo_list[iter] - elbo_list[iter - 1]) < tol:
            break
        iter += 1
    return elbo_list[1:iter+1], m, s2, phi
