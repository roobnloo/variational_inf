import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, randint, uniform

rng = np.random.default_rng(101)

kcat = 3
n = 50
sigma = 6

centers = norm.rvs(scale=sigma, size=kcat, random_state=rng)
cats = randint.rvs(0, kcat, size = n, random_state=rng)
xobs = np.array([norm.rvs(loc=centers[c], scale=2, random_state=rng) for c in cats])

def elbo(m:np.ndarray, s2:np.ndarray, phi:np.ndarray):
    t1 = sum(-math.log(2 * math.pi * sigma**2)/2 - mk**2/(2 * sigma**2) for mk in m)
    t2 = n*(-math.log(kcat) - math.log(2 * math.pi)/2)\
        - sum(phi[i, k] * (s2[k] + (m[k] - xobs[i])**2) for k in range(kcat) for i in range(n)) / 2
    t3 = np.sum(phi * np.log(phi))
    t3 = np.where(np.isnan(t3), 0, t3)
    t4 = sum(-math.log(2 * math.pi * s2k)/2 - 1/2 for s2k in s2)
    return t1 + t2 - t3 - t4

def update_cluster_assgn(m:np.ndarray, s2:np.ndarray, xi):
    result = np.array([math.exp(m[k] * xi - (s2[k] + m[k]**2) / 2) for k in range(kcat)])
    return result / sum(result)

def update_center(phi:np.ndarray, xobs:np.ndarray, k):
    muk = np.dot(phi[:, k], xobs) / (1/(sigma**2) + np.sum(phi[:, k]))
    s2k = 1 / (1/(sigma**2) + np.sum(phi[:, k]))
    return muk, s2k

def initialize():
    phi_init = np.reshape(uniform.rvs(size=n * kcat, random_state=rng), (n, kcat))
    phi_sum = phi_init.sum(axis=1)
    return norm.rvs(scale=2, size=kcat, random_state=rng),\
            np.ones(kcat), phi_init / phi_sum[:, np.newaxis]

def cavi(xobs:np.ndarray, kcat, tol=1e-4):
    m, s2, phi = initialize()
    elbo_list = [0] * 1000
    elbo_list[0] = -math.inf
    iter = 1
    while True:
        for i in range(n):
            phi[i,:] = update_cluster_assgn(m, s2, xobs[i])
        for k in range(kcat):
            m[k], s2[k] = update_center(phi, xobs, k)
        elbo_list[iter] = elbo(m, s2, phi)
        if abs(elbo_list[iter] - elbo_list[iter - 1]) < tol:
            break
        iter += 1
    return elbo_list[1:iter+1], m, s2, phi

elbo_list, m, s2, phi = cavi(xobs, kcat)

plt.plot(elbo_list)
plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.show()

def catmap(m:np.ndarray, centers:np.ndarray):
    mmask = np.ma.array(m, mask = False)
    cenmask = np.ma.array(centers, mask = False)
    cm = dict()
    while not all(mmask.mask):
        marg = mmask.argmax()
        cenarg = cenmask.argmax()
        cm[marg] = cenarg
        mmask.mask[marg] = True
        cenmask.mask[cenarg] = True
    return cm

cm = catmap(m, centers)
cats_assgn = [cm[i] for i in phi.argmax(axis=1)]
misclassified = sum(cats_assgn != cats)
print(misclassified)
print(misclassified / n)

x = np.linspace(-20, 15, 100)
fig, ax = plt.subplots()
ax.plot(x, norm.pdf(x, m[0], 2), color="#E83845")
ax.plot(x, norm.pdf(x, m[1], 2), color="#288BA8")
ax.plot(x, norm.pdf(x, m[2], 2), color="#746AB0")
colmap = {0:"#E83845", 1:"#746AB0", 2:"#288BA8"}
ax.scatter(xobs, [0]*n, c=[colmap[cat] for cat in cats], zorder = 100, alpha=0.6)
ax.vlines(centers, 0, 0.22, colors = ["#E83845", "#746AB0", "#288BA8"], linestyles='dotted')
ax.set_title("Gaussian Mixture Model via Variational Inference")
plt.show()