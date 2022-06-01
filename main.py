import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, randint
from cavi_gmm import cavi

rng = np.random.default_rng(101)

kcat = 3
n = 50
sigma = 6

centers = norm.rvs(scale=sigma, size=kcat, random_state=rng)
cats = randint.rvs(0, kcat, size = n, random_state=rng)
xobs = np.array([norm.rvs(loc=centers[c], scale=2, random_state=rng) for c in cats])

elbo_list, m, s2, phi = cavi(xobs, kcat, sigma, rng)

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