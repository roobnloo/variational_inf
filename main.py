import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, randint
from cavi_gmm import cavi

rng = np.random.default_rng(101)

kcat = 3
n = 50
sigma = 6

centers = norm.rvs(scale=sigma, size=kcat, random_state=rng)
cats = randint.rvs(0, kcat, size=n, random_state=rng)
xobs = np.array([norm.rvs(loc=centers[c], scale=2, random_state=rng)
                 for c in cats])

result = cavi(xobs, kcat, sigma, rng=rng)
elbo_list = result.elbo
m = result.mean_mat[:, -1]
s2 = result.s2_mat[:, -1]
phi = result.phi_arr[:, :, -1]

plt.plot(elbo_list)
plt.ylabel("ELBO")
plt.xlabel("Iteration")
plt.show()


def catmap(m: np.ndarray, centers: np.ndarray):
    mmask = np.ma.array(m, mask=False)
    cenmask = np.ma.array(centers, mask=False)
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

c1 = "#E83845"
c2 = "#288BA8"
c3 = "#746AB0"
x = np.linspace(-20, 15, 100)
fig, ax = plt.subplots()
ax.plot(x, norm.pdf(x, m[0], 2), color=c1)
ax.plot(x, norm.pdf(x, m[1], 2), color=c2)
ax.plot(x, norm.pdf(x, m[2], 2), color=c3)
colmap = {0: c1, 1: c3, 2: c2}
ax.scatter(xobs, [0]*n, c=[colmap[cat] for cat in cats], zorder=100, alpha=0.6)
ax.vlines(centers, 0, 0.22, colors=[c1, c3, c2], linestyles='dotted')
ax.set_title("Gaussian Mixture Model via Variational Inference")
plt.show()
