import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm, randint
from cavi_gmm import cavi

rng = np.random.default_rng(101)


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


def k3n50():
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
    # s2 = result.s2_mat[:, -1]
    phi = result.phi_arr[:, :, -1]

    plt.plot(elbo_list)
    plt.ylabel("ELBO")
    plt.xlabel("Iteration")
    plt.show()

    cm = catmap(m, centers)
    cats_assgn = [cm[i] for i in phi.argmax(axis=1)]
    misclassified = sum(cats_assgn != cats)
    print(misclassified)
    print(misclassified / n)

    c1 = "#E83845"
    c2 = "#288BA8"
    c3 = "#746AB0"
    x = np.linspace(-20, 15, 100)
    fig, ax = plt.subplots(1, 3)

    colmap = {0: c1, 1: c3, 2: c2}
    for i in range(3):
        ax[i].plot(x, norm.pdf(x, result.mean_mat[0, i], 2), color=c1)
        ax[i].plot(x, norm.pdf(x, result.mean_mat[1, i], 2), color=c2)
        ax[i].plot(x, norm.pdf(x, result.mean_mat[2, i], 2), color=c3)
        ax[i].scatter(xobs, [0]*n, c=[colmap[cat] for cat in cats],
                      zorder=100, alpha=0.6)
        ax[i].vlines(centers, 0, 0.22, colors=[c1, c3, c2], linestyles='dotted')
        ax[i].set_title(f"Iteration {i}")
    fig.set_size_inches(15, 5)
    fig.savefig('k350.png', bbox_inches='tight')
    plt.show()


# k3n50()

def k5n500():
    kcat = 5
    n = 500
    sigma = 12

    centers = norm.rvs(scale=sigma, size=kcat, random_state=rng)
    cats = randint.rvs(0, kcat, size=n, random_state=rng)
    xobs = np.array([norm.rvs(loc=centers[c], scale=2, random_state=rng)
                    for c in cats])

    result = cavi(xobs, kcat, sigma, rng=rng)
    elbo_list = result.elbo
    m = result.mean_mat[:, -1]
    # s2 = result.s2_mat[:, -1]
    phi = result.phi_arr[:, :, -1]

    plt.plot(elbo_list)
    plt.ylabel("ELBO")
    plt.xlabel("Iteration")
    plt.savefig('k5n500elbo.png', bbox_inches='tight', facecolor='white')
    plt.show()

    catm = catmap(m, centers)
    cats_assgn = [catm[i] for i in phi.argmax(axis=1)]
    misclassified = sum(cats_assgn != cats)
    print(misclassified)
    print(misclassified / n)

    x = np.linspace(-35, 17, 100)
    fig, ax = plt.subplots(1, 4)

    viridis = cm.get_cmap('viridis', 6)
    coords = range(4)
    itermap = dict(zip(coords, [0, 5, 10, 15]))
    for i in coords:
        for j in range(kcat):
            ax[i].plot(x, norm.pdf(x, result.mean_mat[j, itermap[i]], 2),
                       color=viridis(j))
        ax[i].scatter(xobs, [0]*n,
                      zorder=100, alpha=0.2)
        ax[i].vlines(centers, 0, 0.22, colors=viridis(range(kcat)),
                     linestyles='dotted')
        ax[i].set_title(f"Iteration {itermap[i]}")
    fig.set_size_inches(18, 5)
    fig.savefig('k5n500.png', bbox_inches='tight', facecolor='white')
    plt.show()

k5n500()
