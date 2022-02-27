import scipy
import numpy as np
import qmcpy as qp
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
font = font_manager.FontProperties(style='normal', size=35)


class Grid:     
    def __init__(self, randomize=False, seed=None):
        self.randomize = randomize
        if randomize:
            np.random.seed(seed)

    def gen_samples(self, n):
        grid = [(i+1)/(n+1) for i in range(n)]
        if self.randomize:
            return np.random.permutation(grid)
        else:
            return grid

SEED = 100
np.random.seed(SEED)

RUNS = 20

d = 5

# synthetic d-dimensional least squares problem
wopt = np.random.randn(d) 

# data distribution
# x ~ N(0,I_d)
# y ~ N(x'wopt, 1)
# goal is to learn w, which is to minimize E[0.5*(x'w - y)^2]

# run sgd with fixed step size for now
etas = [0.01, 0.05]

# total number of iterations/samples we shall take
T = 1024 * 4 *16
n = T 

# init
w0 = np.zeros(d)

# generate our samples to be used in an online fashion
gaussian_samples = {}

# Quasi Random Distributions
qrng_distribs = [
    (lambda s: qp.IIDStdUniform(dimension=d+1, seed=s)),
    (lambda s: qp.Lattice(dimension=d+1, randomize=True, seed=s)),
    (lambda s: qp.Sobol(dimension=d+1, randomize=True, seed=s)),
    (lambda s: qp.Halton(dimension=d+1, randomize=True, seed=s))] # ,
   #  Grid(randomize=True, seed=SEED)]

qrng_names = ["IID Uniform",
              "Shifted Lattice",
              "Scrambled Sobol",
              "Randomized Halton" ] #,
              #"Grid"]

for i, (distrib, distrib_name) in enumerate(zip(qrng_distribs, qrng_names)):
    data = []
    for run in range(RUNS):
        qmc_samples = distrib(SEED + run).gen_samples(n)
        data_run = []
        for q in qmc_samples:
            xi = scipy.stats.norm.ppf(q[0:d])
            mu = xi.T@wopt
            # inverse transform to get qmc gaussian samples
            yi = scipy.stats.norm.ppf(q[d], loc=mu, scale=1)
            data_run.append((xi, yi))
        data.append(data_run)
    gaussian_samples[distrib_name] = data


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
for rng_name, samples in gaussian_samples.items():
    subopt_list = np.zeros((RUNS, n))
    for run in range(RUNS):
        wk = w0
        subopt = []
        full_gradient = np.zeros(d)
        sub_gradient = np.zeros(d)
        for (xi, yi) in samples[run]:
            full_gradient += (xi.T@wk - yi)*xi
        full_gradient /= len(samples[run])
        for i, (xi, yi) in enumerate(samples[run]):
            sub_gradient += (xi.T@wk - yi)*xi

            qmc_lhs = np.linalg.norm(sub_gradient - (i+1) * full_gradient) / (i+1)
            subopt.append(qmc_lhs)
        subopt_list[run] = subopt

    ax.plot(np.mean(subopt_list, axis=0), label=rng_name)
    ax.fill_between(list(range(n)), np.percentile(subopt_list, 25, axis=0), np.percentile(subopt_list, 75, axis=0), alpha=0.3)
    ax.set_title("qmc bound")
    # ax.set_yscale("log")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1)
    ax.set_xlabel("n",fontsize=35)
    ax.set_ylabel("LHS of QMC",fontsize=35)
    ax.legend(loc='upper right', borderaxespad=0., prop=font)

plt.tight_layout()
# plt.show()
plt.savefig('bound.png',transparent=True)