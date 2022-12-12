"""
ACMS 80770-03: Deep Learning with Graphs
Instructor: Navid Shervani-Tabar
Fall 2022
University of Notre Dame
Homework 4: Programming assignment
Problem 1
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=UserWarning)


class kernel:
    def __init__(self, K, R, d, J, lamb_max):
        # -- filter properties
        self.R = float(R)
        self.J = J
        self.K = K
        self.d = d
        self.lamb_max = torch.tensor(lamb_max)

        # -- Half-Cosine kernel
        self.a = R * np.log(lamb_max) / (J - R + 1)
        self.g_hat = lambda lamb: d[0] + d[1] * np.cos(2 * np.pi * (lamb / self.a + 0.5)) if 0 <= -lamb < self.a else 0

    def wavelet(self, lamb, j):
        """
            constructs wavelets ($j\in [2, J]$).
        :param lamb: eigenvalue (analogue of frequency).
        :param j: filter index in the filter bank.
        :return: filter response to input eigenvalues.
        """
        return self.g_hat(np.log(lamb) - self.a * (j - 1) / self.R)

    def scaling(self, lamb):
        """
            constructs scaling function (j=1).
        :param lamb: eigenvalue (analogue of frequency).
        :return: filter response to input eigenvalues.
        """
        b, c = 0, 0
        for k in range(1, self.K + 1):
            b += self.d[k] ** 2
        b = self.R / 2 * b
        for j in range(2, self.J + 1):
            c += abs(self.wavelet(lamb, j) ** 2)
        e = self.R * self.d[0] ** 2 + b - c
        if abs(e) < 1e-8:
            e = 0
        return np.sqrt(e)


# -- define filter-bank
lamb_max = 2
J = 8
filter_bank = kernel(K=1, R=3, d=[0.5, 0.5], J=J, lamb_max=lamb_max)

# -- plot filters+
axis = np.arange(0.01, 2.01, 0.01)

z = []
for x in np.arange(0.01, 2.01, 0.01):
    z.append(filter_bank.scaling(x))
plt.plot(axis, z, label="j = 1")

for j in range(2, J + 1):
    z = []
    for x in np.arange(0.01, 2.01, 0.01):
        z.append(filter_bank.wavelet(x, j))
    plt.plot(axis, z, label=f"j = {j}")

plt.xlabel('Lambda')
plt.ylabel('filters')
plt.legend()
plt.show()
