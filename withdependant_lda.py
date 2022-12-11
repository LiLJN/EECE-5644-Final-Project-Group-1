import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sys import float_info
from math import ceil, floor
from scipy.optimize import minimize
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import cm

mydataset = pd.read_excel('mydataset.xlsx')

mydataset_t = mydataset.T

labels = mydataset['Survived']
X0 = mydataset['SibSp']
X1 = mydataset['Parch']
N0 = sum(mydataset['SibSplabel'])
N1 = sum(mydataset['Parchlabel'])

X = np.array([X0, X1]).T

plt.figure(figsize=(10, 8))
plt.plot(X[labels==0, 0], X[labels==0, 1], 'b.', label="Class 0")
plt.plot(X[labels==1, 0], X[labels==1, 1], 'r+', label="Class 1")

plt.xlabel(r"SibSp")
plt.ylabel(r"Parch")
plt.title("Data and True Labels")
plt.legend()
plt.axis('equal')
plt.show()

def perform_lda(X, labels, C=2):
    mu = np.array([np.mean(X[labels == i], axis=0).reshape(-1, 1) for i in range(C)])
    cov = np.array([np.cov(X[labels == i].T) for i in range(C)])
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    idx = lambdas.argsort()[::-1]
    U = U[:, idx]
    w = U[:, 0]
    z = X.dot(w)
    return w, z

w_lda, z = perform_lda(X, labels)
slope = w_lda[1] / w_lda[0]
x_bounds = np.array([np.min(X[:, 0]), np.max(X[:, 0])])

fig = plt.figure(figsize=(6, 16))
x0 = X[labels == 0]
x1 = X[labels == 1]
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(x0[:, 0], x0[:, 1], 'b.', x1[:, 0], x1[:, 1], 'r+')
ax1.plot(x_bounds, slope * x_bounds, c='orange')
z_rec = z.reshape(-1, 1).dot(w_lda.reshape(-1, 1).T)
ax1.plot([z_rec[labels == 0, 0], X[labels == 0, 0]], [z_rec[labels == 0, 1], X[labels == 0, 1]], 'b.:', alpha=0.2)
ax1.plot([z_rec[labels == 1, 0], X[labels == 1, 0]], [z_rec[labels == 1, 1], X[labels == 1, 1]], 'r+:', alpha=0.2)
ax1.set_xlabel(r"$x_1$ Parch")
ax1.set_ylabel(r"$x_2$ SibSp")
ax1.legend(["Class 0", "Class 1", r"$w_{LDA}$"])
ax1.set_aspect('equal')

ax2 = fig.add_subplot(2, 1, 2)
z0 = z[labels == 0]
z1 = z[labels == 1]
ax2.plot(z0, np.zeros(len(z0)), 'b.', z1, np.zeros(len(z1)), 'r+')
ax2.set_xlabel(r"$w_{LDA}^\intercal x$")
plt.show()

