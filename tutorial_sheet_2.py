# TUTORIAL SHEET 2 - MY CODE

# imports

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# random seed

np.random.seed(5)

# (1) prior, posterior, and likelihood calculation

# (a) defining the gaussian function

def gaussian(x, mu, sig):
	return((1/(sig * np.sqrt(2 * np.pi)))*np.exp(-(((x - mu)/sig) ** 2)/2))

# print(gaussian(1, 0, 0.5))

# (b) define two probability distributions

x = np.linspace(-10, 20, 200)

pxw1 = gaussian(x, 2, 1.5) + gaussian(x, 7, 0.5)
pxw1 = pxw1/np.trapz(pxw1, x)

pxw2 = gaussian(x, 8, 2.5) + gaussian(x, 3.5, 1)
pxw2 = pxw2/np.trapz(pxw2, x)

fix, ax = plt.subplots()

plt.plot(x, pxw1)
plt.plot(x, pxw2)
# plt.show()

# (c) calculate the posterior distribution

pw1 = 0.9
pw2 = 0.1

px = (pxw1 * pw1) + (pxw2 * pw2)

pw1x = (pxw1 * pw1) / (px)
pw2x = (pxw2 * pw2) / (px)

fix, ax = plt.subplots()

plt.xlim(-3, 15)
plt.plot(x, pw1x)
plt.plot(x, pw2x)
# plt.show()

# (2) classification with bayes

X, y = datasets.make_classification(n_samples = 1000, n_features = 2, n_informative = 2, n_redundant = 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

clf = GaussianNB() # this is effectively what we did in Q1, but in a function
clf.fit(X_train, y_train)

r = 3
def gen_sample_grid(npx=200, npy=200, limit=1):
    x1line = np.linspace(-limit, limit, npx)
    x2line = np.linspace(-limit, limit, npy)
    x1grid, x2grid = np.meshgrid(x1line, x2line)
    Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
    return Xgrid,x1line,x2line

Xgrid, x1line, x2line = gen_sample_grid(limit = r)

classVals = clf.predict(Xgrid)
classVals = np.reshape(classVals, [200, 200]) # predict the value at each point in the 200x200 grid

fig, ax = plt.subplots()
plt.contourf(x1line, x2line, classVals) # plot expected classification using a contour plot

ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
plt.xlim(-r, r)
plt.ylim(-r, r)

plt.show()

y_test_model = clf.predict(X_test)

nTot = len(y_test) 
nMatch = 0 
for i in range(len(y_test)):
	if y_test[i] == y_test_model[i]:
		nMatch += 1

print(100 * nMatch / nTot)

probVals = clf.predict_proba(Xgrid)
probGrid = np.reshape(probVals[:, 0], [200, 200])

fig, ax = plt.subplots()
plt.contourf(x1line,x2line,probGrid)

ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
plt.xlim(-r, r)
plt.ylim(-r, r)

plt.show()