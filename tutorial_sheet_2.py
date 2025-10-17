# TUTORIAL SHEET 2 - MY CODE

# imports

import numpy as np
import matplotlib.pyplot as plt

# random seed

np.random.seed(5)

# (1) prior, posterior, and likelihood calculation

# (a) defining the gaussian function

def gaussian(x, mu, sig):
	return((1/(sig * np.sqrt(2 * np.pi)))*np.exp(-(((x - mu)/sig) ** 2)/2))

# print(gaussian(1, 0, 0.5))

# (b) define two probability distributions


# (2) classification with bayes

"""
fig, ax = plt.subplots() 
plt.plot(...)
---
clf = GaussianNB()
clf.fit(X_train, y_train)
---
classVals = clf.predict(Xgrid)
classVals = np.reshape(classVals, [200, 200])
---
nTot = len(y_test) 
nMatch = 0 
for i in range(len(y_test)):
	if y_test[i] == y_test_model[i]:
		nMatch += 1

print(100 * nMatch / nTot)
---
probGrid = np.reshape(probVals[:, 0], [200, 200])
"""