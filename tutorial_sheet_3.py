# TUTORIAL SHEET 3 - MY CODE

# imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests as rq
from io import StringIO
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

# random seed

np.random.seed(0)

# (1) standard linear regression --------------------------------------------------------------------------------------

# (a) import dataset using pandas

df = pd.read_csv('xray.csv')
x = np.array(df['Distance (mm)'][:])
y = np.array(df['Total absorption'][:])

# (b) perform linear regression [part of coursework submission]

m = len(x)
Sxi = np.sum(x)
Sxi2 = np.sum(x**2)
Syi = np.sum(y)
Syixi = np.dot(x, y) # apparently the same as using np.sum(x*y), which is a bit easier to remember

A = np.array([[Sxi, Sxi2],[m, Sxi]])
b = np.array([Syixi, Syi])

beta = np.linalg.solve(A, b)
beta1, beta2 = beta

print(beta1, beta2)

xfit = np.linspace(0, 6, 200)
yfit = beta1 + beta2*xfit

fig, ax = plt.subplots()
plt.plot(xfit, yfit, color='black')
plt.scatter(x, y)
plt.show()

# (2) higher order regression [part of coursework submission] ----------------------------------------------------------

m = len(x)
Sxi = np.sum(x)
Sxi2 = np.sum(x**2)
Sxi3 = np.sum(x**3)
Sxi4 = np.sum(x**4)
Syi = np.sum(y)
Sxiyi = np.dot(x, y)
Sxi2yi = np.dot(x**2, y)

A2 = np.array([[Sxi2, Sxi3, Sxi4], [Sxi, Sxi2, Sxi3], [m, Sxi, Sxi2]])
b2 = np.array([Sxi2yi, Sxiyi, Syi])

beta = np.linalg.solve(A2, b2)
beta1, beta2, beta3 = beta

print(beta1, beta2, beta3)

xpoly = np.linspace(0, 6, 200)
ypoly = beta1 + beta2*xpoly + beta3*(xpoly**2)

fig, ax = plt.subplots()
plt.plot(xfit, yfit, color = 'black')
plt.plot(xpoly, ypoly, color = 'red')
plt.text(0.5,100,'Pranav Puranam',size=20,zorder=0.,color='#aaaaaa')
plt.scatter(x, y)
plt.savefig('submission_a')
plt.show()

# (3) using scikit-learn to perform regression ------------------------------------------------------------------------

# (a) perform a linear regression

df2 = pd.read_csv('hdpeVel.csv')
df2 = df2.set_index('T/C f/MHz')

freq = df2.columns.values.astype(float)*1e6
temp = df2.index.values.astype(float)
vel = df2.to_numpy()

tot_values = len(freq)*len(temp)

x1grid, x2grid = np.meshgrid(freq, temp) 
Xgrid = np.concatenate([x1grid.reshape([tot_values, 1]), 
	x2grid.reshape([tot_values, 1])], axis=1) 
ygrid = vel.reshape([tot_values, 1])

reg = LinearRegression()
reg.fit(Xgrid, ygrid)

y_lin = reg.predict(Xgrid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], ygrid, marker='x', color='#000000')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], y_lin, marker='o', color='#ff0000')
plt.show()

poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(Xgrid)

print(X_poly.shape)
print(poly.powers_)

reg_poly = LinearRegression()
reg_poly.fit(X_poly, ygrid)

y_poly = reg_poly.predict(X_poly)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], ygrid, marker='x', color='#000000')
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], y_poly, marker='o', color='#0000ff')
plt.show()