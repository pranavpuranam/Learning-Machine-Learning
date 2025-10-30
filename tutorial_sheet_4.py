# TUTORIAL SHEET 4 - MY CODE

# imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# random seed

np.random.seed(0)

# (1) scaling ----------------------------------------------------------------------------------------------------------

tensile_df = pd.read_csv('tensile_strength.csv')

t = np.array(tensile_df['Temperature (deg C)'][:])
s = np.array(tensile_df['Ultimate tensile strength (Pa)'][:])

t_mean = np.mean(t)
t_std = np.std(t)
s_mean = np.mean(s)
s_std = np.std(s)

t_scale = (t - t_mean)/(t_std)
s_scale = (s - s_mean)/(s_std)

fig, ax = plt.subplots()
plt.hist(s_scale)
plt.show()

scArray = np.array([[t_mean, s_mean],[t_std, s_std]])
np.savetxt('scaleParams.txt', scArray)

loadedScales = np.loadtxt('scaleParams.txt')

# (2) plotting linear discriminant functions ---------------------------------------------------------------------------

# (3) training linear discriminant functions ---------------------------------------------------------------------------

# (4) plotting classification areas ------------------------------------------------------------------------------------

# (5) conclusions ------------------------------------------------------------------------------------------------------

"""
This tutorial has demonstrated linear discriminant functions and aimed to give you an appreciation
for the way they behave. The examples we have looked at are fairly abstract, with us focusing on
the mathematical and computational aspects of their use rather than applying them to any specific
application or data. This is really a reflection on how they are more fundamental tools which are useful
to understand and upon which other methods – SVMs and neural networks in particular – are based.
"""


"""
fig, ax = plt.subplots() 
plt.hist(s_scale) 
plt.show()
---
scArray = np.array([[t_mean, s_mean],[t_std, s_std]]) 
np.savetxt('scaleParams.txt',scArray)
---
loadedScales = np.loadtxt('scaleParams.txt')
---
from sklearn.datasets import make_blobs

# Generate data with two distinct clusters
X, yt = make_blobs(n_samples=400, centers=2, n_features=2, cluster_std=2.0, random_state=42)
# Convert labels to -1 and 1
y = yt * 2 - 1

# Initialize weights and bias as floats
w0 = 0.0
w = np.array([0.0, 1.0], dtype=float)  

# Learning rate
delta = #come up with something!
iterations = #likewise!
---
# Compute decision function and error
g = X @ w + w0
res = g - y
E = np.sum(res**2)

# Compute gradients
dgdw = X.T @ res
dgdw0 = np.sum(res)

---
import sys
if a1.shape != (3, 1):
  print("Error!! Shape of a1 is incorrect.")
  sys.exit()
if Xgrid.shape != (npx*npy, 2):
  print("Error!! Shape of Xgrid is incorrect.")
  sys.exit()

#Ygrid is defined as the same as Xgrid, except it has 1  
#at the beginning - this therefore adds a column of ones to the left 
Ygrid = np.concatenate([np.ones([npx * npy,1]), Xgrid],axis=1)

#calculate each of the five functions as before 
g1 = np.matmul(Ygrid,a1) 
g2 = np.matmul(Ygrid,a2) 
g3 = np.matmul(Ygrid,a3) 
g4 = np.matmul(Ygrid,a4) 
g5 = np.matmul(Ygrid,a5)

#combine all five functions together 
gconc = np.concatenate([g1, g2, g3, g4, g5],axis=1)

#find which of the values is largest for each row - this 
#corresponds to which i has the largest gi(x) which is what
#we want (see definition in the notes)
omega=np.argmax(gconc,axis=1)

#put back onto 2D grid so it can easily be plotted 
omega = np.reshape(omega, [npx, npy]) 
"""