import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

# A simple implemetation of a 2-dimensional lattice

## Basis vectors
basis_vec = np.array([
    [2,0],
    [0,2]
])

x = []
y = []

for i in range(-5, 5):
  for j in range(-5, 5):
    points = i*basis_vec[:,0] + j*basis_vec[:,1]
    x.append(points[0])
    y.append(points[1])

plt.scatter(x, y)
plt.grid()
plt.axis('equal')

plt.savefig('mini_lattice.png', dpi = 300)
