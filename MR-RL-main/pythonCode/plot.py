import numpy as np
import matplotlib.pyplot as plt

with open('dx_list.txt', 'r') as f:
    x = np.array([float(line.strip()) for line in f])

with open('dy_list.txt', 'r') as f:
    y = np.array([float(line.strip()) for line in f])


fig, ax = plt.subplots()
ax.plot(x, y, label='Original data')

coefficients = np.polyfit(x, y, 15)
poly = np.poly1d(coefficients)


y_fit = poly(x)
ax.plot(x, y_fit, 'r-', linewidth=2, label='Fit')
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.legend()

plt.show()
