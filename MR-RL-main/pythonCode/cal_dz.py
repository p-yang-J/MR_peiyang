from scipy.optimize import least_squares
import numpy as np

# 定义函数，计算预测的z值和真实z值之间的差距
def read_file(file_name):
    with open(file_name, 'r') as f:
        return [float(line.strip()) for line in f]

def fun(x, points):
    a, b, c, r = x
    return [(a - xi)**2 + (b - yi)**2 + (c - zi)**2 - r**2 for xi, yi, zi in points]

def calculate_z(x, y, a, b, c, r):
    inside_sqrt = r**2 - (x - a)**2 - (y - b)**2
    if inside_sqrt < 0:
        raise ValueError("No solutions. The point is outside of the sphere.")
    sqrt_val = np.sqrt(inside_sqrt)
    return c + sqrt_val#c - sqrt_val

# 初始猜测值
x0 = np.array([0, 0, 0, 1])

x_coords = read_file('dx_list.txt')
y_coords = read_file('dy_list.txt')
z_coords = read_file('dz_list.txt')

points = np.array([
    [0, 0, 3000],
    [3000, 3000, 4000],
    [5000, 6000, 5300],
    [-5000, 5000, 3000],
    [2000, 3000, 3800],
    [3000, 3000, 4000],
    [5000, 6000, 6500],
    [5000, 7000, 6000]
])


res = least_squares(fun, x0, args=(points,))

a, b, c, r = res.x


count = 0
for x, y, z in zip(x_coords, y_coords, z_coords):
    theoretical_z = calculate_z(x, y, a, b, c, r)
    if z > theoretical_z:
        count += 1
print(count)