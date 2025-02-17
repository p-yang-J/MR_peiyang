import math

def read_file(file_name):
    with open(file_name, 'r') as f:
        return [float(line.strip()) for line in f]

def compute_trajectory_length(x_coords, y_coords, z_coords):
    length = 0.0
    for i in range(1, len(x_coords)):
        dx = x_coords[i] - x_coords[i - 1]
        dy = y_coords[i] - y_coords[i - 1]
        dz = z_coords[i] - z_coords[i - 1]
        length += math.sqrt(dx * dx + dy * dy + dz * dz)
    return length

x_coords = read_file('dx_list.txt')
y_coords = read_file('dy_list.txt')
z_coords = read_file('dz_list.txt')

length = compute_trajectory_length(x_coords, y_coords, z_coords)

print(f"The length of the trajectory is: {length}")
