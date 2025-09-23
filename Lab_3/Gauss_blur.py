import numpy as np
import math

def gaussian_function(x, y, sigma):
    return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

sigma = 1.0
kernel_size = 3
center = math.ceil(kernel_size / 2) - 1
print(center)

gaussian_matrix = np.zeros((kernel_size, kernel_size))
for i in range(kernel_size):
    for j in range(kernel_size):
        x = j - center
        y = i - center
        gaussian_matrix[i, j] = gaussian_function(x, y, sigma)

print("Матрица Гаусса до нормировки:")
print(gaussian_matrix)
print(f"Сумма элементов: {np.sum(gaussian_matrix):.6f}")
