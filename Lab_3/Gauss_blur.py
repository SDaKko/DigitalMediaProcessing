import numpy as np
import math
import cv2


def gaussian_function(x, y, sigma):
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


sigma = 3.0
kernel_size = 5
center = math.ceil(kernel_size / 2) - 1
print(center)

gaussian_matrix = np.zeros((kernel_size, kernel_size))
for i in range(kernel_size):
    for j in range(kernel_size):
        x = j - center
        y = i - center
        gaussian_matrix[i, j] = gaussian_function(x, y, sigma)

print("Матрица Гаусса до нормировки:")
if kernel_size == 7:
    for i in range(kernel_size):
        for j in range(kernel_size):
            print(gaussian_matrix[i, j], end='  ')
        print()
else:
    print(gaussian_matrix)
print(f"Сумма элементов: {np.sum(gaussian_matrix):.6f}")


def normalize_kernel(kernel):
    return kernel / np.sum(kernel)


normalized_matrix = normalize_kernel(gaussian_matrix)

print("\nМатрица Гаусса после нормировки:")

if kernel_size == 7:
    for i in range(kernel_size):
        for j in range(kernel_size):
            print(normalized_matrix[i, j], end='  ')
        print()
else:
    print(normalized_matrix)
print(f"Сумма элементов: {np.sum(normalized_matrix):.6f}")


def Gauss_filter(image, ker):
    height, width = image.shape
    pad = kernel_size // 2
    res_matrix = image.copy()

    print("Высота и ширина изображения:", height, width)

    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            region = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            res_matrix[i, j] = np.sum(region * ker)

    print(res_matrix)

    return res_matrix


img = cv2.imread(r'C:\Pics\waterfall.gif', cv2.IMREAD_GRAYSCALE)


print(img)
cv2.imshow("original_img", img)
cv2.waitKey(1000)

img_after_hand_blur = Gauss_filter(img, normalized_matrix)

cv2.imshow("img_after_hand_blur", img_after_hand_blur)
cv2.waitKey(1000)

blurred = cv2.GaussianBlur(img, (5, 5), 3.0)
cv2.imshow("img_after_cv_blur", blurred)
cv2.waitKey(0)




