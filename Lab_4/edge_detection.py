import numpy as np
import cv2

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])

def create_gauss_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp((-1) * (x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel

def normalize_kernel(kernel):
    kernel_sum = np.sum(kernel)
    normalized = kernel / kernel_sum
    return normalized

def apply_gauss_filter_manual(image, kernel):
    filtered_image = image.copy()
    kernel_size = kernel.shape[0]
    padding = kernel_size // 2
    height, width = image.shape

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            start_y = y - padding
            end_y = y + padding + 1
            start_x = x - padding
            end_x = x + padding + 1

            window = image[start_y:end_y, start_x:end_x]

            new_value = 0
            for k in range(kernel_size):
                for l in range(kernel_size):
                    new_value += window[k, l] * kernel[k, l]

            filtered_image[y, x] = new_value

    return filtered_image

def read_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {image_path}")
        return None

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("Исходное изображение загружено и преобразовано в ЧБ")
    print(f"Размер изображения: {gray_image.shape}")

    return gray_image

def apply_gaussian_blur(image, kernel_size=5, sigma=1.0, use_cv2=False):
    print(f"Применение размытия Гаусса: kernel_size={kernel_size}, sigma={sigma}")

    if use_cv2:
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    else:
        kernel = create_gauss_kernel(kernel_size, sigma)
        kernel = normalize_kernel(kernel)
        blurred = apply_gauss_filter_manual(image, kernel)
        blurred = blurred.astype(np.uint8)

    print("Размытие Гаусса применено")
    return blurred

def compute_gradients(image):
    height, width = image.shape

    Gx = np.zeros((height, width), dtype=np.float32)
    Gy = np.zeros((height, width), dtype=np.float32)
    magnitude = np.zeros((height, width), dtype=np.float32)
    angle = np.zeros((height, width), dtype=np.float32)

    print(f"Вычисление градиентов для изображения {height}x{width}")

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Вычисляем градиенты через свертку
            gx_val = 0
            gy_val = 0

            for i in range(3):
                for j in range(3):
                    pixel_value = image[y + i - 1, x + j - 1]
                    gx_val += pixel_value * sobel_x[i, j]
                    gy_val += pixel_value * sobel_y[i, j]

            Gx[y, x] = gx_val
            Gy[y, x] = gy_val

            # Вычисляем длину градиента
            magnitude[y, x] = np.sqrt(gx_val ** 2 + gy_val ** 2)

            # Вычисляем угол градиента
            if gx_val != 0:
                angle_rad = np.arctan2(gy_val, gx_val)
                angle_deg = np.degrees(angle_rad)

                # Приводим угол к диапазону 0-180 градусов
                if angle_deg < 0:
                    angle_deg += 180

                angle[y, x] = angle_deg
            else:
                angle[y, x] = 90  # Если Gx = 0, то вертикальное направление, то есть по иксу функция яркости не меняется

    print("Градиенты вычислены")
    return Gx, Gy, magnitude, angle

def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    suppressed = np.zeros((height, width), dtype=np.float32)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            current_angle = angle[y, x]
            current_magnitude = magnitude[y, x]

            if (0 <= current_angle < 22.5) or (157.5 <= current_angle <= 180):
                direction = 0  # Горизонтальное
            elif 22.5 <= current_angle < 67.5:
                direction = 45  # Диагональ 45 градусов
            elif 67.5 <= current_angle < 112.5:
                direction = 90  # Вертикальное
            elif 112.5 <= current_angle < 157.5:
                direction = 135  # Диагональ 135 градусов
            else:
                direction = 0

            if direction == 0:  # Горизонтальное
                neighbor1 = magnitude[y, x + 1]  # справа
                neighbor2 = magnitude[y, x - 1]  # слева
            elif direction == 45:  # Диагональ 45 градусов
                neighbor1 = magnitude[y + 1, x + 1]  # слева сверху
                neighbor2 = magnitude[y - 1, x - 1]  # справа снизу
            elif direction == 90:  # Вертикальное
                neighbor1 = magnitude[y + 1, x]  # снизу
                neighbor2 = magnitude[y - 1, x]  # сверху
            elif direction == 135:  # Диагональ 135 градусов
                neighbor1 = magnitude[y + 1, x - 1]  # слева снизу
                neighbor2 = magnitude[y - 1, x + 1]  # справа сверху

            if current_magnitude >= neighbor1 and current_magnitude >= neighbor2:
                suppressed[y, x] = current_magnitude
            else:
                suppressed[y, x] = 0

    return suppressed

def double_threshold_filtering(magnitude, low_ratio=0.03, high_ratio=0.15):
    """Двойная пороговая фильтрация"""
    height, width = magnitude.shape
    result = np.zeros((height, width), dtype=np.uint8)


    max_grad = np.max(magnitude)
    low_threshold = max_grad * low_ratio
    high_threshold = max_grad * high_ratio

    print(f"Максимальный градиент: {max_grad:.2f}")
    print(f"Нижний порог ({low_ratio * 100}%): {low_threshold:.2f}")
    print(f"Верхний порог ({high_ratio * 100}%): {high_threshold:.2f}")


    strong_edges = np.zeros((height, width), dtype=bool)
    weak_edges = np.zeros((height, width), dtype=bool)

    print("Применение двойной пороговой фильтрации...")

    for y in range(height):
        for x in range(width):
            grad_value = magnitude[y, x]

            if grad_value >= high_threshold:
                strong_edges[y, x] = True
                result[y, x] = 255
            elif grad_value >= low_threshold:
                weak_edges[y, x] = True
                result[y, x] = 128
            else:
                result[y, x] = 0


    final_result = result.copy()

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if weak_edges[y, x]:
                has_strong_neighbor = False


                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if strong_edges[ny, nx]:
                                has_strong_neighbor = True
                                break
                    if has_strong_neighbor:
                        break

                if has_strong_neighbor:
                    final_result[y, x] = 255

                    strong_edges[y, x] = True

                else:
                    final_result[y, x] = 0

    print("Двойная пороговая фильтрация завершена")
    return final_result

def canny_implementation(image_path, blur_kernel_size=5, blur_sigma=1.0,
                         use_cv2_blur=False, low_ratio=0.03, high_ratio=0.15):
    print("=" * 60)
    print("АЛГОРИТМ КАННИ")
    print("=" * 60)


    print("\n--- ЗАДАНИЕ 1 ---")
    print("Чтение изображения, преобразование в ЧБ и размытие Гаусса")
    gray_image = read_and_convert_to_grayscale(image_path)
    if gray_image is None:
        return

    cv2.imshow('1. Исходное ЧБ изображение', gray_image)
    cv2.waitKey(500)

    blurred_image = apply_gaussian_blur(gray_image, blur_kernel_size, blur_sigma, use_cv2_blur)

    cv2.imshow('2. После размытия Гаусса', blurred_image)
    cv2.waitKey(500)


    print("\n--- ЗАДАНИЕ 2 ---")
    print("Вычисление матриц длин и углов градиентов")
    Gx, Gy, magnitude, angle = compute_gradients(blurred_image)

    magnitude_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude_normalized = magnitude_normalized.astype(np.uint8)
    cv2.imshow('3. Длина градиента (магнитуда)', magnitude_normalized)
    cv2.waitKey(500)

    angle_normalized = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX)
    angle_normalized = angle_normalized.astype(np.uint8)
    cv2.imshow('4. Угол градиента', angle_normalized)
    cv2.waitKey(500)


    print(f"\nМатрица длин градиентов (первые 10x10):")
    print(magnitude[:10, :10].astype(int))
    print(f"\nМатрица углов градиентов (первые 10x10):")
    print(angle[:10, :10].astype(int))


    print("\n--- ЗАДАНИЕ 3 ---")
    print("Подавление немаксимумов")
    suppressed = non_maximum_suppression(magnitude, angle)
    suppressed_normalized = cv2.normalize(suppressed, None, 0, 255, cv2.NORM_MINMAX)
    suppressed_normalized = suppressed_normalized.astype(np.uint8)
    cv2.imshow('5. После подавления немаксимумов', suppressed_normalized)
    cv2.waitKey(500)

    print("\n--- ЗАДАНИЕ 4 ---")
    print("Двойная пороговая фильтрация")
    final_edges = double_threshold_filtering(suppressed, low_ratio, high_ratio)
    cv2.imshow('6. После двойной пороговой фильтрации', final_edges)
    cv2.waitKey(500)

    edges_cv2 = cv2.Canny(blurred_image, 68, 88)
    cv2.imshow('7. Canny от OpenCV (для сравнения)', edges_cv2)
    cv2.waitKey(500)

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    cv2.imwrite('1_original.jpg', gray_image)
    cv2.imwrite('2_gaussian.jpg', blurred_image)
    cv2.imwrite('3_gradient_magnitude.jpg', magnitude_normalized)
    cv2.imwrite('4_gradient_angle.jpg', angle_normalized)
    cv2.imwrite('5_non_max_suppression.jpg', suppressed_normalized)
    cv2.imwrite('6_final_edges.jpg', final_edges)
    cv2.imwrite('7_opencv_canny.jpg', edges_cv2)

    print("\nРезультаты сохранены в файлы:")
    print("1_original.jpg - исходное ЧБ изображение")
    print("2_gaussian.jpg - после размытия Гаусса")
    print("3_gradient_magnitude.jpg - длина градиента")
    print("4_gradient_angle.jpg - угол градиента")
    print("5_non_max_suppression.jpg - после подавления немаксимумов")
    print("6_final_edges.jpg - финальные границы (ручной алгоритм)")
    print("7_opencv_canny.jpg - границы от OpenCV")

    print("\nНажмите любую клавишу для выхода...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def edge_deteciton():

    image_path = r'C:\Pics\apples.jpg'


    canny_implementation(
        image_path=image_path,
        blur_kernel_size=5,
        blur_sigma=5,
        use_cv2_blur=False,
        low_ratio=0.21,
        high_ratio=0.27
    )

edge_deteciton()
    