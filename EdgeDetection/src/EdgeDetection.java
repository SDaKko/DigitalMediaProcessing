import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.highgui.HighGui;

public class EdgeDetection {

    private static final float[][] SOBEL_X = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
    };

    private static final float[][] SOBEL_Y = {
            {-1, -2, -1},
            {0, 0, 0},
            {1, 2, 1}
    };

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        edgeDetection();
    }

    public static void edgeDetection() {
        String imagePath = "C:\\Pics\\apples.jpg";

        cannyImplementation(
                imagePath,
                5,
                5.0,
                false,
                0.21,
                0.27
        );
    }

    public static void cannyImplementation(String imagePath, int blurKernelSize, double blurSigma,
                                           boolean useCv2Blur, double lowRatio, double highRatio) {
        System.out.println("=".repeat(60));
        System.out.println("АЛГОРИТМ КАННИ");
        System.out.println("=".repeat(60));

        // Шаг 1: Чтение изображения, преобразование в ЧБ и размытие Гаусса
        System.out.println("\n--- ЗАДАНИЕ 1 ---");
        System.out.println("Чтение изображения, преобразование в ЧБ и размытие Гаусса");

        Mat grayImage = readAndConvertToGrayscale(imagePath);
        if (grayImage.empty()) {
            return;
        }

        showImage("1. Исходное ЧБ изображение", grayImage, 500);

        Mat blurredImage = applyGaussianBlur(grayImage, blurKernelSize, blurSigma, useCv2Blur);

        showImage("2. После размытия Гаусса", blurredImage, 500);

        // Шаг 2: Вычисление градиентов
        System.out.println("\n--- ЗАДАНИЕ 2 ---");
        System.out.println("Вычисление матриц длин и углов градиентов");

        GradientResult gradients = computeGradients(blurredImage);
        Mat Gx = gradients.Gx;
        Mat Gy = gradients.Gy;
        Mat magnitude = gradients.magnitude;
        Mat angle = gradients.angle;

        Mat magnitudeNormalized = new Mat();
        Core.normalize(magnitude, magnitudeNormalized, 0, 255, Core.NORM_MINMAX);
        magnitudeNormalized.convertTo(magnitudeNormalized, CvType.CV_8UC1);
        showImage("3. Длина градиента (магнитуда)", magnitudeNormalized, 500);

        Mat angleNormalized = new Mat();
        Core.normalize(angle, angleNormalized, 0, 255, Core.NORM_MINMAX);
        angleNormalized.convertTo(angleNormalized, CvType.CV_8UC1);
        showImage("4. Угол градиента", angleNormalized, 500);

        System.out.println("\nМатрица длин градиентов (первые 10x10):");
        printMatRegion(magnitude, 10, 10);

        System.out.println("\nМатрица углов градиентов (первые 10x10):");
        printMatRegion(angle, 10, 10);

        // Шаг 3: Подавление немаксимумов
        System.out.println("\n--- ЗАДАНИЕ 3 ---");
        System.out.println("Подавление немаксимумов");

        Mat suppressed = nonMaximumSuppression(magnitude, angle);
        Mat suppressedNormalized = new Mat();
        Core.normalize(suppressed, suppressedNormalized, 0, 255, Core.NORM_MINMAX);
        suppressedNormalized.convertTo(suppressedNormalized, CvType.CV_8UC1);
        showImage("5. После подавления немаксимумов", suppressedNormalized, 500);

        // Шаг 4: Двойная пороговая фильтрация
        System.out.println("\n--- ЗАДАНИЕ 4 ---");
        System.out.println("Двойная пороговая фильтрация");

        Mat finalEdges = doubleThresholdFiltering(suppressed, lowRatio, highRatio);
        showImage("6. После двойной пороговой фильтрации", finalEdges, 500);

        // Сравнение с OpenCV Canny
        Mat edgesCv2 = new Mat();
        Imgproc.Canny(blurredImage, edgesCv2, 68, 88);
        showImage("7. Canny от OpenCV (для сравнения)", edgesCv2, 500);

        System.out.println("\n" + "=".repeat(60));
        System.out.println("РЕЗУЛЬТАТЫ");
        System.out.println("=".repeat(60));

        Imgcodecs.imwrite("1_original.jpg", grayImage);
        Imgcodecs.imwrite("2_gaussian.jpg", blurredImage);
        Imgcodecs.imwrite("3_gradient_magnitude.jpg", magnitudeNormalized);
        Imgcodecs.imwrite("4_gradient_angle.jpg", angleNormalized);
        Imgcodecs.imwrite("5_non_max_suppression.jpg", suppressedNormalized);
        Imgcodecs.imwrite("6_final_edges.jpg", finalEdges);
        Imgcodecs.imwrite("7_opencv_canny.jpg", edgesCv2);

        System.out.println("\nРезультаты сохранены в файлы:");
        System.out.println("1_original.jpg - исходное ЧБ изображение");
        System.out.println("2_gaussian.jpg - после размытия Гаусса");
        System.out.println("3_gradient_magnitude.jpg - длина градиента");
        System.out.println("4_gradient_angle.jpg - угол градиента");
        System.out.println("5_non_max_suppression.jpg - после подавления немаксимумов");
        System.out.println("6_final_edges.jpg - финальные границы (ручной алгоритм)");
        System.out.println("7_opencv_canny.jpg - границы от OpenCV");

        System.out.println("\nНажмите любую клавишу для выхода...");
        HighGui.waitKey(0);
        HighGui.destroyAllWindows();
    }

    public static Mat readAndConvertToGrayscale(String imagePath) {
        Mat image = Imgcodecs.imread(imagePath);
        if (image.empty()) {
            System.out.println("Ошибка: не удалось загрузить изображение " + imagePath);
            return new Mat();
        }

        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        System.out.println("Исходное изображение загружено и преобразовано в ЧБ");
        System.out.println("Размер изображения: " + grayImage.rows() + "x" + grayImage.cols());

        return grayImage;
    }

    public static float[][] createGaussKernel(int size, double sigma) {
        float[][] kernel = new float[size][size];
        int center = size / 2;
        double twoPiSigmaSq = 2 * Math.PI * sigma * sigma;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                double x = i - center;
                double y = j - center;
                kernel[i][j] = (float) ((1.0 / twoPiSigmaSq) * Math.exp(-(x * x + y * y) / (2 * sigma * sigma)));
            }
        }

        return kernel;
    }

    public static float[][] normalizeKernel(float[][] kernel) {
        int size = kernel.length;
        float sum = 0;

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sum += kernel[i][j];
            }
        }

        float[][] normalized = new float[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                normalized[i][j] = kernel[i][j] / sum;
            }
        }

        return normalized;
    }

    public static Mat applyGaussFilterManual(Mat image, float[][] kernel) {
        int kernelSize = kernel.length;
        int padding = kernelSize / 2;
        int height = image.rows();
        int width = image.cols();

        Mat filteredImage = new Mat(height, width, CvType.CV_32F);
        image.convertTo(filteredImage, CvType.CV_32F);

        for (int y = padding; y < height - padding; y++) {
            for (int x = padding; x < width - padding; x++) {
                float newValue = 0;

                for (int k = 0; k < kernelSize; k++) {
                    for (int l = 0; l < kernelSize; l++) {
                        int pixelY = y - padding + k;
                        int pixelX = x - padding + l;
                        float pixelValue = (float) filteredImage.get(pixelY, pixelX)[0];
                        newValue += pixelValue * kernel[k][l];
                    }
                }

                filteredImage.put(y, x, newValue);
            }
        }

        Mat result = new Mat();
        filteredImage.convertTo(result, CvType.CV_8UC1);
        return result;
    }

    public static Mat applyGaussianBlur(Mat image, int kernelSize, double sigma, boolean useCv2) {
        System.out.println("Применение размытия Гаусса: kernel_size=" + kernelSize + ", sigma=" + sigma);

        if (useCv2) {
            Mat blurred = new Mat();
            Imgproc.GaussianBlur(image, blurred, new Size(kernelSize, kernelSize), sigma);
            System.out.println("Размытие Гаусса применено (OpenCV)");
            return blurred;
        } else {
            float[][] kernel = createGaussKernel(kernelSize, sigma);
            kernel = normalizeKernel(kernel);
            Mat blurred = applyGaussFilterManual(image, kernel);
            System.out.println("Размытие Гаусса применено (ручная реализация)");
            return blurred;
        }
    }

    public static GradientResult computeGradients(Mat image) {
        int height = image.rows();
        int width = image.cols();

        Mat Gx = new Mat(height, width, CvType.CV_32F, Scalar.all(0));
        Mat Gy = new Mat(height, width, CvType.CV_32F, Scalar.all(0));
        Mat magnitude = new Mat(height, width, CvType.CV_32F, Scalar.all(0));
        Mat angle = new Mat(height, width, CvType.CV_32F, Scalar.all(0));

        System.out.println("Вычисление градиентов для изображения " + height + "x" + width);

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float gxVal = 0;
                float gyVal = 0;

                // Свертка с ядрами Собеля
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        double pixelValue = image.get(y + i - 1, x + j - 1)[0];
                        gxVal += pixelValue * SOBEL_X[i][j];
                        gyVal += pixelValue * SOBEL_Y[i][j];
                    }
                }

                Gx.put(y, x, gxVal);
                Gy.put(y, x, gyVal);

                // Вычисление длины градиента
                double mag = Math.sqrt(gxVal * gxVal + gyVal * gyVal);
                magnitude.put(y, x, mag);

                // Вычисление угла градиента
                if (gxVal != 0) {
                    double angleRad = Math.atan2(gyVal, gxVal);
                    double angleDeg = Math.toDegrees(angleRad);

                    // Приведение угла к диапазону 0-180 градусов
                    if (angleDeg < 0) {
                        angleDeg += 180;
                    }

                    angle.put(y, x, angleDeg);
                } else {
                    angle.put(y, x, 90.0); // Вертикальное направление
                }
            }
        }

        System.out.println("Градиенты вычислены");
        return new GradientResult(Gx, Gy, magnitude, angle);
    }

    public static Mat nonMaximumSuppression(Mat magnitude, Mat angle) {
        int height = magnitude.rows();
        int width = magnitude.cols();

        Mat suppressed = new Mat(height, width, CvType.CV_32F, Scalar.all(0));

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                double currentAngle = angle.get(y, x)[0];
                double currentMagnitude = magnitude.get(y, x)[0];

                int direction;
                if ((0 <= currentAngle && currentAngle < 22.5) || (157.5 <= currentAngle && currentAngle <= 180)) {
                    direction = 0; // Горизонтальное
                } else if (22.5 <= currentAngle && currentAngle < 67.5) {
                    direction = 45; // Диагональ 45 градусов
                } else if (67.5 <= currentAngle && currentAngle < 112.5) {
                    direction = 90; // Вертикальное
                } else if (112.5 <= currentAngle && currentAngle < 157.5) {
                    direction = 135; // Диагональ 135 градусов
                } else {
                    direction = 0;
                }

                double neighbor1 = 0, neighbor2 = 0;

                switch (direction) {
                    case 0: // Горизонтальное
                        neighbor1 = magnitude.get(y, x + 1)[0]; // справа
                        neighbor2 = magnitude.get(y, x - 1)[0]; // слева
                        break;
                    case 45: // Диагональ 45 градусов
                        neighbor1 = magnitude.get(y + 1, x + 1)[0]; // слева сверху
                        neighbor2 = magnitude.get(y - 1, x - 1)[0]; // справа снизу
                        break;
                    case 90: // Вертикальное
                        neighbor1 = magnitude.get(y + 1, x)[0]; // снизу
                        neighbor2 = magnitude.get(y - 1, x)[0]; // сверху
                        break;
                    case 135: // Диагональ 135 градусов
                        neighbor1 = magnitude.get(y + 1, x - 1)[0]; // слева снизу
                        neighbor2 = magnitude.get(y - 1, x + 1)[0]; // // справа сверху
                        break;
                }

                if (currentMagnitude >= neighbor1 && currentMagnitude >= neighbor2) {
                    suppressed.put(y, x, currentMagnitude);
                } else {
                    suppressed.put(y, x, 0);
                }
            }
        }

        return suppressed;
    }

    public static Mat doubleThresholdFiltering(Mat magnitude, double lowRatio, double highRatio) {
        int height = magnitude.rows();
        int width = magnitude.cols();

        Mat result = new Mat(height, width, CvType.CV_8UC1, Scalar.all(0));

        Core.MinMaxLocResult minMax = Core.minMaxLoc(magnitude);
        double maxGrad = minMax.maxVal;

        double lowThreshold = maxGrad * lowRatio;
        double highThreshold = maxGrad * highRatio;

        System.out.printf("Максимальный градиент: %.2f\n", maxGrad);
        System.out.printf("Нижний порог (%.0f%%): %.2f\n", lowRatio * 100, lowThreshold);
        System.out.printf("Верхний порог (%.0f%%): %.2f\n", highRatio * 100, highThreshold);

        boolean[][] strongEdges = new boolean[height][width];
        boolean[][] weakEdges = new boolean[height][width];

        System.out.println("Применение двойной пороговой фильтрации...");

        // Первый проход: классификация пикселей
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double gradValue = magnitude.get(y, x)[0];

                if (gradValue >= highThreshold) {
                    strongEdges[y][x] = true;
                    result.put(y, x, 255);
                } else if (gradValue >= lowThreshold) {
                    weakEdges[y][x] = true;
                    result.put(y, x, 128);
                } else {
                    result.put(y, x, 0);
                }
            }
        }

        Mat finalResult = result.clone();

        // Второй проход: связывание слабых границ с сильными
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (weakEdges[y][x]) {
                    boolean hasStrongNeighbor = false;

                    // Проверка соседей 3x3
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            if (dy == 0 && dx == 0) continue;

                            int ny = y + dy;
                            int nx = x + dx;

                            if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                                if (strongEdges[ny][nx]) {
                                    hasStrongNeighbor = true;
                                    break;
                                }
                            }
                        }
                        if (hasStrongNeighbor) break;
                    }

                    if (hasStrongNeighbor) {
                        finalResult.put(y, x, 255);
                        strongEdges[y][x] = true;
                    } else {
                        finalResult.put(y, x, 0);
                    }
                }
            }
        }

        System.out.println("Двойная пороговая фильтрация завершена");
        return finalResult;
    }

    private static void showImage(String title, Mat image, int delay) {
        HighGui.imshow(title, image);
        HighGui.waitKey(delay);
    }

    private static void printMatRegion(Mat mat, int rows, int cols) {
        rows = Math.min(rows, mat.rows());
        cols = Math.min(cols, mat.cols());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                System.out.printf("%6.0f ", mat.get(i, j)[0]);
            }
            System.out.println();
        }
    }

    // Вспомогательный класс для хранения результатов вычисления градиентов
    private static class GradientResult {
        Mat Gx;
        Mat Gy;
        Mat magnitude;
        Mat angle;

        GradientResult(Mat Gx, Mat Gy, Mat magnitude, Mat angle) {
            this.Gx = Gx;
            this.Gy = Gy;
            this.magnitude = magnitude;
            this.angle = angle;
        }
    }
}