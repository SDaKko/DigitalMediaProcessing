import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.highgui.HighGui;

public class GaussianFilter {

    public static double gaussianFunction(double x, double y, double sigma) {
        return (1.0 / (2.0 * Math.PI * sigma * sigma)) * Math.exp(-(x * x + y * y) / (2.0 * sigma * sigma));
    }

    public static double[][] createGaussianMatrix(double sigma, int kernelSize) {
        double[][] matrix = new double[kernelSize][kernelSize];
        int center = (int) Math.ceil(kernelSize / 2.0) - 1;

        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                double x = j - center;
                double y = i - center;
                matrix[i][j] = gaussianFunction(x, y, sigma);
            }
        }
        return matrix;
    }

    public static double sumMatrix(double[][] matrix) {
        double sum = 0.0;
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                sum += matrix[i][j];
            }
        }
        return sum;
    }

    public static double[][] normalizeKernel(double[][] kernel) {
        double sum = sumMatrix(kernel);
        double[][] normalized = new double[kernel.length][kernel[0].length];

        for (int i = 0; i < kernel.length; i++) {
            for (int j = 0; j < kernel[i].length; j++) {
                normalized[i][j] = kernel[i][j] / sum;
            }
        }
        return normalized;
    }

    public static void printMatrix(double[][] matrix, String title) {
        System.out.println(title);
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                System.out.printf("%.6f  ", matrix[i][j]);
            }
            System.out.println();
        }
        System.out.printf("Сумма элементов: %.6f\n", sumMatrix(matrix));
    }

    public static Mat gaussFilter(Mat image, double[][] kernel) {
        int height = image.rows();
        int width = image.cols();
        int kernelSize = kernel.length;
        int pad = kernelSize / 2;

        Mat result = new Mat(height, width, CvType.CV_64F);
        image.convertTo(result, CvType.CV_64F);


        for (int i = pad; i < height - pad; i++) {
            for (int j = pad; j < width - pad; j++) {
                double sum = 0.0;

                for (int ki = -pad; ki <= pad; ki++) {
                    for (int kj = -pad; kj <= pad; kj++) {
                        double pixelValue = image.get(i + ki, j + kj)[0];
                        double kernelValue = kernel[ki + pad][kj + pad];
                        sum += pixelValue * kernelValue;
                    }
                }

                result.put(i, j, sum);
            }
        }

        Mat displayResult = new Mat();
        result.convertTo(displayResult, CvType.CV_8U);

        return displayResult;
    }

    public static void main(String[] args) {
        System.load("C:\\opencv\\opencv\\build\\java\\x64\\opencv_java490.dll");

        double sigma = 3.0;
        int kernelSize = 5;
        int center = (int) Math.ceil(kernelSize / 2.0) - 1;
        System.out.println("Center: " + center);

        double[][] gaussianMatrix = createGaussianMatrix(sigma, kernelSize);
        printMatrix(gaussianMatrix, "Матрица Гаусса до нормировки:");

        double[][] normalizedMatrix = normalizeKernel(gaussianMatrix);
        printMatrix(normalizedMatrix, "\nМатрица Гаусса после нормировки:");

        String imagePath = "C:\\Pics\\waterfall.jpg";
        Mat img = Imgcodecs.imread(imagePath, Imgcodecs.IMREAD_GRAYSCALE);

        if (img.empty()) {
            System.out.println("Не удалось загрузить изображение: " + imagePath);
            return;
        }

        System.out.println("Информация об изображении:");
        System.out.println("Размер: " + img.rows() + "x" + img.cols());
        System.out.println("Тип: " + CvType.typeToString(img.type()));

        HighGui.imshow("original_img", img);
        HighGui.waitKey(1000);

        Mat imgAfterHandBlur = gaussFilter(img, normalizedMatrix);

        HighGui.imshow("img_after_hand_blur", imgAfterHandBlur);
        HighGui.waitKey(1000);

        HighGui.destroyAllWindows();
    }
}