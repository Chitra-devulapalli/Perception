#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

void sobelEdgeDetector(const cv::Mat& src, cv::Mat& dst, int threshold) {
    int width = src.cols;
    int height = src.rows;
    dst = cv::Mat::zeros(height, width, CV_8UC1);

    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gradientX = 0;
            int gradientY = 0;
            // Performing a 3*3 convolution operation
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixel = src.at<uchar>(y + ky, x + kx);
                    gradientX += pixel * Gx[ky + 1][kx + 1];
                    gradientY += pixel * Gy[ky + 1][kx + 1];
                }
            }
            // Calculating the magnitude of gradient at every pixel 
            int gradient = std::sqrt(gradientX * gradientX + gradientY * gradientY);
            if (gradient > threshold) {
                dst.at<uchar>(y, x) = 255;  // Mark as edge
            } else {
                dst.at<uchar>(y, x) = 0;    // Not an edge
            }
        }
    }
}

// Function to create Laplacian of Gaussian (LoG) kernel
void createLoGKernel(std::vector<std::vector<double>>& kernel, int kSize, double sigma) {
    int half_kSize = kSize / 2;
    double sigma2 = sigma * sigma;
    double sigma4 = sigma2 * sigma2;

    for (int y = -half_kSize; y <= half_kSize; ++y) {
        for (int x = -half_kSize; x <= half_kSize; ++x) {
            double r2 = x * x + y * y;
            kernel[y + half_kSize][x + half_kSize] = (r2 - 2 * sigma2) / (2 * M_PI * sigma4) * std::exp(-r2 / (2 * sigma2));
        }
    }
}

// Function to apply LoG kernel and detect edges using zero-crossing
void laplacianOfGaussianEdgeDetector(const cv::Mat& src, cv::Mat& dst, double sigma, double threshold) {
    int width = src.cols;
    int height = src.rows;
    int kSize = std::round(sigma * 6) | 1; // Ensure odd size
    int half_kSize = kSize / 2;
    std::vector<std::vector<double>> kernel(kSize, std::vector<double>(kSize, 0));

    createLoGKernel(kernel, kSize, sigma);

    // Apply LoG kernel
    cv::Mat logImg = cv::Mat::zeros(height, width, CV_64F);
    for (int y = half_kSize; y < height - half_kSize; ++y) {
        for (int x = half_kSize; x < width - half_kSize; ++x) {
            double sum = 0.0;
            for (int ky = -half_kSize; ky <= half_kSize; ++ky) {
                for (int kx = -half_kSize; kx <= half_kSize; ++kx) {
                    sum += src.at<uchar>(y + ky, x + kx) * kernel[ky + half_kSize][kx + half_kSize];
                }
            }
            logImg.at<double>(y, x) = sum;
        }
    }

    // Zero-crossing detection
    dst = cv::Mat::zeros(height, width, CV_8UC1);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            bool zeroCrossing = false;
            double centerPixel = logImg.at<double>(y, x);

            // Check 4-neighbors
            if ((centerPixel > 0 && logImg.at<double>(y + 1, x) < 0) || (centerPixel < 0 && logImg.at<double>(y + 1, x) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y - 1, x) < 0) || (centerPixel < 0 && logImg.at<double>(y - 1, x) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y, x + 1) < 0) || (centerPixel < 0 && logImg.at<double>(y, x + 1) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y, x - 1) < 0) || (centerPixel < 0 && logImg.at<double>(y, x - 1) > 0)) zeroCrossing = true;

            // Optional: Check diagonal neighbors for more robustness
            if ((centerPixel > 0 && logImg.at<double>(y + 1, x + 1) < 0) || (centerPixel < 0 && logImg.at<double>(y + 1, x + 1) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y - 1, x - 1) < 0) || (centerPixel < 0 && logImg.at<double>(y - 1, x - 1) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y + 1, x - 1) < 0) || (centerPixel < 0 && logImg.at<double>(y + 1, x - 1) > 0)) zeroCrossing = true;
            if ((centerPixel > 0 && logImg.at<double>(y - 1, x + 1) < 0) || (centerPixel < 0 && logImg.at<double>(y - 1, x + 1) > 0)) zeroCrossing = true;

            // Threshold the zero-crossings to suppress weak edges
            if (zeroCrossing && std::abs(centerPixel) > threshold) {
                dst.at<uchar>(y, x) = 255;
            }
        }
    }
}


// Canny edge detector
void cannyEdgeDetector(const cv::Mat& src, cv::Mat& dst, double lowThreshold, double highThreshold) {
    int width = src.cols;
    int height = src.rows;
    cv::Mat gradient, orientation;
    sobelEdgeDetector(src, gradient);
    orientation = cv::Mat::zeros(height, width, CV_64F);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = src.at<uchar>(y, x+1) - src.at<uchar>(y, x-1);
            int gy = src.at<uchar>(y+1, x) - src.at<uchar>(y-1, x);
            orientation.at<double>(y, x) = std::atan2(gy, gx);
        }
    }

    // Non-maximum suppression
    cv::Mat nms = cv::Mat::zeros(height, width, CV_8UC1);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            double angle = orientation.at<double>(y, x) * 180 / M_PI;
            if (angle < 0) angle += 180;

            int neighbor1 = 0, neighbor2 = 0;
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                neighbor1 = gradient.at<uchar>(y, x-1);
                neighbor2 = gradient.at<uchar>(y, x+1);
            } else if (22.5 <= angle && angle < 67.5) {
                neighbor1 = gradient.at<uchar>(y-1, x+1);
                neighbor2 = gradient.at<uchar>(y+1, x-1);
            } else if (67.5 <= angle && angle < 112.5) {
                neighbor1 = gradient.at<uchar>(y-1, x);
                neighbor2 = gradient.at<uchar>(y+1, x);
            } else if (112.5 <= angle && angle < 157.5) {
                neighbor1 = gradient.at<uchar>(y-1, x-1);
                neighbor2 = gradient.at<uchar>(y+1, x+1);
            }

            if (gradient.at<uchar>(y, x) >= neighbor1 && gradient.at<uchar>(y, x) >= neighbor2) {
                nms.at<uchar>(y, x) = gradient.at<uchar>(y, x);
            } else {
                nms.at<uchar>(y, x) = 0;
            }
        }
    }

    // Double threshold and edge tracking by hysteresis
    dst = cv::Mat::zeros(height, width, CV_8UC1);
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            if (nms.at<uchar>(y, x) >= highThreshold) {
                dst.at<uchar>(y, x) = 255;
            } else if (nms.at<uchar>(y, x) >= lowThreshold) {
                bool strongEdge = false;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        if (nms.at<uchar>(y + ky, x + kx) >= highThreshold) {
                            strongEdge = true;
                            break;
                        }
                    }
                    if (strongEdge) break;
                }
                if (strongEdge) dst.at<uchar>(y, x) = 255;
            }
        }
    }
}


// Main function to test the edge detectors
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    cv::Mat src = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Error: Unable to load image " << argv[1] << std::endl;
        return -1;
    }

    cv::Mat sobel, derivative, log, canny;
    sobelEdgeDetector(src, sobel);
    derivativeOfGradient(src, derivative);
    laplacianOfGaussian(src, log);
    cannyEdgeDetector(src, canny, 50, 100);

    cv::imshow("Original", src);
    cv::imshow("Sobel", sobel);
    cv::imshow("Laplacian of Gaussian", log);
    cv::imshow("Canny", canny);
    cv::waitKey(0);

    return 0;
}