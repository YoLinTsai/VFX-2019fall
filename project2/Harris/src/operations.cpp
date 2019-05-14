#include "Harris.h"
#include <iomanip>

void HarrisFeatureDetector::convolution(const cv::Mat& src, cv::Mat& dest, double**& kernel, int& kernel_size) {
    int half_width = kernel_size >> 1;
    int bl_x = half_width;
    int bl_y = half_width;
    int ur_x = src.rows-1-half_width;
    int ur_y = src.cols-1-half_width;
    for (int row = 0; row < src.rows; ++row) {
        for (int col = 0; col < src.cols; ++col) {
            if (row < bl_x || col < bl_y || row > ur_x || col > ur_y) {
                *(dest.ptr(row, col)) = 0;
                continue;
            }
            this->convolve(src, dest, row, col, kernel, half_width);
        }
    }
}

void HarrisFeatureDetector::convolve(const cv::Mat& src, cv::Mat& dest, int row, int col, double**& kernel, int& half_width) {
    double value = 0;
    for (int x = row-half_width; x <= row+half_width; ++x) {
        for (int y = col-half_width; y <= col+half_width; ++y) {
            const uchar* pos_ptr = src.ptr(x, y);
            value += (*pos_ptr) * (kernel[x-row+half_width][y-col+half_width]);
        }
    }
    *(dest.ptr(row, col)) = value;
}

void HarrisFeatureDetector::convolution(const std::vector<std::vector<double> >& src, std::vector<std::vector<double> >& dest, double**& kernel, int& kernel_size) {
    int half_width = kernel_size >> 1;
    int bl_x = half_width;
    int bl_y = half_width;
    int ur_x = src.size()-1-half_width;
    int ur_y = src.at(0).size()-1-half_width;
    dest.resize(src.size());
    for (int row = 0; row < (int)src.size(); ++row) {
        dest[row].resize(src.at(0).size());
        for (int col = 0; col < (int)src.at(0).size(); ++col) {
            if (row < bl_x || col < bl_y || row > ur_x || col > ur_y) {
                dest[row][col] = 0;
                continue;
            }
            this->convolve(src, dest, row, col, kernel, half_width);
        }
    }
}

void HarrisFeatureDetector::convolve(const std::vector<std::vector<double> >& src, std::vector<std::vector<double> >& dest, int row, int col, double**& kernel, int& half_width) {
    double value = 0;
    for (int x = row-half_width; x <= row+half_width; ++x) {
        for (int y = col-half_width; y <= col+half_width; ++y) {
            value += (src[x][y]) * (kernel[x-row+half_width][y-col+half_width]);
        }
    }
    dest[row][col] = value;
}

void HarrisFeatureDetector::multiply(cv::Mat& dest, const cv::Mat& img1, const cv::Mat& img2) {
    for (int row = 0; row < img1.rows; ++row) {
        for (int col = 0; col < img1.cols; ++col) {
            *(dest.ptr(row, col)) = (*(img1.ptr(row, col))) * (*(img2.ptr(row, col)));
        }
    }
}

void HarrisFeatureDetector::multiply(std::vector<std::vector<double> >& dest, const cv::Mat& img1, const cv::Mat& img2) {
    dest.resize(img1.rows);
    for (int row = 0; row < img1.rows; ++row) {
        dest[row].resize(img1.cols);
        for (int col = 0; col < img1.cols; ++col) {
            dest[row][col] = (*(img1.ptr(row, col))) * (*(img2.ptr(row, col)));
        }
    }
}

bool HarrisFeatureDetector::isMax(const std::vector<std::vector<double> >& src, const int& row, const int& col, const int& width) {
    for (int x = row-width; x <= row+width; ++x) {
        for (int y = col-width; y <= col+width; ++y) {
            if (x == row && y == col) continue;
            if (src[row][col] <= src[x][y]) return false;
        }
    }
    return true;
}
