#include "Harris.h"
#include <iomanip>

void HarrisFeatureDetector::init_kernels(std::string type) {
    _kernel_x = new double*[3];
    for (int i = 0; i < 3; ++i) _kernel_x[i] = new double[3];
    _kernel_y = new double*[3];
    for (int i = 0; i < 3; ++i) _kernel_y[i] = new double[3];

    if (type == "sobel") {
        _kernel_x[0][0] = 1.0/8;
        _kernel_x[0][1] = 0.0/8;
        _kernel_x[0][2] = -1.0/8;
        _kernel_x[1][0] = 2.0/8;
        _kernel_x[1][1] = 0.0/8;
        _kernel_x[1][2] = -2.0/8;
        _kernel_x[2][0] = 1.0/8;
        _kernel_x[2][1] = 0.0/8;
        _kernel_x[2][2] = -1.0/8;

        _kernel_y[0][0] = 1.0/8;
        _kernel_y[0][1] = 2.0/8;
        _kernel_y[0][2] = 1.0/8;
        _kernel_y[1][0] = 0.0/8;
        _kernel_y[1][1] = 0.0/8;
        _kernel_y[1][2] = 0.0/8;
        _kernel_y[2][0] = -1.0/8;
        _kernel_y[2][1] = -2.0/8;
        _kernel_y[2][2] = -1.0/8;

        _kernel_size = 3;
    }
    if (type == "simple") {
        _kernel_x[0][0] = 1.0/3;
        _kernel_x[0][1] = 0.0/3;
        _kernel_x[0][2] = -1.0/3;
        _kernel_x[1][0] = 1.0/3;
        _kernel_x[1][1] = 0.0/3;
        _kernel_x[1][2] = -1.0/3;
        _kernel_x[2][0] = 1.0/3;
        _kernel_x[2][1] = 0.0/3;
        _kernel_x[2][2] = -1.0/3;

        _kernel_y[0][0] = 1.0/3;
        _kernel_y[0][1] = 1.0/3;
        _kernel_y[0][2] = 1.0/3;
        _kernel_y[1][0] = 0.0/3;
        _kernel_y[1][1] = 0.0/3;
        _kernel_y[1][2] = 0.0/3;
        _kernel_y[2][0] = -1.0/3;
        _kernel_y[2][1] = -1.0/3;
        _kernel_y[2][2] = -1.0/3;

        _kernel_size = 3;
    }
    std::cerr << "\t> x kernel -- " << type << " size " << _kernel_size << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cerr << "\t  ";
        for (int j = 0; j < 3; ++j) {
            std::cerr << std::setw(6) << _kernel_x[i][j] << ' ';
        } std::cerr << std::endl;
    }
    std::cerr << std::endl;
    std::cerr << "\t> y kernel -- " << type << " size " << _kernel_size << std::endl;
    for (int i = 0; i < 3; ++i) {
        std::cerr << "\t  ";
        for (int j = 0; j < 3; ++j) {
            std::cerr << std::setw(6) << _kernel_y[i][j] << ' ';
        } std::cerr << std::endl;
    }
}

int HarrisFeatureDetector::create_guassian_kernel_3() {
    _guassian_filter = new double*[3];
    for (int i = 0; i < 3; ++i) _guassian_filter[i] = new double[3];
    _guassian_filter[0][0] = 1.0/16;
    _guassian_filter[0][1] = 1.0/8;
    _guassian_filter[0][2] = 1.0/16;
    _guassian_filter[1][0] = 1.0/8;
    _guassian_filter[1][1] = 1.0/4;
    _guassian_filter[1][2] = 1.0/8;
    _guassian_filter[2][0] = 1.0/16;
    _guassian_filter[2][1] = 1.0/8;
    _guassian_filter[2][2] = 1.0/16;
    return 3;
}

int HarrisFeatureDetector::create_guassian_kernel_5() {
    _guassian_filter = new double*[5];
    for (int i = 0; i < 5; ++i) _guassian_filter[i] = new double[5];
    _guassian_filter[0][0] = 1.0/273;
    _guassian_filter[0][1] = 4.0/273;
    _guassian_filter[0][2] = 7.0/273;
    _guassian_filter[0][3] = 4.0/273;
    _guassian_filter[0][4] = 1.0/273;
    _guassian_filter[1][0] = 4.0/273;
    _guassian_filter[1][1] = 16.0/273;
    _guassian_filter[1][2] = 26.0/273;
    _guassian_filter[1][3] = 16.0/273;
    _guassian_filter[1][4] = 4.0/273;
    _guassian_filter[2][0] = 7.0/273;
    _guassian_filter[2][1] = 26.0/273;
    _guassian_filter[2][2] = 41.0/273;
    _guassian_filter[2][3] = 26.0/273;
    _guassian_filter[2][4] = 7.0/273;
    _guassian_filter[3][0] = 4.0/273;
    _guassian_filter[3][1] = 16.0/273;
    _guassian_filter[3][2] = 26.0/273;
    _guassian_filter[3][3] = 16.0/273;
    _guassian_filter[3][4] = 4.0/273;
    _guassian_filter[4][0] = 1.0/273;
    _guassian_filter[4][1] = 4.0/273;
    _guassian_filter[4][2] = 7.0/273;
    _guassian_filter[4][3] = 4.0/273;
    _guassian_filter[4][4] = 1.0/273;
    return 5;
}

