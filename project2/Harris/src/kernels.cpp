#include "Harris.h"
#include <iomanip>

void HarrisFeatureDetector::init_kernels(std::string type) {
    _kernel_x = new double*[5];
    for (int i = 0; i < 5; ++i) _kernel_x[i] = new double[5];
    _kernel_y = new double*[5];
    for (int i = 0; i < 5; ++i) _kernel_y[i] = new double[5];

    if (type == "sobel") {
        _kernel_x[0][0] = -5.0/240;
        _kernel_x[0][1] = -4.0/240;
        _kernel_x[0][2] = 0.0/240;
        _kernel_x[0][3] = 4.0/240;
        _kernel_x[0][4] = 5.0/240;
        _kernel_x[1][0] = -8.0/240;
        _kernel_x[1][1] = -10.0/240;
        _kernel_x[1][2] = 0.0/240;
        _kernel_x[1][3] = 10.0/240;
        _kernel_x[1][4] = 8.0/240;
        _kernel_x[2][0] = -10.0/240;
        _kernel_x[2][1] = -20.0/240;
        _kernel_x[2][2] = 0.0/240;
        _kernel_x[2][3] = 20.0/240;
        _kernel_x[2][4] = 10.0/240;
        _kernel_x[3][0] = -8.0/240;
        _kernel_x[3][1] = -10.0/240;
        _kernel_x[3][2] = 0.0/240;
        _kernel_x[3][3] = 10.0/240;
        _kernel_x[3][4] = 8.0/240;
        _kernel_x[4][0] = -5.0/240;
        _kernel_x[4][1] = -4.0/240;
        _kernel_x[4][2] = 0.0/240;
        _kernel_x[4][3] = 4.0/240;
        _kernel_x[4][4] = 5.0/240;

        _kernel_y[0][0] = -5.0/240;
        _kernel_y[0][1] = -8.0/240;
        _kernel_y[0][2] = -10.0/240;
        _kernel_y[0][3] = -8.0/240;
        _kernel_y[0][4] = -5.0/240;
        _kernel_y[1][0] = -4.0/240;
        _kernel_y[1][1] = -10.0/240;
        _kernel_y[1][2] = -20.0/240;
        _kernel_y[1][3] = -10.0/240;
        _kernel_y[1][4] = -4.0/240;
        _kernel_y[2][0] = 0.0/240;
        _kernel_y[2][1] = 0.0/240;
        _kernel_y[2][2] = 0.0/240;
        _kernel_y[2][3] = 0.0/240;
        _kernel_y[2][4] = 0.0/240;
        _kernel_y[3][0] = 4.0/240;
        _kernel_y[3][1] = 10.0/240;
        _kernel_y[3][2] = 20.0/240;
        _kernel_y[3][3] = 10.0/240;
        _kernel_y[3][4] = 4.0/240;
        _kernel_y[4][0] = 5.0/240;
        _kernel_y[4][1] = 8.0/240;
        _kernel_y[4][2] = 10.0/240;
        _kernel_y[4][3] = 8.0/240;
        _kernel_y[4][4] = 5.0/240;

        _kernel_size = 5;
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
    for (int i = 0; i < 5; ++i) {
        std::cerr << "\t  ";
        for (int j = 0; j < 5; ++j) {
            std::cerr << std::setw(6) << _kernel_x[i][j] << ' ';
        } std::cerr << std::endl;
    }
    std::cerr << std::endl;
    std::cerr << "\t> y kernel -- " << type << " size " << _kernel_size << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cerr << "\t  ";
        for (int j = 0; j < 5; ++j) {
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

