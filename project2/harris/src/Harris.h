#ifndef __HARRIS_H__
#define __HARRIS_H__

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include "minHeap.h"

class HarrisFeatureDetector
{
public:
    HarrisFeatureDetector(double k, int f, char* m) { _k = k; _features = f; _mode = m; }
    ~HarrisFeatureDetector() {}

    // top API
    void read_image(char*);
    void run();
    void dump(std::ostream&);
    void show_result();

    // core
    void compute_derivatives();
    void compute_products();
    void compute_products_of_sums();
    void compute_response();
    void compute_local_max_R();

    void show_images();
    void init_kernels(std::string);
    int  create_guassian_kernel_3();
    int  create_guassian_kernel_5();

    // matrix operations
    void convolution(const cv::Mat&, cv::Mat&, double**&, int&);
    void convolution(const std::vector<std::vector<double> >&, std::vector<std::vector<double> >&, double**&, int&);
    void convolve(const cv::Mat&, cv::Mat&, int, int, double**&, int&);
    void convolve(const std::vector<std::vector<double> >&, std::vector<std::vector<double> >&, int, int, double**&, int&);
    void multiply(cv::Mat&, const cv::Mat&, const cv::Mat&);
    void multiply(std::vector<std::vector<double> >&, const cv::Mat&, const cv::Mat&);
    bool isMax(const std::vector<std::vector<double> >&, const int&, const int&, const int&);

private:
    cv::Mat      _image_rgb;
    cv::Mat      _image_gray;
    cv::Mat      _image_blur;

    cv::Mat      _image_dx;
    cv::Mat      _image_dy;

    int          _kernel_size;
    int          _features;
    double       _k;
    double**     _kernel_x;
    double**     _kernel_y;
    double**     _guassian_filter;

    std::string  _mode;

    std::vector<std::vector<double> >      _image_dxdy;
    std::vector<std::vector<double> >      _image_dxdx;
    std::vector<std::vector<double> >      _image_dydy;

    std::vector<std::vector<double> >      _image_Sxx;
    std::vector<std::vector<double> >      _image_Sxy;
    std::vector<std::vector<double> >      _image_Syy;

    std::vector<std::vector<double> >      _R;

    minHeap<PRObject>                      _result;
};

#endif /* __HARRIS_H__ */
