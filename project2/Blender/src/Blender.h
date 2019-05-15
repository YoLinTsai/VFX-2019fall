#ifndef __BLENDER_H__
#define __BLENDER_H__

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

using std::string;

struct BBox
{
    BBox() {}
    ~BBox() {}

    void SetLowerLeftX(int i) { _lx = i; }
    void SetLowerLeftY(int i) { _ly = i; }
    void SetUpperRightX(int i) { _ux = i; }
    void SetUpperRightY(int i) { _uy = i; }

    int GetLowerLeftX() const { return _lx; }
    int GetLowerLeftY() const { return _ly; }
    int GetUpperRightX() const { return _ux; }
    int GetUpperRightY() const { return _uy; }

    int GetWidth() const { return _ux - _lx + 1; }
    int GetHeight() const { return _uy - _ly + 1; }

    bool contains(const int& x, const int& y) const { return ( _lx <= x && x <= _ux && _ly <= y && y <= _uy); }

    void print() const {
        std::cout << "(" << _lx << ", " << _ly << ") (" << _ux << ", " << _uy << ")" << std::endl;
    }

    int _lx;
    int _ly;
    int _ux;
    int _uy;
};

class Blender
{
public:
    Blender(string i1, string i2, string b, string n) { this->read(i1, i2, b, n); }
    ~Blender() {}

    void read(string, string, string, string);
    void blend();
    void show() const;
    void save(const std::string) const;

    void restricted_blend();
    void full_blend();

private:
    cv::Mat _image1;
    cv::Mat _image2;
    cv::Mat _origin_img1;
    cv::Mat _origin_img2;
    cv::Mat _result;
    BBox    _bbox1;
    BBox    _bbox2;

    std::string _next_box_file;
    BBox    _nextbox1;
    BBox    _nextbox2;

    int _1to2x;
    int _1to2y;
};

#endif /* __BLENDER_H__ */
