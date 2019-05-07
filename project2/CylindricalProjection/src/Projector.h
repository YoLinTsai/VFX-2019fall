#ifndef __PROJECTOR_H__
#define __PROJECTOR_H__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>

class Projector
{
public:
    Projector(char* image, char* output, int f) { _image_name = image; _output_name = output; _focal = f; }
    ~Projector() {}

    void RUN();

    bool read();
    void project();
    void show() const;
    void save() const;

/**********************************************************************
*                          inline functions                          *
**********************************************************************/
    inline bool out_of_range(const int& r, const int& c, const int& rows, const int& cols) {
        if (-_row_ref <= r && r < rows-_row_ref && -_col_ref <= c && c < cols-_col_ref) return false;
        return true;
    }

    inline void coordinate_shift(int& r, int& c) {
        r -= _row_ref;
        c -= _col_ref;
    }

    inline void shift_back(int& r, int& c) {
        r += _row_ref;
        c += _col_ref;
    }

    inline void cylindrical2plane(int& dest_x, int& dest_y, const int& x, const int& y) {
        dest_x = (double)_focal * tan( (double)x / (double)_focal );
        dest_y = sqrt( (double)(x*x) + (double)(_focal*_focal) ) * y / _focal;
    }

private:
    char*       _image_name;
    char*       _output_name;

    cv::Mat     _image;
    cv::Mat     _result;

    int         _focal;

    int       _row_ref;
    int       _col_ref;
};

#endif /* __PROJECTOR_H__ */
