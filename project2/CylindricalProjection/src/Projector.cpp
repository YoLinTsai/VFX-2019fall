#include "Projector.h"
#include <string>

void Projector::RUN() {
    if (this->read()) this->project();
}

bool Projector::read() {
    std::cerr << std::endl << "[ Reading image \"" << _image_name<< "\" ]" << std::endl;
    _image = cv::imread(_image_name, 1);
    if (!_image.data) {
        std::cerr << "\t> Failed! Could not open or find the image!" << std::endl;
        return false;
    }
    else {
        std::cerr << "\t> Success! Image scale " << _image.rows << " x " << _image.cols << std::endl;
        return true;
    }
}

void Projector::project() {
    std::cerr << "\t> Projecting to cylindrical coordinate with focal length " << _focal << " (pixels)" << std::endl;
    _result.create(_image.rows, _image.cols, CV_8UC3);

/**********************************************************************
*                           find mid point                           *
**********************************************************************/
    _row_ref = _image.rows / 2;
    _col_ref = _image.cols / 2;

/**********************************************************************
*                          backward warpping                          *
**********************************************************************/
    for (int row = 0; row < (int)_result.rows; ++row) {
        for (int col = 0; col < (int)_result.cols; ++col) {
            uchar* pptr = _result.ptr(row, col);
            this->coordinate_shift(row, col);
            int row_ref, col_ref;
            this->cylindrical2plane(col_ref, row_ref, col, row);
            if (this->out_of_range(row_ref, col_ref, _result.rows, _result.cols)) {
                pptr[0] = 0;
                pptr[1] = 0;
                pptr[2] = 0;
            }
            else {
                this->shift_back(row_ref, col_ref);
                const uchar* ref_image_ptr = _image.ptr(row_ref, col_ref);
                pptr[0] = ref_image_ptr[0];
                pptr[1] = ref_image_ptr[1];
                pptr[2] = ref_image_ptr[2];
            }
            this->shift_back(row, col);
        }
    }
}

void Projector::show() const {
    cv::namedWindow("Cylindrical Projection", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cylindrical Projection", _result);
    std::cout << "\t> press any key to exit" << std::endl;
    cv::waitKey(0);
}

void Projector::save() const {
    std::string suffix = ".jpg";
    std::cerr << "[ Saving image \"" << _output_name+suffix << "\" ]" << std::endl;
    cv::imwrite(std::string(_output_name)+suffix, _result);
}
