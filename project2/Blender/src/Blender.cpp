#include "Blender.h"
#include <fstream>
#include <iostream>

using std::cout;
using std::endl;

void Blender::read(string image1, string image2, string bbox_file) {
    _image1 = cv::imread(image1, 1);
    _image2 = cv::imread(image2, 1);
    std::ifstream file;
    file.open(bbox_file);
    std::string token;
    file >> token;
    _bbox1.SetLowerLeftX(std::stoi(token));
    file >> token;
    _bbox1.SetLowerLeftY(std::stoi(token));
    file >> token;
    _bbox1.SetUpperRightX(std::stoi(token));
    file >> token;
    _bbox1.SetUpperRightY(std::stoi(token));
    file >> token;
    _bbox2.SetLowerLeftX(std::stoi(token));
    file >> token;
    _bbox2.SetLowerLeftY(std::stoi(token));
    file >> token;
    _bbox2.SetUpperRightX(std::stoi(token));
    file >> token;
    _bbox2.SetUpperRightY(std::stoi(token));

    assert(_bbox1.GetWidth() == _bbox2.GetWidth());
    assert(_bbox1.GetHeight() == _bbox1.GetHeight());

    // let _image1 be the left image, _image2 be the right image
    if (_bbox1.GetLowerLeftX() < _bbox2.GetLowerLeftX()) {
        std::swap(_bbox1, _bbox2);
        std::swap(_image1, _image2);
    }
    _1to2x = _bbox2.GetLowerLeftX() - _bbox1.GetLowerLeftX();
    _1to2y = _bbox2.GetLowerLeftY() - _bbox1.GetLowerLeftY();
}

void Blender::blend() {
    cout << "\t> blending width:  " << _bbox1.GetWidth() << endl;
    cout << "\t> blending height: " << _bbox1.GetHeight() << endl;
    _result.create(_image1.rows*2-_bbox1.GetHeight(), _image1.cols*2-_bbox1.GetWidth(), CV_8UC3);

    // save the original images
    _origin_img1 = _image1.clone();
    _origin_img2 = _image2.clone();
    for (int row = 0; row < _image1.rows; ++row) {
        for (int col = 0; col < _image1.cols; ++col) {
            if (10 < row && row < _image1.rows-10 && 10 < col && col < _image1.cols-10) continue;
            uchar* p1 = _origin_img1.ptr(row, col);
            uchar* p2 = _origin_img2.ptr(row, col);
            if (p1[0] < 15 && p1[1] < 15 && p1[2] < 15) {
                p1[0] = 255;
                p1[1] = 0;
                p1[2] = 0;
            }
            if (p2[0] < 15 && p2[1] < 15 && p2[2] < 15) {
                p2[0] = 255;
                p2[1] = 0;
                p2[2] = 0;
            }
        }
    }

    // set the blend width constant
    int BLEND_WIDTH = _bbox1.GetWidth() * 0.1;
    cout << "\t> blending constant: " << BLEND_WIDTH << endl;

    // find middle line
    int middle1 = _bbox1.GetLowerLeftX() + _bbox1.GetWidth()/2;
    int middle2 = _bbox2.GetLowerLeftX() + _bbox2.GetWidth()/2;

    // weight the blending area first
    int min_col = middle1 - BLEND_WIDTH / 2;
    int max_col = middle1 + BLEND_WIDTH / 2;
    for (int row = 0; row < _image1.rows; ++row) {
        for (int col = 0; col < _image1.cols; ++col) {
            if (row < _bbox1.GetLowerLeftY() || row > _bbox1.GetUpperRightY()) continue;
            if (min_col <= col && col <= max_col) {
                const uchar* p2 = _origin_img2.ptr(row+_1to2y, col+_1to2x);
                uchar* pptr = _image1.ptr(row, col);
                if (p2[0] == 255 && !p2[1] && !p2[2]) {
                    /*
                    pptr[0] = 0;
                    pptr[1] = 255;
                    pptr[2] = 0;
                    */
                    continue;
                }
                float weight = (max_col - col) / (float)BLEND_WIDTH;
                pptr[0] *= weight;
                pptr[1] *= weight;
                pptr[2] *= weight;
            }
            else if (max_col < col && col <= _bbox1.GetUpperRightX()) {
                const uchar* p2 = _origin_img2.ptr(row+_1to2y, col+_1to2x);
                uchar* pptr = _image1.ptr(row, col);
                if (p2[0] == 255 && !p2[1] && !p2[2]) {
                    /*
                    pptr[0] = 0;
                    pptr[1] = 255;
                    pptr[2] = 0;
                    */
                }
                else {
                    pptr[0] = pptr[1] = pptr[2] = 0;
                }
            }
        }
    }

    min_col = middle2 - BLEND_WIDTH / 2;
    max_col = middle2 + BLEND_WIDTH / 2;
    for (int row = 0; row < _image2.rows; ++row) {
        for (int col = 0; col < _image2.cols; ++col) {
            if (row < _bbox2.GetLowerLeftY() || row > _bbox2.GetUpperRightY()) continue;
            if (min_col <= col && col <= max_col) {
                const uchar* p1 = _origin_img1.ptr(row-_1to2y, col-_1to2x);
                uchar* pptr = _image2.ptr(row, col);
                if (p1[0] == 255 && !p1[1] && !p1[2]) {
                    /*
                    pptr[0] = 0;
                    pptr[1] = 255;
                    pptr[2] = 0;
                    */
                    continue;
                }
                float weight = (col - min_col) / (float)BLEND_WIDTH;
                pptr[0] *= weight;
                pptr[1] *= weight;
                pptr[2] *= weight;
            }
            else if (_bbox2.GetLowerLeftX() <= col && col < min_col) {
                const uchar* p1 = _origin_img1.ptr(row-_1to2y, col-_1to2x);
                uchar* pptr = _image2.ptr(row, col);
                if (p1[0] == 255 && !p1[1] && !p1[2]) {
                    /*
                    pptr[0] = 0;
                    pptr[1] = 255;
                    pptr[2] = 0;
                    */
                }
                else {
                    pptr[0] = pptr[1] = pptr[2] = 0;
                }
            }
        }
    }

    // find the bounding box for the two images
    BBox bbox_image1;
    BBox bbox_image2;
    if (_bbox1.GetLowerLeftY() > _bbox2.GetLowerLeftY()) {
        bbox_image1.SetLowerLeftY(0);
        bbox_image1.SetLowerLeftX(0);
        bbox_image1.SetUpperRightX(_image1.cols-1);
        bbox_image1.SetUpperRightY(_image1.rows-1);

        bbox_image2.SetLowerLeftX(_bbox1.GetLowerLeftX());
        bbox_image2.SetLowerLeftY(_bbox1.GetLowerLeftY());
        bbox_image2.SetUpperRightX(_result.cols-1);
        bbox_image2.SetUpperRightY(_result.rows-1);
    }
    else {
        bbox_image1.SetLowerLeftY(_bbox2.GetLowerLeftY() - _bbox1.GetLowerLeftY());
        bbox_image1.SetLowerLeftX(0);
        bbox_image1.SetUpperRightX(_image1.cols-1);
        bbox_image1.SetUpperRightY(_result.rows-1);

        bbox_image2.SetLowerLeftX(_bbox1.GetLowerLeftX());
        bbox_image2.SetLowerLeftY(_bbox1.GetLowerLeftY());
        bbox_image2.SetUpperRightX(_result.cols-1);
        bbox_image2.SetUpperRightY(_image2.rows-1);
    }

    // blend
    for (int row = 0; row < (int)_result.rows; ++row) {
        for (int col = 0; col < (int)_result.cols; ++col) {
            uchar* pptr = _result.ptr(row, col);
            if (bbox_image1.contains(col, row) && !bbox_image2.contains(col, row)) {
                uchar* p1 = _image1.ptr(row-bbox_image1.GetLowerLeftY(), col-bbox_image1.GetLowerLeftX());
                pptr[0] = p1[0];
                pptr[1] = p1[1];
                pptr[2] = p1[2];
            }
            else if (!bbox_image1.contains(col, row) && bbox_image2.contains(col, row)) {
                uchar* p2 = _image2.ptr(row-bbox_image2.GetLowerLeftY(), col-bbox_image2.GetLowerLeftX());
                pptr[0] = p2[0];
                pptr[1] = p2[1];
                pptr[2] = p2[2];
            }
            else if (bbox_image1.contains(col, row) && bbox_image2.contains(col, row)) {
                uchar* p1 = _image1.ptr(row-bbox_image1.GetLowerLeftY(), col-bbox_image1.GetLowerLeftX());
                uchar* p2 = _image2.ptr(row-bbox_image2.GetLowerLeftY(), col-bbox_image2.GetLowerLeftX());
                pptr[0] = p2[0] + p1[0];
                pptr[1] = p2[1] + p1[1];
                pptr[2] = p2[2] + p1[2];
            }
            else {
                pptr[0] = pptr[1] = pptr[2] = 0;
            }
        }
    }
}

void Blender::show() const {
    cv::namedWindow("left image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right image", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("left origin", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("right origin", cv::WINDOW_AUTOSIZE);
    cv::namedWindow("blended image", cv::WINDOW_AUTOSIZE);
    cv::imshow("left image", _image1);
    cv::imshow("right image", _image2);
    cv::imshow("left origin", _origin_img1);
    cv::imshow("right origin", _origin_img2);
    cv::imshow("blended image", _result);
    cout << "\tpress any key to close all windows" << endl;
    cv::waitKey(0);
}

void Blender::save(const std::string filename) const {
    std::cerr << "[ Saving image \"" << filename << "\" ]" << std::endl;
    cv::imwrite(filename, _result);
}
