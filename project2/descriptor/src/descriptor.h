#ifndef _DESCRIPTOR_H_
#define _DESCRIPTOR_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;
#define PI 3.14159265

class SIFTfeatureDescriptor
{
public:
	SIFTfeatureDescriptor() {}
	~SIFTfeatureDescriptor() {}

	void read_image(char*);
	void read_keypoint(char*, const int);
	void generate_keypoint_descriptor();
	bool check_position(int*);
	void draw_arrow_theta();
	void draw_arrow_orientation();
	void run();
	//calculate m(x,y)
	void generate_m_map(const cv::Mat);
	void retrieve_4X4_feature(const int,const int, float*, const int, int);
	void retrieve_MSOP_feature(int ,int, float, float*);
	void fill_bin(float*, const float, const float, int);
	void l2_norm_clip(float*, const float, int);
	int get_weighted_orientation(int, int);
	void FilterCreation(float**, int, float);
	void gaussian_test();
	std::vector<float*> all_feature;
	std::vector<int> all_orientation;
	cv::Mat _image_rgb;
	int** keypoint; //keypoint[index][2]

private:

	cv::Mat _image_gray; 
	cv::Mat _display;
	//cv::Mat _gaussian_gray;

	int _height;
	int _width;

	float** Gaussian;
	float** Gaussian_9x9;
	float** Gaussian_17x17;
	float** m_map;
	float** theta_map;

	int keypoint_num;
	const static int degree_map[];
	const static int degree_bucket[];
};

#endif