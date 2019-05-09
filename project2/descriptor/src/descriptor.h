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
	void draw_arrow();
	void run();
	//calculate m(x,y)
	void generate_m_map(const cv::Mat);
	void retrieve_4X4_feature(const int,const int, float*, const int);
	void fill_bin(float*, const float, const float);
	void l2_norm_clip(float*, const float);
	float** keypoint_descriptor; //keypoint[index][32]

private:
	cv::Mat _image_rgb;
	cv::Mat _image_gray; 

	int _height;
	int _width;

	float** m_map;
	float** theta_map;

	int** keypoint; //keypoint[index][2]
	int keypoint_num;
	const static int degree_map[];
	float* *all_feature;

};

#endif