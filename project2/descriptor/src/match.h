#ifndef _MATCH_H_
#define _MATCH_H_

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <math.h>
#include <vector>

class Matcher
{
public:
	Matcher() {}
	~Matcher() {}

	void feature_match(std::vector<float*>, std::vector<float*>, int**, int**);
	void draw_match_pair(cv::Mat, cv::Mat, int**, int**);
	void homograph_matrix(int, double*, double*, double*);
	std::vector<std::vector<int>> match_map;
private:

};

#endif