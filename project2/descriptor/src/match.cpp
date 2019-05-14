#include "match.h"
//#include"lapacke.h"
#include <iostream>
#include <fstream>
#include <numeric>


float inner_product(const float* a, const float* b, int len)
{
	float accum = 0.0;
	for(int i=0;i<len;i++)
	{
		accum += a[i]*b[i];
	}
	return accum;
}

template <typename T>
std::vector<int> sort_indexes(const std::vector<T> &v) {

  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;
}

void Matcher::draw_match_pair(cv::Mat mat1, cv::Mat mat2, int** key1, int** key2)
{
	cv::Mat dst;
	cv::hconcat(mat1, mat2, dst);
    int width = mat1.cols;
	int sz = match_map.size();
	for(int i=0;i<sz;i++)
	{
		int index_1 = match_map[i][0];
		int index_2 = match_map[i][1];
		int x_1 = key1[index_1][0];
		int y_1 = key1[index_1][1];
		int x_2 = key2[index_2][0];
		int y_2 = key2[index_2][1];
		cv::line(dst, cv::Point(x_1, y_1), cv::Point(x_2+width, y_2), cv::Scalar(0, 255, 0), 1);
	}
	for(int i=0;i<512;i++)
	{
		cv::circle(dst, cv::Point(key1[i][0],key1[i][1]), 2, cv::Scalar(0,0,255), -1);
	}
	for(int i=0;i<512;i++)
	{
		cv::circle(dst, cv::Point(key2[i][0]+width,key2[i][1]), 2, cv::Scalar(0,0,255), -1);
	}
	cv::namedWindow("match_result", cv::WINDOW_AUTOSIZE); 
    cv::imshow("match_result", dst);
    cv::waitKey(0);
    cv::imwrite("match_test.jpg",dst);
    exit(0);
	return;
}

float euclidean_dis(float* feature_1, float* feature_2, int dim)
{
	float accum = 0.0;
	for(int i=0;i<dim;i++)
	{
		accum += (feature_1[i] - feature_2[i]) * (feature_1[i] - feature_2[i]);
	}
	return sqrt(accum);
}

void Matcher::feature_match(std::vector<float*> all_feature_1, std::vector<float*> all_feature_2, int** key1, int** key2)
{
	std::vector<float> distance;
	int sz1 = all_feature_1.size();
	int sz2 = all_feature_2.size();

	float accum = 0.0;
	for(int i=0;i<sz1;i++)
	{
		for(int j=0;j<sz2;j++)
		{
			distance.push_back(euclidean_dis(all_feature_1[i], all_feature_2[j], 121));
		}
		std::vector<int> sorted_index = sort_indexes(distance);
		accum+=(distance[sorted_index[1]]);
		distance.clear();
	}
	float second_mean = accum/sz1;
	for(int i=0;i<sz1;i++)
	{
		for(int j=0;j<sz2;j++)
		{
			distance.push_back(euclidean_dis(all_feature_1[i], all_feature_2[j], 121));
		}
		std::vector<int> sorted_index = sort_indexes(distance);
		if(distance[sorted_index[0]]/second_mean<0.5)
		{
			std::vector<int> tmp;
			tmp.push_back(i);
			tmp.push_back(sorted_index[0]);
			match_map.push_back(tmp);
		}
		distance.clear();
	}
	/*
	double* kp1 = new double[8];
	double* kp2 = new double[8];
	double* x = new double[8];
	int _index[4] = {61, 34, 39, 45};
	for(int i=0;i<4;i++)
	{
		kp1[i*2] = double(key1[match_map[_index[i]][0]][0]);
		kp1[i*2+1] = double(key1[match_map[_index[i]][0]][1]);
		kp2[i*2] = double(key2[match_map[_index[i]][1]][0]);
		kp2[i*2+1] = double(key2[match_map[_index[i]][1]][1]);
	}
	for(int i=0;i<4;i++)
	{
		printf("%d %d\n",int(kp1[2*i]),int(kp1[2*i+1]));
	}
	for(int i=0;i<4;i++)
	{
		printf("%d %d\n",int(kp2[2*i]),int(kp2[2*i+1]));
	}
	homograph_matrix(8, kp1, kp2, x);
	*/
	return;
}

void Matcher::homograph_matrix(int dim, double* kp1, double* kp2, double* x)
{
	/*
	char    TRANS = 'N';
	int     N = 8;
	int     NRHS = 1;
    int     LDA = N;
    int     LDB = N;
    int     INFO;
    int     IPIV[3];

    double A[64] = {0.0};
    double B[8] = {0.0};

    for(int i=0;i<64;i++) A[i]=0;
    for(int i=0;i<8;i++)
    {
    	if(i%2==0)
    	{
    		A[8*i+0] = kp1[int(i/2)*2];
    		A[8*i+1] = kp1[int(i/2)*2+1];
    		A[8*i+2] = 1;
    		A[8*i+6] = -kp2[int(i/2)*2] * kp1[int(i/2)*2];
    		A[8*i+7] = -kp2[int(i/2)*2] * kp1[int(i/2)*2+1];
    	}
    	else
    	{
    		A[8*i+3] = kp1[int(i/2)*2];
    		A[8*i+4] = kp1[int(i/2)*2+1];
    		A[8*i+5] = 1;
    		A[8*i+6] = -kp2[int(i/2)*2+1] * kp1[int(i/2)*2];
    		A[8*i+7] = -kp2[int(i/2)*2+1] * kp1[int(i/2)*2+1];	
    	}
    }

    for(int i=0;i<8;i++)
    {
    	B[i] = kp2[i];
    }
    for(int i=0;i<8;i++)
    {
    	printf("%f\n",B[i]);
    }
    for(int i=0;i<8;i++)
    {
    	for(int j=0;j<8;j++)
    	{
    		printf("%1.3f ",A[i*8+j]);
    	}
    	printf("\n");
    }
    dgetrf_(&N,&N,A,&LDA,IPIV,&INFO);
    dgetrs_(&TRANS,&N,&NRHS,A,&LDA,IPIV,B,&LDB,&INFO);
    double check = 0;
    for(int i=0;i<8;i++)
    {
    	printf("%f\n",A[i]);
    	check += A[i];
    }
    printf("check:%1.4f\n",check);
    */

    double A[64] = {0.0};
    double B[8] = {0.0};

    for(int i=0;i<64;i++) A[i]=0;
    for(int i=0;i<8;i++)
    {
    	if(i%2==0)
    	{
    		A[8*i+0] = kp1[int(i/2)*2];
    		A[8*i+1] = kp1[int(i/2)*2+1];
    		A[8*i+2] = 1;
    		A[8*i+6] = -kp2[int(i/2)*2] * kp1[int(i/2)*2];
    		A[8*i+7] = -kp2[int(i/2)*2] * kp1[int(i/2)*2+1];
    	}
    	else
    	{
    		A[8*i+3] = kp1[int(i/2)*2];
    		A[8*i+4] = kp1[int(i/2)*2+1];
    		A[8*i+5] = 1;
    		A[8*i+6] = -kp2[int(i/2)*2+1] * kp1[int(i/2)*2];
    		A[8*i+7] = -kp2[int(i/2)*2+1] * kp1[int(i/2)*2+1];	
    	}
    }
    for(int i=0;i<8;i++)
    {
    	B[i] = kp2[i];
    }
    cv::Mat _A(8,8,CV_64FC1);
    cv::Mat inv_A(8,8,CV_64FC1);
    for(int i=0;i<8;i++)
    {
    	for(int j=0;j<8;j++)
    	{
    		_A.at<double>(j,i) = A[8*i+j];
    	}
    }

    cv::invert(_A, inv_A,cv::DECOMP_SVD);

    for(int i=0;i<8;i++)
    {
    	float accum = 0;
    	for(int j=0;j<8;j++)
    	{
    		accum+=inv_A.at<double>(j,i)*B[j];
    	}
    	x[i] = accum;
    }
	return;
}
