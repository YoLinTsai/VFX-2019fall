#include "descriptor.h"
#include <iostream>
#include <fstream>

const int SIFTfeatureDescriptor::degree_map[] = {0, 45, 90, 135, 180, 225, 270, 315, 360}; 

void SIFTfeatureDescriptor::read_image(char* filename)
{
	std::cerr << std::endl << "[ Reading image \"" << filename << "\" ]" << std::endl;
    _image_rgb = cv::imread(filename, 1);
    if (!_image_rgb.data) {
        std::cerr << "\t> Failed! Could not open or find the image!" << std::endl;
    }
    else {
        std::cerr << "\t> Success! Image scale " << _image_rgb.rows << " x " << _image_rgb.cols << std::endl;
        std::cerr << "\t> Converting to gray scale..." << std::endl;
        cv::cvtColor(_image_rgb, _image_gray, cv::COLOR_RGB2GRAY);
    }
    return;
}

//fill keypoint[][]
void SIFTfeatureDescriptor::read_keypoint(char* filename, const int num)
{
    keypoint_num = num;
    keypoint = new int*[num];
    for(int i = 0; i < keypoint_num; i++)
    {
        keypoint[i] = new int[2]; 
    }
    std::cerr << std::endl << "[ Reading keypoint file \"" << filename << "\" ]" << std::endl;
    std::string line;
    std::ifstream fp(filename);
    int index_cnt = 0;
    while(getline(fp, line))
    {
        std::istringstream iss(line);
        if(index_cnt==0) { index_cnt +=1; continue; }
        int x, y;
        char separator;
        iss >>x>>separator>>y;
        keypoint[index_cnt-1][0] = x;
        keypoint[index_cnt-1][1] = y;
        index_cnt += 1;
    }
}

//calculate m_Mat theta_Mat
void SIFTfeatureDescriptor::generate_m_map(const cv::Mat L)
{
    int height = L.rows; 
    int width = L.cols;
    _height = height;
    _width = width;
    printf("%d, %d\n",height,width);

    m_map = new float*[height];
    theta_map = new float*[height];

    for(int i = 0; i < height; i++)
    {
        m_map[i] = new float[width];
        theta_map[i] = new float[width];    
    }

    for(int i=1;i<height-1;i++)
    {
        for(int j=1;j<width-1;j++)
        {
            float b = L.at<uchar>(i+1, j) - L.at<uchar>(i-1, j);
            float a = L.at<uchar>(i, j+1) - L.at<uchar>(i, j-1);

            m_map[i][j] = sqrt( pow(a, 2) + pow(b, 2));

            if(a!=0){
                theta_map[i][j] = atan(b/a);
            }
            else{
                if(b>=0){   theta_map[i][j] = PI/2;}
                else{       theta_map[i][j] = PI*3/2;}
            }
            /*printf("i:%d ,j:%d ,value:%1.6f\n",i,j,theta_map[i][j]*180/PI);*/
        }
    }
	return;
}

void SIFTfeatureDescriptor::draw_arrow()
{
    cv::Mat dst;
    //dst = _image_gray.clone();
    cv::cvtColor(_image_gray, dst, cv::COLOR_GRAY2BGR);
    for(int i=0;i<keypoint_num;i++)
    {
        int x_1 = keypoint[i][0];
        int y_1 = keypoint[i][1];
        printf("(x1, y1): %d, %d\n",x_1,y_1);

        cv::Point point1 = cv::Point(x_1, y_1);
        int x_2 = int(x_1 + sin(theta_map[x_1][y_1])*m_map[x_1][y_1]*0.5);
        int y_2 = int(y_1 + cos(theta_map[x_1][y_1])*m_map[x_1][y_1]*0.5);

        printf("(x2, y2): %d, %d\n",x_2,y_2);
        cv::Point point2 = cv::Point(x_2, y_2);
        arrowedLine(dst, point1, point2, cv::Scalar(0, 255, 255), 1);
    }
    cv::namedWindow("keypoint_arrow", cv::WINDOW_AUTOSIZE); 
    cv::imshow("keypoint_arrow", dst);
    cv::waitKey(0);

    return;
}

void SIFTfeatureDescriptor::l2_norm_clip(float* feature, const float threshold)
{
    float accum = 0.0;
    while(true)
    {
        for(int i=0;i<128;i++)
        {
            accum += feature[i]*feature[i];
        }
        float norm = sqrt(accum);

        bool check = true;
        for(int i=0;i<128;i++)
        {
            feature[i] = feature[i]/norm;
            if(feature[i]>0.2)
            {
                check = false;
                feature[i]=0.2;
            }
        }
        if(check==true) {break;}
    }
}

bool SIFTfeatureDescriptor::check_position(int pos[2])
{
    if((pos[0]-7<=0)||(pos[0]+8>=_height-1)){
        return 0;
    }
    if((pos[1]-7<=0)||(pos[1]+8>=_width-1)){
        return 0;
    }
    return 1;
}

void SIFTfeatureDescriptor::fill_bin(float* F, const float _m, const float _theta)
{
    int degree = int(_theta*180/PI);
    if(degree<0) { degree+=360; }

    for(int i=0;i<8;i++)
    {
        if(degree==degree_map[i])
        {
            F[i]+=_m;
        }
        else if(degree_map[i]<degree && degree<degree_map[i+1])
        {
            //printf("d1: %d, d2: %d\n",degree-degree_map[i],degree_map[i+1]-degree);
            //printf("v1: %f, v2: %f\n",cos((degree-degree_map[i])*PI/180),cos((degree_map[i+1]-degree)*PI/180));
            F[i]    += _m*cos((degree-degree_map[i])*PI/180);
            if(i==7) { F[0] += _m*cos((degree_map[i+1]-degree)*PI/180); }
            else { F[i+1] += _m*cos((degree_map[i+1]-degree)*PI/180); }
        }
    }
    return;
}


void SIFTfeatureDescriptor::retrieve_4X4_feature(const int pos_x, const int pos_y, float* feature, const int iter)
{   
    float* F = new float[8];
    for(int i=0;i<8;i++) { F[i] = 0.0; }

    int feature_index = 8*iter;
    int delta_x[4] = {-3, -2, -1, 0};
    int delta_y[4] = {-3, -2, -1, 0};
    for(int x_i=0;x_i<4;x_i++)
    {
        for(int y_i=0;y_i<4;y_i++)
        {
            int _x = pos_x + delta_x[x_i];
            int _y = pos_y + delta_y[y_i];
            float _m = m_map[_x][_y];
            float _theta = theta_map[_x][_y];
            fill_bin(F, _m, _theta);
        }
    }
    for(int i=0;i<8;i++)
    {
        feature[feature_index+i] = F[i];
    }
    return;
}

void SIFTfeatureDescriptor::generate_keypoint_descriptor()
{
    all_feature = new float*[keypoint_num];
    for(int i=0;i<keypoint_num;i++)
    {
        if(!check_position(keypoint[i])){
            continue;
        }
        float* feature = new float[128];
        int _x = keypoint[i][0];
        int _y = keypoint[i][1];
        int x_delta_pos[4] = {-4, 0, 4, 8};
        int y_delta_pos[4] = {-4, 0, 4, 8};
        for(int x_i=0;x_i<4;x_i++)
        {
            for(int y_i=0;y_i<4;y_i++)
            {
                retrieve_4X4_feature(_x+x_delta_pos[x_i], _y+y_delta_pos[y_i], feature, x_i*4+y_i);
            }
        }
        l2_norm_clip(feature, 0.2);
        all_feature[i] = feature;   
    }
    return;
}
void SIFTfeatureDescriptor::run()
{
    cv::Mat gaussianblur;
    GaussianBlur(_image_gray, gaussianblur, cv::Size(3,3) ,0);
    generate_m_map(gaussianblur);
    //draw_arrow();
    generate_keypoint_descriptor();
    for(int i=0;i<128;i++)
    {
        printf("%f ",all_feature[keypoint_num-1][i]);
    }
    printf("\n");
    return;
}