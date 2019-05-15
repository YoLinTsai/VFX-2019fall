#include "descriptor.h"
#include <iostream>
#include <fstream>
#include <numeric>

const int SIFTfeatureDescriptor::degree_map[] = {0, 45, 90, 135, 180, 225, 270, 315, 360}; 
const int SIFTfeatureDescriptor::degree_bucket[] = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 
    110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 
    260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360}; 

void SIFTfeatureDescriptor::FilterCreation(float** GKernel, int k_size, float _s)
{ 
    // intialising standard deviation to 1.0 
    float sigma = _s; 
    float r, s = 2.0 * sigma * sigma; 
  
    // sum is for normalization 
    double sum = 0.0; 
    int l_bound = -k_size/2;
    int r_bound =(k_size/2)+1;

    // generating 5x5 kernel 
    for (int x = l_bound; x < r_bound; x++) { 
        for (int y = l_bound; y < r_bound; y++) { 
            r = sqrt(x * x + y * y); 
            GKernel[x+r_bound-1][y+r_bound-1] = (exp(-(r * r) / s)) / (M_PI * s);
            sum += GKernel[x+r_bound-1][y+r_bound-1]; 
        } 
    }

    // normalising the Kernel 
    for (int i = 0; i < k_size; ++i) 
        for (int j = 0; j < k_size; ++j)
        {
            GKernel[i][j] /= sum;
        }
} 

void SIFTfeatureDescriptor::read_image(char* filename)
{
    
	std::cerr << std::endl << "[ Reading image \"" << filename << "\" ]" << std::endl;
    _image_rgb = cv::imread(filename, 1);
    _height = _image_rgb.rows;
    _width = _image_rgb.cols;
    if (!_image_rgb.data) {
        std::cerr << "\t> Failed! Could not open or find the image!" << std::endl;
    }
    else {
        std::cerr << "\t> Success! Image scale " << _image_rgb.rows << " x " << _image_rgb.cols << std::endl;
        std::cerr << "\t> Converting to gray scale..." << std::endl;
        cv::cvtColor(_image_rgb, _image_gray, cv::COLOR_BGR2GRAY);
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

template <typename T>
int find_max_index(const std::vector<T> &v) {

  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  std::sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

  return idx[0];
}

//calculate m_Mat theta_Mat
void SIFTfeatureDescriptor::generate_m_map(const cv::Mat L)
{
    int height = L.rows; 
    int width = L.cols;

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
            float a = (L.at<uchar>(i+1, j) - L.at<uchar>(i-1, j)); //x
            float b = (L.at<uchar>(i, j-1) - L.at<uchar>(i, j+1)); //y
            m_map[i][j] = sqrt( pow(a, 2) + pow(b, 2));

            if(a!=0){
                if(b/a<0)
                {
                    if(a<0){
                        theta_map[i][j] = -atan(-b/a)+PI;   
                    }
                    else{
                        theta_map[i][j] = -atan(-b/a)+2*PI;
                    }
                }
                else if(b/a>0){
                    if((b<0)&&(a<0)){
                        theta_map[i][j] = atan(b/a)+PI;
                    }
                    else{
                        theta_map[i][j] = atan(b/a);
                    }       
                }
                else{
                    if(a>0){theta_map[i][j] = 0;}
                    else{theta_map[i][j] = PI;}
                }
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

void SIFTfeatureDescriptor::draw_arrow_orientation()
{
    cv::Mat dst;
    cv::cvtColor(_image_gray, dst, cv::COLOR_GRAY2RGB);
    for(int i=0;i<keypoint_num;i++)
    {
        int x_1 = keypoint[i][0];
        int y_1 = keypoint[i][1];
        cv::Point point1 = cv::Point(x_1, y_1);

        float theta = float(all_orientation[i]*PI/180);
        //float theta = theta_map[x_1][y_1];

        int x_2 = int(x_1 + cos(theta)*15);
        int y_2 = int(y_1 - sin(theta)*15);
        cv::Point point2 = cv::Point(x_2, y_2);

        arrowedLine(dst, point1, point2, cv::Scalar(0, 255, 255), 1);
        cv::circle(dst, point1, 1, cv::Scalar(255), -1);
    }
    cv::namedWindow("keypoint_arrow", cv::WINDOW_AUTOSIZE); 
    cv::imshow("keypoint_arrow", dst);
    cv::waitKey(0);
    return;
}

void SIFTfeatureDescriptor::draw_arrow_theta()
{
    cv::Mat dst;
    cv::cvtColor(_display, dst, cv::COLOR_GRAY2RGB);
    for(int i=0;i<keypoint_num;i++)
    {
        int x_1 = keypoint[i][0]*20+10;
        int y_1 = keypoint[i][1]*20+10;
        cv::Point point1 = cv::Point(x_1, y_1);

        float theta = theta_map[keypoint[i][1]][keypoint[i][0]];

        int x_2 = int(x_1 + cos(theta)*m_map[keypoint[i][0]][keypoint[i][1]]*0.3);
        int y_2 = int(y_1 - sin(theta)*m_map[keypoint[i][0]][keypoint[i][1]]*0.3);
        cv::Point point2 = cv::Point(x_2, y_2);

        arrowedLine(dst, point1, point2, cv::Scalar(0, 255, 255), 1);
        cv::circle(dst, point1, 1, cv::Scalar(255, 0, 0), -1);
    }
    cv::namedWindow("keypoint_arrow", cv::WINDOW_AUTOSIZE); 
    cv::imshow("keypoint_arrow", dst);
    cv::waitKey(0);
    cv::imwrite("theta.png",dst);
    return;
}

void SIFTfeatureDescriptor::l2_norm_clip(float* feature, const float threshold, int sz)
{
    float accum = 0.0;
    for(int i=0;i<sz;i++)
    {
        accum += feature[i];
    }
    float mean = accum/sz;
    accum = 0.0;
    for(int i=0;i<sz;i++)
    {
        accum+=(feature[i]-mean)*(feature[i]-mean);
    }
    float std = sqrt(accum/sz);
    for(int i=0;i<sz;i++)
    {
        feature[i] = (feature[i] - mean)/std;
    }
    /*
    float accum = 0.0;
    for(int i=0;i<sz;i++)
    {
        accum += feature[i]*feature[i];
    }
    float norm = sqrt(accum);

    bool check = true;
    for(int i=0;i<sz;i++)
    {
        feature[i] = feature[i]/norm;
        if(feature[i]> threshold)
        {
            check = false;
            feature[i] = threshold;
        }
    }
    if(check==false)
    {
        accum = 0.0;
        for(int i=0;i<sz;i++)
        {
            accum += feature[i]*feature[i];
        }
        norm = sqrt(accum);

        for(int i=0;i<sz;i++)
        {
            feature[i] = feature[i]/norm;
        }
    }
    */

    return;
}


//!!!!!!!!!!!!!!!! check change 20*sqrt(2) = 28

bool SIFTfeatureDescriptor::check_position(int pos[2])
{
    if((pos[0]-10<=0)||(pos[0]+10>=_width-1)){
        return 1;
    }
    if((pos[1]-10<=0)||(pos[1]+10>=_height-1)){
        return 1;
    }
    return 1;
}

void SIFTfeatureDescriptor::fill_bin(float* F, const float _m, const float _theta, int orientation)
{
    int degree = int(_theta*180/PI) - orientation;
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

void SIFTfeatureDescriptor::retrieve_4X4_feature(const int pos_x, const int pos_y, float* feature, const int iter, int orientation)
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
            float G = Gaussian_17x17[(iter/4)*4+delta_x[x_i]+3][(iter%4)*4+delta_y[y_i]+3];
            fill_bin(F, _m*G*10, _theta, orientation);
        }
    }
    for(int i=0;i<8;i++)
    {
        feature[feature_index+i] = F[i];
    }
    return;
}

int SIFTfeatureDescriptor::get_weighted_orientation(int pos_x, int pos_y)
{
    int delta_pos[5] = {-2, -1, 0, 1, 2};
    std::vector<float> bin_bucket(36, 0.0);
    float g_m_map[5][5] = {0.0};
    Gaussian_9x9 = new float*[5];
    for(int i=0;i<5;i++){Gaussian_9x9[i]=new float[5];}
    FilterCreation(Gaussian_9x9, 5, 1.5);
    for(int i=0;i<5;i++)
    {
        for(int j=0;j<5;j++)
        {
            int _x = pos_x + delta_pos[i];
            int _y = pos_y + delta_pos[i];
            g_m_map[i][j] = m_map[_x][_y] * Gaussian_9x9[i][j];
        }
    }

    for(int i=0;i<5;i++)
    {
        for(int j=0;j<5;j++)
        {
            int _x = pos_x + delta_pos[i];
            int _y = pos_y + delta_pos[j];
            float _theta = theta_map[_x][_y];
            int degree = int(_theta*180/PI);
            for(int k=0;k<36;k++)
            {
                if((degree_bucket[k]<=degree) && (degree<degree_bucket[k+1]))
                {
                    bin_bucket[k]+=g_m_map[i][j];
                }
            }
        }
    }
    int max_index = find_max_index(bin_bucket);
    //if(pos_x==9&&pos_y==9) printf("orientation: %d\n", (degree_bucket[max_index]+degree_bucket[max_index+1])/2);
    return (degree_bucket[max_index]+degree_bucket[max_index+1])/2;
}

void SIFTfeatureDescriptor::retrieve_MSOP_feature(int pos_x, int pos_y, float orientation, float* feature)
{
    
    int delta_pixel[11] = {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5};
    for(int i=0;i<11;i++)
    {
        for(int j=0;j<11;j++)
        {
            int delta_x = delta_pixel[i];
            int delta_y = delta_pixel[j];
            int _x = pos_x + delta_x;
            int _y = pos_y + delta_y;
            feature[i+j*11] = _image_gray.at<uchar>(_x, _y);
        }
    }
    
    /*
    float theta = (90-orientation)*PI/180;
    int delta_pos[8] = {-15, -10, -5, 0, 5, 10, 15, 20};
    int delta_pixel[5] = {-4, -3, -2, -1, 0};
    for(int i=0;i<8;i++)
    {
        for(int j=0;j<8;j++)
        {
            float sum = 0.0;
            for(int k=0;k<5;k++)
            {
                for(int l=0;l<5;l++)
                {
                    int delta_x = delta_pos[i] + delta_pixel[k];
                    int delta_y = delta_pos[j] + delta_pixel[l];
                    int _x = pos_x + delta_x;
                    int _y = pos_y + delta_y;
                    //int x_r = int(float(delta_x)*cos(theta)-float(-delta_y)*sin(theta));
                    //int y_r = int(float(delta_x)*sin(theta)+float(-delta_y)*cos(theta));
                    //int _x = pos_x + x_r;
                    //int _y = pos_y - y_r;
                    sum += _image_gray.at<uchar>(_x, _y);
                }
            }
            feature[i+j*8] = sum/25;
        }
    }
    */
    return;
}

void SIFTfeatureDescriptor::generate_keypoint_descriptor()
{
    /*
    Gaussian_17x17 = new float*[17];
    for(int i=0;i<17;i++){Gaussian_17x17[i]=new float[17];}
    FilterCreation(Gaussian_17x17, 17, 2);
    */
    for(int i=0;i<keypoint_num;i++)
    {
        if(!check_position(keypoint[i])){
            continue;
        }
        float* feature = new float[121];
        int _x = (keypoint[i][0]);
        int _y = (keypoint[i][1]);


        int orientation;
        orientation = get_weighted_orientation(_x, _y);
        all_orientation.push_back(orientation);
        retrieve_MSOP_feature(_y, _x, orientation, feature);
        /*
        int x_delta_pos[4] = {-4, 0, 4, 8};
        int y_delta_pos[4] = {-4, 0, 4, 8};
        for(int x_i=0;x_i<4;x_i++)
        {
            for(int y_i=0;y_i<4;y_i++)
            {
                retrieve_4X4_feature(_x+x_delta_pos[x_i], _y+y_delta_pos[y_i], feature, x_i*4+y_i, orientation);
            }
        }
        l2_norm_clip(feature, 0.2);
        */
        l2_norm_clip(feature, 0.2, 121);
        all_feature.push_back(feature);
    }
    return;
}

void SIFTfeatureDescriptor::gaussian_test()
{
    int k_size = 17;
    Gaussian = new float*[k_size];
    for(int i=0;i<k_size;i++)
    {
        Gaussian[i] = new float[k_size];
    }
    Gaussian[0][0] = 0.5;
    FilterCreation(Gaussian, k_size, 3);

    cv::Mat _gaussian_gray = cv::Mat(17, 17, CV_8U, cv::Scalar(0));

    for(int i=0;i<k_size;i++)
    {
        for(int j=0;j<k_size;j++)
        {
            _gaussian_gray.at<uchar>(i,j) = int(Gaussian[i][j]*14000);
        }
    }

    _display = cv::Mat(340, 340, CV_8U, cv::Scalar(0));
    cv::Mat _m_map_display = cv::Mat(340, 340, CV_8U, cv::Scalar(0));

    for(int i=0;i<17;i++)
    {
        for(int j=0;j<17;j++)
        {
            for(int k=0;k<20;k++)
            {
                for(int l=0;l<20;l++)
                {
                    _display.at<uchar>(i*20+k,j*20+l) = _gaussian_gray.at<uchar>(i,j);
                }
            }
        }
    }

    cv::imwrite("gaussian.png",_display);
    this->generate_m_map(_gaussian_gray);
    for(int i=1;i<16;i++)
    {
        for(int j=1;j<16;j++)
        {
            for(int k=0;k<20;k++)
            {
                for(int l=0;l<20;l++)
                {
                    _m_map_display.at<uchar>(i*20+k,j*20+l) = int(m_map[i][j]);
                }
            }
        }
    }
    cv::namedWindow("keypoint_arrow", cv::WINDOW_AUTOSIZE); 
    cv::imshow("keypoint_arrow", _m_map_display);
    cv::waitKey(0);
    cv::imwrite("m_map.png",_m_map_display);
    this->draw_arrow_theta();
}

void SIFTfeatureDescriptor::run()
{
    //this->gaussian_test();
    cv::GaussianBlur( _image_gray, _image_gray, cv::Size( 5, 5 ), 0, 0);
    this->generate_m_map(_image_gray);
    this->generate_keypoint_descriptor();
    this->draw_arrow_orientation();
    return;
}
