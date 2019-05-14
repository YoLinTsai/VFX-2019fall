#include "Harris.h"
#include "minHeap.h"
#include <fstream>

int main(int argc, char *argv[])
{
    if (argc != 6) {
        std::cout << "Usage: ./detector <image> <k> <number of features> <output filename> <sobel mode: opencv/custom>" << std::endl;
        exit(0);
    }
    double k = std::stof(argv[2]);
    int numFeatures = std::stoi(argv[3]);
    char* image = argv[1];
    char* output = argv[4];
    char* mode = argv[5];
    std::ofstream file(output);

    HarrisFeatureDetector detector(k, numFeatures, mode);
    detector.read_image(image);
    detector.run();
    detector.dump(file);
    // detector.show_images();
    return 0;
}
