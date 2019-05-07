#include "Projector.h"

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "Usage: ./CylindricalProjector <image> <focal length> <output image>" << std::endl;
        exit(-1);
    }
    char* image       = argv[1];
    int   focallength = std::stoi(argv[2]);
    char* output      = argv[3];

    Projector p(image, output, focallength);
    p.RUN();
    p.show();
    p.save();

    return 0;
}
