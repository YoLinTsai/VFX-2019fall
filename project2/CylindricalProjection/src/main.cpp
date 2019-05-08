#include "Projector.h"
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int main(int argc, char *argv[])
{
    if (argc != 4) {
        std::cout << "Usage: ./CylindricalProjector <image directory> <focal length info> <output directory>" << std::endl;
        exit(-1);
    }
    char* image_dir       = argv[1];
    char* focallengthinfo = argv[2];
    char* output_dir      = argv[3];

    // read info file
    std::vector<std::pair<std::string, double> > info;
    std::ifstream file;
    file.open(focallengthinfo);
    std::string buffer;
    while(getline(file, buffer)) {
        if (!buffer.length()) continue;
        std::istringstream iss(buffer);
        std::string filename;
        double focallength;
        iss >> filename;
        iss >> buffer;
        focallength = std::stof(buffer);
        info.push_back(std::pair<std::string, double>(filename, focallength));
    }
    file.close();

    int count = 0;
    for (auto it = info.begin(); it != info.end(); ++it) {
        Projector p(std::string(image_dir)+"/"+(*it).first, std::string(output_dir)+"/"+(*it).first, (*it).second);
        p.RUN();
        p.show();
        p.save();
        ++count;
    }
    std::cout << "[ " << count << " images projected! ]" << std::endl;

//     Projector p(image, output, focallength);
//     p.RUN();
//     p.show();
//     p.save();

    return 0;
}
