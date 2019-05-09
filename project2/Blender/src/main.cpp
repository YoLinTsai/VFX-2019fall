#include "Blender.h"
#include <iostream>

using std::cout;
using std::endl;

int main(int argc, char *argv[])
{
    /*
     * [bounding box format]
     * (the first and second line correspond to image1 and image2 respectively)
     * (the size of the two boxes must match)
     * X corresponds to column
     * Y corresponds to row
     *
     * LowerLeftX LowerLeftY UpperRightX UpperRightY
     * LowerLeftX LowerLeftY UpperRightX UpperRightY
     *
     * ex:
     * 284 0 383 511
     * 0 0 99 511
     */
    cout << "[NOTICE] The output image name must end with a legal suffix (ex: .jpg/.png etc.)" << endl;
    if (argc != 5) {
        cout << "Usage: ./Blender <image1> <image2> <bbox of the overlap region in txt file> <output image name>" << endl;
        exit(-1);
    }

    char* image1 = argv[1];
    char* image2 = argv[2];
    char* bbox   = argv[3];
    char* out    = argv[4];

    Blender b(image1, image2, bbox);
    b.blend();
    b.show();
    b.save(out);

    return 0;
}
