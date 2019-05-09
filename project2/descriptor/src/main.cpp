#include "descriptor.h"
#include <fstream>

int main(int argc, char *argv[])
{
	char* img_path = argv[1];
	char* keypoint_path = argv[2];
	int keypoint_num = atoi(argv[3]);
	SIFTfeatureDescriptor descriptor;
	descriptor.read_image(img_path);
	descriptor.read_keypoint(keypoint_path, keypoint_num);
	descriptor.run();
	return 0;
}