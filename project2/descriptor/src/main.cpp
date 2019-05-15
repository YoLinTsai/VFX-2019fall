#include "descriptor.h"
#include "match.h"
#include <fstream>

int main(int argc, char *argv[])
{
	char* img_path_1 = argv[1];
	char* keypoint_path_1 = argv[2];
	char* img_path_2 = argv[3];
	char* keypoint_path_2 = argv[4];
	int keypoint_num = atoi(argv[5]);
	char* bbox_filename = argv[6];
	SIFTfeatureDescriptor descriptor;
	descriptor.read_image(img_path_1);
	descriptor.read_keypoint(keypoint_path_1, keypoint_num);
	descriptor.run();

	SIFTfeatureDescriptor _descriptor;
	_descriptor.read_image(img_path_2);
	_descriptor.read_keypoint(keypoint_path_2, keypoint_num);
	_descriptor.run();

	Matcher matcher(bbox_filename);
	matcher.feature_match(descriptor.all_feature, _descriptor.all_feature,
								descriptor.keypoint, _descriptor.keypoint);
	matcher.draw_match_pair(descriptor._image_rgb, _descriptor._image_rgb, 
							descriptor.keypoint, _descriptor.keypoint);
	return 0;
}
