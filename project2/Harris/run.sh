if [ ! -d ../keypoints ]; then
    mkdir ../keypoints
fi;
./HarrisDetector ../data/projected_park_downsample/DSC03196.JPG.jpg 0.04 $1 ../keypoints/DSC03196.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03197.JPG.jpg 0.04 $1 ../keypoints/DSC03197.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03198.JPG.jpg 0.04 $1 ../keypoints/DSC03198.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03199.JPG.jpg 0.04 $1 ../keypoints/DSC03199.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03200.JPG.jpg 0.04 $1 ../keypoints/DSC03200.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03201.JPG.jpg 0.04 $1 ../keypoints/DSC03201.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03202.JPG.jpg 0.04 $1 ../keypoints/DSC03202.key custom
./HarrisDetector ../data/projected_park_downsample/DSC03203.JPG.jpg 0.04 $1 ../keypoints/DSC03203.key custom
