if [ ! -d ../../boundingboxes ]; then
    mkdir ../../boundingboxes
fi;
./descriptor ../../data/projected_park_downsample/DSC03196.JPG.jpg ../../keypoints/DSC03196.key ../../data/projected_park_downsample/DSC03197.JPG.jpg ../../keypoints/DSC03197.key $1 ../../boundingboxes/196_197.box
./descriptor ../../data/projected_park_downsample/DSC03197.JPG.jpg ../../keypoints/DSC03197.key ../../data/projected_park_downsample/DSC03198.JPG.jpg ../../keypoints/DSC03198.key $1 ../../boundingboxes/197_198.box
./descriptor ../../data/projected_park_downsample/DSC03198.JPG.jpg ../../keypoints/DSC03198.key ../../data/projected_park_downsample/DSC03199.JPG.jpg ../../keypoints/DSC03199.key $1 ../../boundingboxes/198_199.box
./descriptor ../../data/projected_park_downsample/DSC03199.JPG.jpg ../../keypoints/DSC03199.key ../../data/projected_park_downsample/DSC03200.JPG.jpg ../../keypoints/DSC03200.key $1 ../../boundingboxes/199_200.box
./descriptor ../../data/projected_park_downsample/DSC03200.JPG.jpg ../../keypoints/DSC03200.key ../../data/projected_park_downsample/DSC03201.JPG.jpg ../../keypoints/DSC03201.key $1 ../../boundingboxes/200_201.box
./descriptor ../../data/projected_park_downsample/DSC03201.JPG.jpg ../../keypoints/DSC03201.key ../../data/projected_park_downsample/DSC03202.JPG.jpg ../../keypoints/DSC03202.key $1 ../../boundingboxes/201_202.box
./descriptor ../../data/projected_park_downsample/DSC03202.JPG.jpg ../../keypoints/DSC03202.key ../../data/projected_park_downsample/DSC03203.JPG.jpg ../../keypoints/DSC03203.key $1 ../../boundingboxes/202_203.box
rm ../../boundingboxes/201_202.box
echo "240 0 337 590" >> ../../boundingboxes/201_202.box
echo "0 10 97 600" >> ../../boundingboxes/201_202.box
