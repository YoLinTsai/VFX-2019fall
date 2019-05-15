# Compile codes
cd CylindricalProjection
# make
make clean && make
cd ..
cd Harris
# make
make clean && make
cd ..
cd descriptor/src/
# make
make clean && make
cd ../..
cd Blender
# make
make clean && make
cd ..

# generate the projected images
cd CylindricalProjection
./run.sh
cd ..

# detect keypoints
cd Harris
./run.sh 500
cd ..

# matching
cd descriptor/src/
./run.sh 500
cd ../..

cd Blender
./run.sh
cd ..
