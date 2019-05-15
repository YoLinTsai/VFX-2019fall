if [ ! -d ../data/projected_park_downsample ]; then
    mkdir ../data/projected_park_downsample/;
fi;
./CylindricalProjector ../data/park-downsample/ ../data/park-downsample/FocalLengthINFO.txt ../data/projected_park_downsample/
