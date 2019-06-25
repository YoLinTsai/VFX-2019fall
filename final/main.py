from warp import Warp
from argparse import ArgumentParser
import sys
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--feature_dir', required=True)
    parser.add_argument('--grid_dir', required=True)
    parser.add_argument('--warp_dir', required=True)
    args = parser.parse_args()

    images   = os.listdir(args.image_dir)
    features = os.listdir(args.feature_dir)

    image_id = [ (int(img.split('.')[0]), os.path.join(args.image_dir, img)) for img in images ]
    feature_id = [ (int(feat.split('.')[0]), os.path.join(args.feature_dir, feat)) for feat in features ]

    image_id.sort(key=lambda tup: tup[0])
    feature_id.sort(key=lambda tup: tup[0])

    data = zip(image_id, feature_id)
    for image, feat in data:
        # if image[0] != 19: continue
        print ('Processing image:', image[1])
        warp = Warp(image[1], feat[1], 18, 32, args.grid_dir, args.warp_dir, alpha=20)
        warp.warp()
        print ('')


if __name__ == '__main__':
    main()
