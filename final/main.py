from contentwarp import ContentWarp
from argparse import ArgumentParser
import sys
import os


def main():
    parser = ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--feature_dir', required=True)
    args = parser.parse_args()

    images   = os.listdir(args.image_dir)
    features = os.listdir(args.feature_dir)

    image_id = [ (int(img.split('.')[0]), os.path.join(args.image_dir, img)) for img in images ]
    feature_id = [ (int(feat.split('.')[0]), os.path.join(args.feature_dir, feat)) for feat in features ]

    image_id.sort(key=lambda tup: tup[0])
    feature_id.sort(key=lambda tup: tup[0])

    data = zip(image_id, feature_id)
    for image, feat in data:
        print ('Wapping image:', image[1])
        warp = ContentWarp(image[1], feat[1], 18, 32, alpha=0.8)
        warp.warp()


if __name__ == '__main__':
    main()
