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

    print (images)
    print (features)

    sys.exit()
    warp = ContentWarp(sys.argv[1], sys.argv[2], 18, 32, alpha=0.5)


if __name__ == '__main__':
    main()
