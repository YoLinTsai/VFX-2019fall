from contentwarp import ContentWarp
import sys


def main():
    warp = ContentWarp(sys.argv[1], 'feat.txt', 24, 32)


if __name__ == '__main__':
    main()
