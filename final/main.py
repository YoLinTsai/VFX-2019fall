from contentwarp import ContentWarp
import sys


def main():
    warp = ContentWarp(sys.argv[1], 'feat.txt', 24, 32, alpha=0.01)


if __name__ == '__main__':
    main()
