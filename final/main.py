from contentwarp import ContentWarp
import sys


def main():
    warp = ContentWarp(sys.argv[1], '0007.txt', 18, 32, alpha=0.5)


if __name__ == '__main__':
    main()
