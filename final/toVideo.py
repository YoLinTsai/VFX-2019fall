import cv2, os, sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--sequence_dir', required=True)
args = parser.parse_args()

path = [ os.path.join(args.sequence_dir, p) for p in os.listdir(args.sequence_dir) ]
images = []

for img in path:
    imagefile = img.split('/')[-1]
    if imagefile.startswith('.'): continue
    images.append((int(imagefile.split('.')[0]), cv2.imread(img)))

images.sort(key=lambda tup: tup[0])

size = images[0][1].shape[:2][::-1]

writer = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 24.0, size, True)

for img in images: writer.write(img[1])

cv2.destroyAllWindows()
