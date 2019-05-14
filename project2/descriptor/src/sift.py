'''
import cv2

img = cv2.imread('../../data/grail/grail01.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
print(len(kp))

keypoint = kp[139:]
print(len(keypoint))
img=cv2.drawKeypoints(gray,kp[139:], None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

with open('keypoint3.txt', 'w') as writer:
	writer.write('row,col\n')
	for i in range(len(keypoint)):
		writer.write('{},{}\n'.format(int(keypoint[i].pt[0]), int(keypoint[i].pt[1])))
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('../../data/grail/grail00.jpg',0)          # queryImage
img2 = cv2.imread('../../data/grail/grail01.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
with open('keypoint4.txt', 'w') as writer:
	writer.write('row,col\n')
	for i in range(len(kp1)):
		writer.write('{},{}\n'.format(int(kp1[i].pt[0]), int(kp1[i].pt[1])))

with open('keypoint5.txt', 'w') as writer:
	writer.write('row,col\n')
	for i in range(len(kp2)):
		writer.write('{},{}\n'.format(int(kp2[i].pt[0]), int(kp2[i].pt[1])))

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()
