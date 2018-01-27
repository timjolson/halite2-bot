import numpy as np
from utils import blur, circle, plot3d, imshow, imread, imsave
import time
import cv2

# for fill in list(range(0,20,3)):
#     for r in list(range(1,27,5)):
#         blank = np.zeros([100, 100])
#         circle(blank, [50,50], r, 255, fill)
#         bb = blur(blank, r*1.2+2)
#         grad = np.gradient(bb)
#         grad_mag = np.sqrt(np.power(grad[0],2) + np.power(grad[1],2))
#         plot3d(range(100), range(100), bb).show()
#         # plot3d(range(100), range(100), grad_mag).show()
#         cv2.imshow('blank',blank)
#         cv2.imshow('bb',bb)
#         cv2.imshow('g',grad_mag)
#         cv2.waitKey(0)
#         # time.sleep(100)


for fill in list(range(-1,20,3)):
    blank = np.zeros([800, 800])
    circle(blank, [400,400], 380, 1, fill)
    cv2.imshow('',blank)
    cv2.waitKey(200)

for r in range(0,50, 10):
    for x in range(-5,170, 20):
        for y in range(-5, 90, 20):
    # for fill in [-1]:
            for fill in list(range(-1,10,3)):
                blank = np.zeros([71,151])
                circle(blank,[x,y], r, 1, fill)
                cv2.imshow('',blank)
                cv2.waitKey(50)
