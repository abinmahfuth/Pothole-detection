#Credit to niconielsen32

import sys
import cv2
import numpy as np
import time
#import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

import time



# read the two images
img_right = cv2.imread('./stereo_img/face_wink_right.jpeg')               
img_left =  cv2.imread('./stereo_img/face_wink_left.jpeg')


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 6               #Distance between the cameras [cm]
f = 3.3              #Camera lense's focal length [mm]
alpha = 65        #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision



################## CALIBRATION #########################################################

frame_right, frame_left = calibration.undistortRectify(img_right, img_left)

########################################################################################

#Checkpoint #1: imshow the calibrated and uncalibrated:
"""
cv2.imshow('img_rihgt_raw', img_right)
cv2.imshow('img_right_calib', frame_right)

cv2.waitKey(0)

cv2.destroyAllWindows()
"""



################## CALCULATING DEPTH #########################################################

centre_r = (590,1010)
centre_l = (725,815)
#depth = tri.find_depth(centre_r, centre_l, frame_right, frame_left, B, f, alpha)
depth =  tri.find_depth(centre_r, centre_l, frame_right, frame_left, B, f, alpha)
print("Depth in cm: ", depth)



