
#-----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
# Converting each image to RGB from BGR format
bins= 8
def rgb_bgr(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img


# Conversion to HSV image format from RGB

def bgr_hsv(rgb_img):
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    return hsv_img



# image segmentation

# for extraction of green and brown color


def img_segmentation(rgb_img,hsv_img):
    lower_green = np.array([25,0,20])
    upper_green = np.array([100,255,255])
    healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
    result = cv2.bitwise_and(rgb_img,rgb_img, mask=healthy_mask)
    lower_brown = np.array([10,0,10])
    upper_brown = np.array([30,255,255])
    disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
    disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
    final_mask = healthy_mask + disease_mask
    final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
    return final_result




# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


image = cv2.imread('Medicinal Leaf Dataset/Segmented Medicinal Leaf Images/Alpinia Galanga (Rasna)/AG-S-001.jpg')
image = cv2.resize(image, (500, 500))

RGB_BGR       = rgb_bgr(image)
cv2.imshow("RGB_BGR",RGB_BGR)
BGR_HSV       = bgr_hsv(RGB_BGR)
cv2.imshow("BGR_HSV",BGR_HSV)
IMG_SEGMENT   = img_segmentation(RGB_BGR,BGR_HSV)
cv2.imshow("IMG_SEGMENT",IMG_SEGMENT)
cv2.waitKey(0)


fv_hu_moments = fd_hu_moments(IMG_SEGMENT)
fv_haralick   = fd_haralick(IMG_SEGMENT)
fv_histogram  = fd_histogram(IMG_SEGMENT)

# Concatenate 

global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

print(global_feature)