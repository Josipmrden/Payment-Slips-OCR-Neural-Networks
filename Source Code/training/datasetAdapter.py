from scipy import ndimage

from scipy.ndimage import measurements

import cv2
import numpy as np

WHITE = [255, 255, 255]

def adapt_picture(img, midSize, newSize):
    image_shape = img.shape[:2]
    oldh, oldw = image_shape

    thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    if oldh >= oldw:
        freeSpace = int((oldh - oldw) / 2)

        img = cv2.copyMakeBorder(img, 0, 0, freeSpace, freeSpace, cv2.BORDER_CONSTANT, value=0)
    else:
        freeSpace = int((oldw - oldh) / 2)

        img = cv2.copyMakeBorder(img, freeSpace, freeSpace, 0, 0, cv2.BORDER_CONSTANT, value=0)

    img = cv2.resize(img, (midSize, midSize))
    padding = int((newSize - midSize) / 2)


    img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

    shiftx, shifty = getBestShift(img)
    shifted = shift(img, shiftx, shifty)
    img = shifted
    return img


def getBestShift(img):
    cy, cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted