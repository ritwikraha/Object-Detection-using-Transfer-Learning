import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np
import cv2

img = cv2.imread('test.jpg')
bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov3-tiny')

output_image = draw_bbox(img, bbox, label, conf)

cv2.imwrite('detected.jpg',output_image)
