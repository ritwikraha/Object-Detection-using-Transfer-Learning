import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import numpy as np
import cv2

## Video input from cameras connected with the system.
## 0 -> Webcam, Use 1,2 etc depending on which USB port your camera is connected
cap = cv2.VideoCapture(0)

while(True):
	ret, frame = cap.read()

    #cv2.imshow('frames', frame)
	bbox, label, conf = cv.detect_common_objects(frame, confidence=0.25, model='yolov3-tiny')
	output_image = draw_bbox(frame, bbox, label, conf)
	cv2.imshow('detected', output_image)
    
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
