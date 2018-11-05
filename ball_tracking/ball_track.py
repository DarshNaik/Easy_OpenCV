import cv2
import argparse
from imutils.video import VideoStream
from collections import deque
import time
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-v',"--video",help = "Path for (optional) video file of object motion")
ap.add_argument('-b', '--buffer', type = int, default = 64)
args = vars(ap.parse_args())

# greenL = (29, 86, 6)
# greenU = (64,255,255)
greenL = (110,50,50)
greenU = (130,255,255)
pts = deque(maxlen=args["buffer"])

if not args.get('video', False):
    vs = VideoStream(src = 0).start()
else:
    vs = cv2.VideoCapture(args['video'])

time.sleep(2.0)

while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    if frame is None:
        break

    frame = imutils.resize(frame,width=900)
    blurred = cv2.GaussianBlur(frame, (11,11),0)
    hsv = cv2.cvtColor(blurred , cv2.COLOR_BGR2HSV)

    # series of erode and dilation for the mask to remove
    # small dots left
    mask = cv2.inRange(hsv, greenL, greenU)
    mask = cv2.erode(mask,None,iterations = 2)
    mask = cv2.dilate(mask,None,iterations = 2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None

    if len(cnts)>0:
        c = max(cnts,key = cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        if radius>10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,0,255), 2)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)


    cv2.imshow('Frame',cv2.flip(frame,1))
    cv2.imshow('mask',mask)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

if not args.get("video", False):
	vs.stop()
else:
	vs.release()

cv2.destroyAllWindows()
