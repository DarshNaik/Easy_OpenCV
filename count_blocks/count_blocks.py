import cv2
import argparse
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = 'True',help = "path to input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original Image",image)
#######################write = "This is original img"

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 30, 150)

thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
output = image.copy()

for c in cnts:
	cv2.drawContours(output, [c], -1, (240, 0, 159), 3)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,	(240, 0, 159), 2)
cv2.imshow("contourss", output)
cv2.waitKey(0)
