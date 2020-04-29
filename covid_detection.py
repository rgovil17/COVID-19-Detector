import os
import sys
import argparse
import cv2
import numpy as np
import keras
from keras.models import *
from keras.preprocessing import image

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args["image"]

if not os.path.isfile(img_path):
	print("The specified path does not exist.")
	sys.exit()

## PREDICTION ===============

model = load_model("model_10.h5")

img = image.load_img(img_path, target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
pred = model.predict_classes(img)
# print(pred[0,0])

## ==========================

## DISPLAY =================

img_pic = cv2.imread(img_path)
img_pic = cv2.resize(img_pic,(700,700))
# cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Image',720,720)
while(1):
	cv2.imshow("Image",img_pic)
	res = 'positive' if pred == 0 else 'negative'
	cv2.putText(img_pic, 'Result: Corona '+res,(50,75),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
	if cv2.waitKey(20) & 0xFF ==27:
		break

cv2.destroyAllWindows()