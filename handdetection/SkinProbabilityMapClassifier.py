import cv2
import numpy as np

class SPMClassifier:
    name = "SPM Classifier"

    def predict(self,image):
        blur = cv2.blur(image,(3,3))
        hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        return mask