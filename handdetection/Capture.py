import cv2
import numpy as np

class Capture:

    def __init__(self,learning):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  300)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
        cv2.namedWindow('RealTimeHandDetection')
        self.t_minus = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.t = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.t_plus  = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.bgsub = cv2.createBackgroundSubtractorKNN()
        self.skinClassifier = learning

    def callableObject(x):
        pass

    def diffImg(self,t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

    def startCapture(self):
        while(True):
            kernel_square = np.ones((11,11),np.uint8)
            kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            ret, frame = self.cap.read()
            fgmask = self.bgsub.apply(frame)
            blur = cv2.blur(frame,(3,3))

            self.t_minus = self.t
            self.t = self.t_plus
            self.t_plus = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)

            frameDelta = self.diffImg(self.t_minus, self.t, self.t_plus)
            ret, thresh_motion = cv2.threshold(frameDelta, 30, 255, 0)
            motion_and_background = cv2.add(thresh_motion,fgmask)
            dilation = cv2.dilate(thresh_motion,kernel_square,iterations = 1)
            erosion = cv2.erode(dilation,kernel_square,iterations = 1)
            dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
            filtered = cv2.medianBlur(dilation2,5)
            cv2.imshow("Processed" , filtered)

            perdictedImage = self.skinClassifier.predict(frame)
            cv2.imshow("Processed" , perdictedImage)

            ret,thresh = cv2.threshold(filtered,127,255,0)
            image,contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max_area=100
            ci=0
            for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
            if ci == 0:
                continue
            cnts = contours[ci]
            hull = cv2.convexHull(cnts)
            hull2 = cv2.convexHull(cnts,returnPoints = False)
            cv2.drawContours(frame,[hull],-1,(255,255,255),2)
            cv2.imshow('Original',frame)
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break


    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()