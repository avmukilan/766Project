import cv2
import numpy as np

class Capture:

    def __init__(self,learning):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  400)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
        cv2.namedWindow('RealTimeHandDetection')
        self.t_minus = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.t = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.t_plus  = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_RGB2GRAY)
        self.bgsub = cv2.createBackgroundSubtractorKNN()
        self.skinClassifier = learning
        self.processed_old = self.t_minus
        self.processed = self.t

    def callableObject(x):
        pass

    def diffImg(self,t0, t1, t2):
        d1 = cv2.absdiff(t2, t1)
        d2 = cv2.absdiff(t1, t0)
        return cv2.bitwise_and(d1, d2)

    def startCapture(self):
        face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        fingerdetect = cv2.CascadeClassifier('cascade.xml')
        handetect = cv2.CascadeClassifier('hand.xml') # detected closed palm

        while(True):
            kernel_square = np.ones((11,11),np.uint8)
            kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            ret, frame = self.cap.read()
            fgmask = self.bgsub.apply(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.blur(frame,(3,3))
            hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

            self.t_minus = self.t
            self.t = self.t_plus
            self.t_plus = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frameDelta = self.diffImg(self.t_minus, self.t, self.t_plus)
            ret, thresh_motion = cv2.threshold(frameDelta, 30, 255, 0)
            motion_and_background = cv2.bitwise_or(thresh_motion,fgmask)

            cv2.imshow("Motion With Background Subtraction", motion_and_background)
            newFrame = cv2.bitwise_and(frame,frame,mask=motion_and_background)
            predictedImage = self.skinClassifier.predict(newFrame)
            cv2.imshow("Predicted " , predictedImage)

            processed = cv2.bitwise_and(motion_and_background, motion_and_background, mask=predictedImage)

            dilation = cv2.dilate(processed,kernel_square,iterations = 1)
            erosion = cv2.erode(dilation,kernel_square,iterations = 1)
            dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
            processed = cv2.medianBlur(dilation2,5)

            cv2.imshow("Processed", processed)

            maskedFrame = cv2.bitwise_and(frame,frame,mask=processed)
            cv2.imshow("Processed Frame",maskedFrame)
            grayFrame = cv2.cvtColor(maskedFrame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("GrayScale",grayFrame)
            facepixels = face.detectMultiScale(grayFrame,1.3,5)

            for (x,y,w,h) in facepixels:
                processed[x:x+w,y:y+h] = [0,0,0]

            fingerPixels = fingerdetect.detectMultiScale(maskedFrame,1.3,5)
            handpixels = handetect.detectMultiScale(maskedFrame,1.3,5)

            #for (x,y,w,h) in handpixels:
            #    frame[y:y+h,x:x+w] = [255,255,255]
            #for (x,y,w,h) in fingerPixels:
            #    frame[y:y+h,x:x+w] = [255,255,255]

            ret,thresh = cv2.threshold(processed,127,255,0)
            image,contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            max_area=3000

            for i in range(len(contours)):
                cnt=contours[i]
                if max_area <= cv2.contourArea(cnt):
                    cnts = contours[i]
                    hull = cv2.convexHull(cnts)
                    hull2 = cv2.convexHull(cnts,returnPoints = False)
                    defects = cv2.convexityDefects(cnts,hull2)

                    FarDefect = []
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnts[s][0])
                        end = tuple(cnts[e][0])
                        far = tuple(cnts[f][0])
                        FarDefect.append(far)
                        cv2.line(frame,start,end,[0,255,0],1)
                        cv2.circle(frame,far,10,[100,255,255],3)

                    moments = cv2.moments(cnts)
                    if moments['m00']!=0:
                        cx = int(moments['m10']/moments['m00'])
                        cy = int(moments['m01']/moments['m00'])
                    center_mass=(cx, cy)
                    cv2.circle(frame, center_mass, 7, [100, 0, 255], 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame,'Center', tuple(center_mass), font, 2, (255, 255, 255), 2)
                    distance_between_defects_to_center = []
                    for i in range(0,len(FarDefect)):
                        x =  np.array(FarDefect[i])
                        center_mass = np.array(center_mass)
                        distance = np.sqrt(np.power(x[0] - center_mass[0], 2) + np.power(x[1] - center_mass[1], 2))
                        distance_between_defects_to_center.append(distance)

                    #Get an average of three shortest distances from finger webbing to center mass
                    sorted_defects_distances = sorted(distance_between_defects_to_center)
                    average_Defect_Distance = np.mean(sorted_defects_distances[0:2])

                    finger = []
                    for i in range(0,len(hull)-1):
                        if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                            if hull[i][0][1] <500 :
                                finger.append(hull[i][0])

                    finger =  sorted(finger,key=lambda x: x[1])
                    fingers = finger[0:5]

                    fingerDistance = []
                    for i in range(0,len(fingers)):
                        distance = np.sqrt(np.power(fingers[i][0] - center_mass[0], 2) + np.power(fingers[i][1] - center_mass[0], 2))
                        fingerDistance.append(distance)
                    result = 0
                    for i in range(0,len(fingers)):
                        if fingerDistance[i] > average_Defect_Distance+130:
                            result = result +1

                    #cv2.putText(frame,str(result),(150,100),font,2,(255,255,255),2)
                    x,y,w,h = cv2.boundingRect(cnts)
                    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
            cv2.imshow('Detection',frame)

            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                break


    def destroy(self):
        fps = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(fps)
        self.cap.release()
        cv2.destroyAllWindows()