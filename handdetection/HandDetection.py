import cv2
import numpy as np

#Open Camera object
capture = cv2.VideoCapture(0)

#Decrease frame size
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

# Creating a window for HSV track bars
cv2.namedWindow('HSV_TrackBar')
# Starting with 100's to prevent error while masking
#h,s,v = 100,100,100

def callableObject(x):
    pass

# Creating track bar
cv2.createTrackbar('h', 'HSV_TrackBar',0,179,callableObject)
cv2.createTrackbar('s', 'HSV_TrackBar',0,255,callableObject)
cv2.createTrackbar('v', 'HSV_TrackBar',0,255,callableObject)

while(True):
    #Capture frames from the camera
    ret, frame = capture.read()

    #Blur the image
    blur = cv2.blur(frame,(3,3))

    #Convert to HSV color space
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    #Create a binary image with where white will be skin colors and rest is black
    mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))

    #Kernel matrices for morphological transformation
    kernel_square = np.ones((11,11),np.uint8)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #Perform morphological transformations to filter out the background noise
    dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
    erosion = cv2.erode(dilation,kernel_square,iterations = 1)
    dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)
    filtered = cv2.medianBlur(dilation2,5)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
    dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
    median = cv2.medianBlur(dilation2,5)
    ret,thresh = cv2.threshold(median,127,255,0)

    image,contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Dilation',median)

    #Find Max contour area (Assume that hand is in the frame)
    max_area=100
    ci=0
    for i in range(len(contours)):
        cnt=contours[i]
        area = cv2.contourArea(cnt)
        if(area>max_area):
            max_area=area
            ci=i
    cnts = contours[ci]
    hull = cv2.convexHull(cnts)
    #Find convex defects
    hull2 = cv2.convexHull(cnts,returnPoints = False)
    defects = cv2.convexityDefects(cnts,hull2)

    #Get defect points and draw them in the original image
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

    #Draw center mass
    cv2.circle(frame, center_mass, 7, [100, 0, 255], 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'Center', tuple(center_mass), font, 2, (255, 255, 255), 2)

    #Distance from each finger defect(finger webbing) to the center mass
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

    #Finger is pointed/raised if the distance of between fingertip to the center mass is larger
    result = 0
    for i in range(0,len(fingers)):
        if fingerDistance[i] > average_Defect_Distance+130:
            result = result +1

    cv2.putText(frame,str(result),(100,100),font,2,(0,0,0),2)
    x,y,w,h = cv2.boundingRect(cnts)
    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.drawContours(frame,[hull],-1,(255,255,255),2)
    ##### Show final image ########
    cv2.imshow('Dilation',frame)
    ###############################
    #close the output video by pressing 'ESC'
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()