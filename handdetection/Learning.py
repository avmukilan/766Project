import numpy as np
import cv2
from sklearn.naive_bayes import GaussianNB

class Learning:

    def __init__(self,trainFileLocation):
        self.trainFile = trainFileLocation
        self.initialize()

    def initialize(self):
        self.readTrainingFile(False)
        self.readTrainingFile(True)
        self.Train()

    def readTrainingFile(self,isHsv):
        print("Reading Data")
        dataFromFile = np.genfromtxt(self.trainFile, dtype=np.int)
        if(isHsv):
            data = dataFromFile[:,0:3]
            self.hsvtrainingData = self.BGR2HSV(data)
            self.hsvcategory= dataFromFile[:,3]
        else:
            self.category= dataFromFile[:,3]
            self.trainingData= dataFromFile[:,0:3]


    def Train(self):
        #self.model = cv2.ml.NormalBayesClassifier_create()
        self.model = GaussianNB()
        self.hsvmodel = GaussianNB()
        #return self.model.train(self.trainingData,cv2.ml.ROW_SAMPLE,self.category)
        self.model.fit(self.trainingData , self.category)
        self.hsvmodel.fit(self.hsvtrainingData, self.hsvcategory)


    def predict(self,img, onlyhsv=True):
        #hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        #ret, resp = self.model.predict(image)
        #predictedLabels = resp.ravel()
        #imgLabels= np.reshape(predictedLabels,(image.shape[0],image.shape[1],1))
        #image = (-(imgLabels - 1) + 1) * 255
        #return resp.ravel()
        if(onlyhsv):
            data = np.reshape(img,(img.shape[0]*img.shape[1],3))
            hsv = self.BGR2HSV(data)
            predictedLabelsHsv= self.hsvmodel.predict(hsv)
            imgLabelsHsv= np.reshape(predictedLabelsHsv,(img.shape[0],img.shape[1],1))
            imageHsv = (-(imgLabelsHsv - 1) + 1) * 255
            return imageHsv
        else:
            data = np.reshape(img,(img.shape[0]*img.shape[1],3))
            hsv = self.BGR2HSV(data)
            predictedLabels= self.model.predict(data)
            imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
            image = (-(imgLabels - 1) + 1) * 255
            predictedLabelsHsv= self.hsvmodel.predict(hsv)
            imgLabelsHsv= np.reshape(predictedLabelsHsv,(img.shape[0],img.shape[1],1))
            imageHsv = (-(imgLabelsHsv - 1) + 1) * 255
            bitwise_and = cv2.bitwise_and(image, imageHsv)
            imageToProcess = (-(bitwise_and-1)+1)*255
            return cv2.bitwise_not(imageToProcess)

    def BGR2HSV(self,bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))
        return hsv

#learning = Learning('data/Skin_NonSkin.txt')
#image = cv2.imread('Images/josh-hartnett-Poster-thumb.jpg')
#cap = cv2.VideoCapture(0)
#ret, frame = cap.read()
#predict = learning.predict(frame)
#predict2 = learning.predict(frame,False)
#predict3 = learning.predict(image,False)
#cv2.imshow('Result',predict)#cv2.imwrite("results/predict.png", predict)
#cv2.imwrite("results/predict2.png", predict2)
#cv2.imwrite("results/predict3.png", predict3)
#cv2.imwrite("results/image.png", frame)




