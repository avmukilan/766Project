import numpy as np
import cv2
import os
import datetime
import pickle
from sklearn.externals import joblib

from sklearn.naive_bayes import GaussianNB

class GenerateSPMModel:

    def trainAndPickle(self):
        self.allRgbData = np.empty([0, 4], dtype=int)
        self.allhsvData = np.empty([0, 4], dtype=int)
        originalDir = "data/rbg/"
        binaryDir = "data/hsv/"
        i = 0
        for file in os.listdir(originalDir):
            #if i >= 5:
            #    break
            rgbFile = originalDir + file
            hsvFile = binaryDir + file
            rgbData = np.genfromtxt(rgbFile, dtype=np.int, delimiter=",")
            hsvData = np.genfromtxt(hsvFile, dtype=np.int, delimiter=",")
            self.allRgbData = np.concatenate((self.allRgbData,rgbData),axis=0)
            self.allhsvData = np.concatenate((self.allhsvData,hsvData),axis=0)
            i += 1
        self.allLabels = self.allRgbData[:,3]
        self.allRgbData = self.allRgbData[:,0:3]
        self.allhsvData = self.allhsvData[:,0:3]

        print("Starting Training")
        start = datetime.datetime.now()
        models = self.fit()
        rgbModel = joblib.dump(models[0],'rgbModelPickle.pkl')
        hsvModel = joblib.dump(models[1],'hsvModelPickle.pkl')
        print(rgbModel)
        print(hsvModel)
        message = "Train Time -" + str(datetime.datetime.now() - start)
        print(message)

    def trainAndPickle2(self):
        file = "data/Skin_NonSkin.txt"
        data = np.genfromtxt(file, dtype=np.int)
        self.allRgbData = data[:,0:3]
        self.allLabels = data[:,3]
        self.allhsvData = self.BGR2HSV(self.allRgbData)
        print("Starting Training")
        start = datetime.datetime.now()
        models = self.fit()
        rgbModel = joblib.dump(models[0],'rgbModelPickle.pkl')
        hsvModel = joblib.dump(models[1],'hsvModelPickle.pkl')
        print(rgbModel)
        print(hsvModel)
        message = "Train Time -" + str(datetime.datetime.now() - start)
        print(message)

    def trainAndPickle3(self):
        self.allhsvData = np.empty([0, 4], dtype=int)
        binaryDir = "data/realtimehsv/"
        i = 0
        for file in os.listdir(binaryDir):
            hsvFile = binaryDir + file
            hsvData = np.genfromtxt(hsvFile, dtype=np.int, delimiter=",")
            self.allhsvData = np.concatenate((self.allhsvData,hsvData),axis=0)
            i += 1
            print("Done-"+ file)
        self.allLabels = self.allhsvData[:,3]
        self.allhsvData = self.allhsvData[:,0:3]
        print(self.allhsvData.shape)
        print("Starting Training")
        start = datetime.datetime.now()
        models = self.fit()
        hsvModel = joblib.dump(models,'SPMPickle.pkl')
        print(hsvModel)
        message = "Train Time -" + str(datetime.datetime.now() - start)
        print(message)

    def BGR2HSV(self,bgr):
        bgr= np.reshape(bgr,(bgr.shape[0],1,3))
        hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
        hsv= np.reshape(hsv,(hsv.shape[0],3))

        return hsv

    def fit(self):
        clfHsv = GaussianNB()
        clfHsv = clfHsv.fit(self.allhsvData, self.allLabels)
        return clfHsv

    def predict(self,image):
        nrgb = self.normalizeRGB(image)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        ndarray = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        hsvndArrap = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))
        predict1 = self.clfRgb.predict(ndarray)
        predict2 = self.clfHsv.predict(hsvndArrap)
        predict = cv2.bitwise_and(predict1, predict2)
        return np.reshape(predict,(nrgb.shape[0] ,nrgb.shape[1]))

    def normalizeRGB(self,img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


if __name__ == "__main__":
    spm = GenerateSPMModel()
    spm.trainAndPickle3()
    #start = datetime.datetime.now()
    #imageToTest = cv2.imread("Images/josh-hartnett-Poster-thumb.jpg")
    #imageToTest = cv2.imread("Images/test.png")
    #predict = spm.predict(imageToTest)
    #cv2.imwrite("results/finalPredicted.png",predict)
    #message = "Time -" + str(datetime.datetime.now() - start)
    #print(message)
