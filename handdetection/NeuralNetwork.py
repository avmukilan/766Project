import numpy as np
import cv2
import os
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib

def ReadData():
    allRgbData = np.empty([0, 4], dtype=int)
    allhsvData = np.empty([0, 4], dtype=int)
    originalDir = "data/rbg/"
    binaryDir = "data/hsv/"
    i = 0
    for file in os.listdir(originalDir):
        if i >= 5:
          break
        rgbFile = originalDir + file
        hsvFile = binaryDir + file
        rgbData = np.genfromtxt(rgbFile, dtype=np.int, delimiter=",")
        hsvData = np.genfromtxt(hsvFile, dtype=np.int, delimiter=",")
        allRgbData = np.concatenate((allRgbData,rgbData),axis=0)
        allhsvData = np.concatenate((allhsvData,hsvData),axis=0)
        i += 1
        print("Done " + str(i))
    allLabels = allRgbData[:,3]
    allRgbData = allRgbData[:,0:3]
    allhsvData = allhsvData[:,0:3]
    return allLabels, allRgbData, allhsvData

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))
    return hsv


def TrainTree(allLabels, allRgbData, allhsvData):
    print("Started Training")
    clfRgb = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 1), random_state=1)
    clfRgb = clfRgb.fit(allRgbData, allLabels)
    print("Finished Training Rgb")
    clfHsv = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 1), random_state=1)
    clfHsv = clfHsv.fit(allhsvData, allLabels)
    print("Finished Training Hsv")
    hsvModel = joblib.dump(clfHsv,'hsvModelPickle.pkl')
    return clfRgb, clfHsv

def ApplyToImage(path):
    allLabels, allRgbData, allhsvData= ReadData()
    clf= TrainTree(allLabels, allRgbData, allhsvData)
    print("Done Training")
    img= cv2.imread(path)
    cv2.imshow("Image",img)
    dataRgb = np.reshape(img,(img.shape[0]*img.shape[1],3))
    dataHsv = BGR2HSV(dataRgb)
    predictedRgb = clf[0].predict(dataRgb)
    predictedHsv = clf[1].predict(dataHsv)

    imgLabels= np.reshape(predictedRgb,(img.shape[0],img.shape[1],1))
    image = (-(imgLabels - 1) + 1) * 255

    imgLabels2= np.reshape(predictedHsv,(img.shape[0],img.shape[1],1))
    image2 = (-(imgLabels2 - 1) + 1) * 255

    cv2.imwrite('results/result_HSV.png', (image))# from [1 2] to [0 255]
    cv2.imwrite('results/result_RGB.png', (image2))

    return image,image2

#---------------------------------------------
image = ApplyToImage("Images/test.png")
bitwise_and = cv2.bitwise_and(image[0], image[1])
imageToProcess = (-(bitwise_and-1)+1)*255
imageToWrite = cv2.bitwise_not(imageToProcess)
cv2.imwrite('results/result.png',imageToWrite)
