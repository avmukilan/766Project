
import numpy as np
import cv2

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def ReadData():
    #Data in format [B G R Label] from
    data = np.genfromtxt('data/Skin_NonSkin.txt', dtype=np.int32)

    labels= data[:,3]
    data= data[:,0:3]

    return data, labels

def BGR2HSV(bgr):
    bgr= np.reshape(bgr,(bgr.shape[0],1,3))
    hsv= cv2.cvtColor(np.uint8(bgr), cv2.COLOR_BGR2HSV)
    hsv= np.reshape(hsv,(hsv.shape[0],3))

    return hsv


def TrainTree(data, labels, flUseHSVColorspace):
    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    trainData, testData, trainLabels, testLabels = train_test_split(data, labels, test_size=0.20, random_state=42)

    print(trainData.shape)
    print(trainLabels.shape)
    print(testData.shape)
    print(testLabels.shape)

    clf = GaussianNB()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3, 1), random_state=1)
    clf = clf.fit(data, labels)
    #print(clf.feature_importances_)
    #print(clf.score(testData, testLabels))

    return clf

def ApplyToImage(path, flUseHSVColorspace):
    data, labels= ReadData()
    clf= TrainTree(data, labels, flUseHSVColorspace)

    img= cv2.imread(path)
    cv2.imshow("Image",img)
    #print(img.shape)
    data= np.reshape(img,(img.shape[0]*img.shape[1],3))
    #print(data.shape)

    if(flUseHSVColorspace):
        data= BGR2HSV(data)

    predictedLabels= clf.predict(data)

    imgLabels= np.reshape(predictedLabels,(img.shape[0],img.shape[1],1))
    image = (-(imgLabels - 1) + 1) * 255
    if (flUseHSVColorspace):
        cv2.imwrite('results/result_HSV.png', (image))# from [1 2] to [0 255]
    else:
        cv2.imwrite('results/result_RGB.png', (image))

    return image


#---------------------------------------------
image1 = ApplyToImage("Images/josh-hartnett-Poster-thumb.jpg", True)
image2 = ApplyToImage("Images/josh-hartnett-Poster-thumb.jpg", False)
bitwise_and = cv2.bitwise_and(image1, image2)
imageToProcess = (-(bitwise_and-1)+1)*255
imageToWrite = cv2.bitwise_not(imageToProcess)
cv2.imwrite('results/result.png',imageToWrite)
