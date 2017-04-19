import pickle
from handdetection.SkinProbabilityMapClassifier import SPMClassifier
import cv2

output = open('SPMPickle.pkl', 'wb')
rgbModel = pickle.dump(SPMClassifier(),output)
output.close()

file = open('SPMPickle.pkl','rb')
classifier = pickle.load(file)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  300)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
#ret, imageToTest = cap.read()
imageToTest = cv2.imread("Images/test.png")
cv2.imwrite("results/image.png",imageToTest)
predict = classifier.predict(imageToTest)
cv2.imwrite("results/OutputTest.png",predict)