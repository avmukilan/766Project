from sklearn.externals import joblib
import cv2
import numpy as np
import datetime

class SPM:

    def __init__(self):
        self.clfRgb = joblib.load('rgbModelPickle.pkl')
        self.clfHsv = joblib.load('hsvModelPickle.pkl')

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
    spm = SPM()
    start = datetime.datetime.now()
    #imageToTest = cv2.imread("Images/josh-hartnett-Poster-thumb.jpg")
    imageToTest = cv2.imread("Images/test.png")
    predict = spm.predict(imageToTest)
    cv2.imwrite("results/finalPredicted.png",predict)
    message = "Time -" + str(datetime.datetime.now() - start)
    print(message)
