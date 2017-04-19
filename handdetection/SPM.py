from sklearn.externals import joblib
import cv2
import numpy as np
import datetime

class SPM:

    def __init__(self):
        self.clfHsv = joblib.load('SPMPickle.pkl')

    def predict(self,image):
        blur = cv2.blur(image,(3,3))
        hsvd = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

        nrgb = self.normalizeRGB(image)
        cv2.imwrite("results/nrgb.png",nrgb)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        cv2.imwrite("results/hsv.png",hsv)
        ndarray = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        hsvndArrap = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))
        predict1 = self.clfRgb.predict(ndarray)
        cv2.imwrite("results/rgbPredict.png",np.reshape(predict1,(nrgb.shape[0] ,nrgb.shape[1])))
        predict2 = self.clfHsv.predict(hsvndArrap)
        cv2.imwrite("results/hsvPredict.png",np.reshape(predict2,(nrgb.shape[0] ,nrgb.shape[1])))
        predict = cv2.bitwise_or(predict1, predict2)
        reshape = np.reshape(predict,(nrgb.shape[0] ,nrgb.shape[1]))
        return reshape


    def predict2(self,image):
        blur = cv2.blur(image,(3,3))
        nrgb = self.normalizeRGB(image)
        cv2.imwrite("results/nrgb.png",nrgb)
        hsv = cv2.cvtColor(blur,cv2.COLOR_RGB2HSV)
        cv2.imwrite("results/hsv.png",hsv)
        ndarray = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        hsvndArrap = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))
        predict1 = self.clfRgb.predict(ndarray)
        cv2.imwrite("results/rgbPredict.png",np.reshape(predict1,(nrgb.shape[0] ,nrgb.shape[1])))
        predict2 = self.clfHsv.predict(hsvndArrap)
        cv2.imwrite("results/hsvPredict.png",np.reshape(predict2,(nrgb.shape[0] ,nrgb.shape[1])))
        predict = cv2.bitwise_or(predict1, predict2)
        reshape = np.reshape(predict,(nrgb.shape[0] ,nrgb.shape[1]))
        finalResult = cv2.merge((reshape,reshape,reshape))
        cv2.imwrite("results/reshape.png",finalResult)
        return finalResult

    def predict3(self,image):
        blur = cv2.blur(image,(3,3))
        hsv = cv2.cvtColor(blur,cv2.COLOR_RGB2HSV)
        hsvreshape = np.reshape(hsv,(hsv.shape[0] * hsv.shape[1], 3))
        predict = self.clfHsv.predict(hsvreshape)
        predictReshape = np.reshape(predict,(image.shape[0] ,image.shape[1]))
        cv2.imwrite("results/reshape.png",predictReshape)
        return predictReshape

def normalizeRGB(self,img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

if __name__ == "__main__":
    spm = SPM()
    start = datetime.datetime.now()
    #imageToTest = cv2.imread("Images/josh-hartnett-Poster-thumb.jpg")
    #imageToTest = cv2.imread("Images/test.png")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  300)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
    ret, imageToTest = cap.read()
    cv2.imwrite("results/image.png",imageToTest)
    predict = spm.predict3(imageToTest)

    cv2.imwrite("results/finalPredicted.png",predict)
    message = "Time -" + str(datetime.datetime.now() - start)
    print(message)
