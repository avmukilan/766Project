import cv2
import numpy as np
import os
import datetime
class SkinProbabiltyMap:

    hist_bin = np.array((100,100))
    low_range = np.array((0.2,0.3))
    high_range = np.array((0.4,0.5))
    skin_histogram = np.zeros((100,100),np.float32)
    non_skin_histogram = np.zeros((100,100),np.float32)
    hsv_skin_histogram = np.zeros((100,100),np.float32)
    hsv_non_skin_histogram = np.zeros((100,100),np.float32)

    def __init__(self):
        originalDir = "Face_Dataset/Pratheepan_Dataset/FacePhoto/"
        binaryDir = "Face_Dataset/Ground_Truth/GroundT_FacePhoto/"
        for file in os.listdir(originalDir):
            completeInputFile = originalDir + file
            file = str.replace(file,"jpg","png")
            file = str.replace(file,"jpeg","png")
            binaryImageFile = binaryDir + file
            trainImage = cv2.imread(completeInputFile)
            mask = cv2.imread(binaryImageFile)[:,:,0]
            self.fit(trainImage,mask)

        self.lookupsh = dict()
        for i in range(0,self.skin_histogram.shape[0]):
            d = dict()
            for j in range(0,self.skin_histogram.shape[1]):
                d[j] = self.skin_histogram[i,j]
            self.lookupsh[i] = d

        self.lookupnsh = dict()
        for i in range(0,self.non_skin_histogram.shape[0]):
            d = dict()
            for j in range(0,self.non_skin_histogram.shape[1]):
                d[j] = self.non_skin_histogram[i,j]
            self.lookupnsh[i] = d

        self.lookuphsh = dict()
        for i in range(0,self.hsv_skin_histogram.shape[0]):
            d = dict()
            for j in range(0,self.hsv_skin_histogram.shape[1]):
                d[j] = self.hsv_skin_histogram[i,j]
            self.lookuphsh[i] = d

        self.lookupnhsh = dict()
        for i in range(0,self.hsv_non_skin_histogram.shape[0]):
            d = dict()
            for j in range(0,self.hsv_non_skin_histogram.shape[1]):
                d[j] = self.hsv_non_skin_histogram[i,j]
            self.lookupnhsh[i] = d


    def bootstrap(self,image):
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        normalisedRGB = self.getNormalisedRGB(image)
        mask_hsv = cv2.inRange(hsv,np.array([2,50,90]),np.array([25,173,255]))
        mask_nrgb = cv2.inRange(normalisedRGB,np.array([0.0,0.28,0.36]),np.array([1.0,0.363,0.465]))
        outputMask = cv2.bitwise_and(mask_hsv, mask_nrgb)
        return outputMask

    def fit(self,image,mask):
        notMask = cv2.bitwise_not(mask)
        nrgb = self.normalizeRGB(image)
        skin_histogram = self.calc_rgb_hist(nrgb, mask)
        non_skin_histogram = self.calc_rgb_hist(nrgb,notMask)
        self.skin_histogram = cv2.add(self.skin_histogram,skin_histogram)
        self.non_skin_histogram = cv2.add(self.non_skin_histogram, non_skin_histogram)

        nrgb = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        hsv_skin_histogram = self.calc_rgb_hist(nrgb, mask)
        hsv_non_skin_histogram = self.calc_rgb_hist(nrgb,notMask)
        self.hsv_skin_histogram = cv2.add(self.hsv_skin_histogram,hsv_skin_histogram)
        self.hsv_non_skin_histogram = cv2.add(self.hsv_non_skin_histogram, hsv_non_skin_histogram)
        
        #skin_pixels = cv2.countNonZero(mask)
        #non_skin_pixels = cv2.countNonZero(notMask)
        #for ubin in range(0,100):
        #    for vbin in range(0,100):
        #        if self.skin_histogram[ubin,vbin] > 0.0:
        #            self.skin_histogram[ubin,vbin] /= skin_pixels
        #        if self.non_skin_histogram[ubin,vbin] > 0.0:
        #            self.non_skin_histogram[ubin,vbin] /= non_skin_pixels
        #self.visualiseHist(self.skin_histogram,self.hist_bin,"skin hist")
        #self.visualiseHist(self.non_skin_histogram,self.hist_bin,"non skin hist")

    """def predict(self,image):
        result = np.zeros((image.shape[0]*image.shape[1]),np.uint8)
        hsv_result = np.zeros((image.shape[0]*image.shape[1]),np.uint8)

        nrgb = self.normalizeRGB(image)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        ndarray = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        hsvndArrap = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))
        start = datetime.datetime.now()

        for i in range(0,ndarray.shape[0]):
            rbin = int( ndarray[i,0] / 256 * 100)
            gbin = int(ndarray[i,1] / 256 * 100)
            skin_hist_val = self.skin_histogram[rbin,gbin]
            if skin_hist_val > 0:
                non_skin_hist_val = self.non_skin_histogram[rbin,gbin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        result[i] = 255
                    else:
                        result[i] = 0
                else:
                    result[i] = 0
            else:
                result[i] = 0

            hsvRbin = int( hsvndArrap[i,0] / 256 * 100)
            hsvGBin = int( hsvndArrap[i,1] / 256 * 100)
            skin_hist_val = self.hsv_skin_histogram[hsvRbin,hsvGBin]
            if skin_hist_val > 0:
                non_skin_hist_val = self.hsv_non_skin_histogram[hsvRbin,hsvGBin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        hsv_result[i] = 255
                    else:
                        hsv_result[i] = 0
                else:
                    hsv_result[i] = 0
            else:
                hsv_result[i] = 0

        message = "Iteration -" + str(datetime.datetime.now() - start)
        print(message)


        final = cv2.bitwise_and(hsv_result, result)
        reshape = np.reshape(final, (image.shape[0], image.shape[1]))
        return reshape"""

    def predict(self,image):
        size = image.shape[0]*image.shape[1]
        nrgb = self.normalizeRGB(image)
        hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        ndarray = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        hsvndArrap = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))

        #start = datetime.datetime.now()
        #prediction = self.predictLoop(ndarray,hsvndArrap,size)
        #message = "Iteration -" + str(datetime.datetime.now() - start)
        #print(message)

        #start = datetime.datetime.now()
        #for i in range(0,size):
        #    pass
        #message = "Iteration2 -" + str(datetime.datetime.now() - start)
        #print(message)

        start = datetime.datetime.now()
        final = self.predictLoop3(ndarray,hsvndArrap,size)
        message = "Iteration3 -" + str(datetime.datetime.now() - start)
        print(message)
        #final = cv2.bitwise_and(prediction[0], prediction[1])
        reshape = np.reshape(final, (image.shape[0], image.shape[1]))
        return reshape

    def predictLoop2(self,ndarray,hsvndArrap,size):
        hsv_result = np.zeros(size,np.uint8)
        for i in range(0,ndarray.shape[0]):
            hsvRbin = int( hsvndArrap[i,0] / 256 * 100)
            hsvGBin = int( hsvndArrap[i,1] / 256 * 100)
            skin_hist_val = self.hsv_skin_histogram[hsvRbin,hsvGBin]
            if skin_hist_val > 0:
                non_skin_hist_val = self.hsv_non_skin_histogram[hsvRbin,hsvGBin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        hsv_result[i] = 255
                    else:
                        hsv_result[i] = 0
                else:
                    hsv_result[i] = 0
            else:
                hsv_result[i] = 0

        return hsv_result

    def predictLoop3(self,ndarray,hsvndArrap,size):
        hsv_result = np.zeros(size,np.uint8)
        for i in range(0,ndarray.shape[0]):
            hsvRbin = int( hsvndArrap[i,0] / 256 * 100)
            hsvGBin = int( hsvndArrap[i,1] / 256 * 100)
            var = self.lookuphsh[hsvRbin]
            skin_hist_val = var[hsvGBin]
            if skin_hist_val > 0:
                var2 = self.lookupnhsh[hsvRbin]
                non_skin_hist_val = var2[hsvGBin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        hsv_result[i] = 255
                    else:
                        hsv_result[i] = 0
                else:
                    hsv_result[i] = 0
            else:
                hsv_result[i] = 0

        return hsv_result

    def predictLoop(self,ndarray,hsvndArrap,size):
        result = np.zeros(size,np.uint8)
        hsv_result = np.zeros(size,np.uint8)
        for i in range(0,ndarray.shape[0]):
            rbin = int( ndarray[i,0] / 256 * 100)
            gbin = int(ndarray[i,1] / 256 * 100)
            skin_hist_val = self.skin_histogram[rbin,gbin]
            if skin_hist_val > 0:
                non_skin_hist_val = self.non_skin_histogram[rbin,gbin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        result[i] = 255
                    else:
                        result[i] = 0
                else:
                    result[i] = 0
            else:
                result[i] = 0

            hsvRbin = int( hsvndArrap[i,0] / 256 * 100)
            hsvGBin = int( hsvndArrap[i,1] / 256 * 100)
            skin_hist_val = self.hsv_skin_histogram[hsvRbin,hsvGBin]
            if skin_hist_val > 0:
                non_skin_hist_val = self.hsv_non_skin_histogram[hsvRbin,hsvGBin]
                if (non_skin_hist_val > 0):
                    if((skin_hist_val / non_skin_hist_val) > 0.8):
                        hsv_result[i] = 255
                    else:
                        hsv_result[i] = 0
                else:
                    hsv_result[i] = 0
            else:
                hsv_result[i] = 0
        return result,hsv_result

    def normalizeRGB(self,img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output

    def visualiseHist(self,hist,bins,histwinname):
        histImg = np.zeros((100,100),np.uint8)
        maxval = cv2.minMaxLoc(hist)
        print(maxval)
        for ubin in range(0,100):
            for vbin in range(0,100):
                binVal = hist[ubin,vbin]
                intensity = int(binVal*255/maxval[1])
                cv2.rectangle(histImg,(ubin,vbin),(ubin+1,vbin+1),(intensity,256,256),cv2.FILLED)

        #returnHist = cv2.cvtColor(histImg,cv2.COLOR_HSV2BGR)
        #cv2.imwrite("results/"+histwinname+".png",histImg)
        return histImg

    def getNormalisedRGB(self,image):
        image = np.float32(image)
        image = np.float32(image)
        height, width, channels = image.shape
        norm = np.zeros((height,width,channels),np.float32)
        b = image[:,:,0]
        g = image[:,:,1]
        r = image[:,:,2]
        sum = b + g + r
        norm[:,:,0] = b/sum
        norm[:,:,1] = g/sum
        norm[:,:,2] = r/sum
        return norm

    def calc_rgb_hist(self,nrgb,mask):
        return cv2.calcHist([nrgb], [0,1], mask, [100,100], [0,255,0,255],True,False)

if __name__ == "__main__":
    spm = SkinProbabiltyMap()
    start = datetime.datetime.now()
    imageToTest = cv2.imread("Images/josh-hartnett-Poster-thumb.jpg")
    predict = spm.predict(imageToTest)
    cv2.imwrite("results/finalPredicted.png",predict)
    message = "Time -" + str(datetime.datetime.now() - start)
    print(message)
