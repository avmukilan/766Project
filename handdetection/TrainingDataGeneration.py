import os
from skimage.io import imread
import cv2
import numpy as np

def readData():
    originalDir = "Face_Dataset/Pratheepan_Dataset/FacePhoto/"
    binaryDir = "Face_Dataset/Ground_Truth/GroundT_FacePhoto/"
    output_file = open("data/facedataNRGB.txt",'w')
    hsv_output = open("data/facedataHSV.txt",'w')

    for file in os.listdir(originalDir):
        completeInputFile = originalDir + file
        file = str.replace(file,"jpg","png")
        file = str.replace(file,"jpeg","png")
        binaryImageFile = binaryDir + file
        originalImage = cv2.imread(completeInputFile)
        nrgb = normalizeRGB(originalImage)
        hsv = cv2.cvtColor(originalImage,cv2.COLOR_RGB2HSV)

        binaryImage = cv2.imread(binaryImageFile)
        binaryImage = binaryImage[:,:,0]

        normalisedImageReshape = np.reshape(nrgb, (nrgb.shape[0] * nrgb.shape[1], 3))
        nhsv = np.reshape(hsv, (hsv.shape[0] * hsv.shape[1], 3))
        binaryImageReshape = np.reshape(binaryImage, (binaryImage.shape[0] * binaryImage.shape[1], 1))


        row = normalisedImageReshape.shape[0]
        rgbtrain = np.concatenate((normalisedImageReshape,binaryImageReshape),axis=1)
        hsvtrain = np.concatenate((nhsv,binaryImageReshape),axis=1)
        #hsvtrain.tofile("data/rbg/"+file+".csv",sep=',')
        #rgbtrain.tofile("data/hsv/"+file+".csv",sep=',')
        np.savetxt("data/rbg/"+file+".csv", rgbtrain, delimiter=",",newline='\n',fmt='%3d')
        np.savetxt("data/hsv/"+file+".csv", hsvtrain, delimiter=",",newline='\n',fmt='%3d')

        """for i in range(0,row):
                red = normalisedImageReshape[i,0]
                green = normalisedImageReshape[i,1]
                blue = normalisedImageReshape[i,2]
                outcome = binaryImageReshape[i]
                hred = nhsv[i,0]
                hgreen = nhsv[i,1]
                hblue = nhsv[i,2]

                string1 = str(red) + "\t" + str(green) + "\t" + str(blue) + "\t" + str(outcome)
                string2 = str(hred) + "\t" + str(hgreen) + "\t" + str(hblue) + "\t" + str(outcome)
                output_file.write(string1 + "\n")
                hsv_output.write(string2 + "\n")"""

        print("Done with image "+ file)

def normalizeRGB(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

readData()