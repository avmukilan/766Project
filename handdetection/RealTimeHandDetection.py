from handdetection.Capture import Capture
from sklearn.externals import joblib

from handdetection.SPM import SPM
import pickle

if __name__ == "__main__":
    file = open('SPMPickle.pkl','rb')
    #file = open('hsvModelPickle.pkl','rb')
    classifier = pickle.load(file)
    #classifier = joblib.load(file)
    cap = Capture(classifier)
    cap.startCapture()
    print('Destroying')
    cap.destroy()