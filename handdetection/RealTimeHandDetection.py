from handdetection.Capture import Capture

from handdetection.SPM import SPM
import pickle

if __name__ == "__main__":
    file = open('SPMPickle.pkl','rb')
    classifier = pickle.load(file)
    cap = Capture(classifier)
    cap.startCapture()
    print('Destroying')
    cap.destroy()