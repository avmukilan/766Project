from handdetection.Learning import Learning
from handdetection.Capture import Capture

#learning = Learning('data/Skin_NonSkin.txt')
from handdetection.SkinProbabiltyMap import SkinProbabiltyMap
from handdetection.tree import Tree

if __name__ == "__main__":
    learning = Tree()
    cap = Capture(learning)
    cap.startCapture()
    print('Destroying')
    cap.destroy()