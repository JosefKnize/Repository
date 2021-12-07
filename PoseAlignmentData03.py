import PoseAlignment as PA
import os
###### MAIN ######
if __name__ == "__main__":
    datasetPath = ".\..\data03\\"
    posePaths = os.listdir(datasetPath)


    for posePath in posePaths:
        if posePath != "MachineData":
            PA.AlignImagesInFolder(datasetPath, posePath)