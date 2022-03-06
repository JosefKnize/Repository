import PoseAlignment as PA
import os
###### MAIN ######
if __name__ == "__main__":
    datasetPath = ".\..\data03\\"
    posePaths = os.listdir(datasetPath)


    # for posePath in posePaths:
    #     if posePath != "MachineData":
    #         PA.AlignImagesInFolder(datasetPath, posePath)

    # for posePath in posePaths:
    #     if posePath == "Mountain":
    #         PA.AlignImagesInFolder(datasetPath, posePath)

    # for posePath in posePaths:
    #     if posePath == "Plank":
    #         PA.AlignImagesInFolder(datasetPath, posePath)

    poses = ["WarriorThree", "Goddes", "FullBow", "LowSideLunge", "Plank", "Pyramid", "Triangle"]
    for posePath in poses:
        PA.AlignImagesInFolder(datasetPath, posePath)

            