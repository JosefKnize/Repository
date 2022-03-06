import PoseAlignment as PA

###### MAIN ######
if __name__ == "__main__":
    datasetPath = ".\..\Data\\"
    posePaths = ["Cobra", "DownwardDog", "Child", "Triangle", "WarriorOne", "WarriorTwo", "WarriorTwoBack" ]
    # posePaths = ["WarriorTwo" ]
    for posePath in posePaths:
        PA.AlignImagesInFolder(datasetPath, posePath) 