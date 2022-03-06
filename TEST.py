
from importlib.resources import path
import os
if __name__ == "__main__":
    Count = 0
    datasetPath = ".\..\Data\\"
    posePaths = ["Cobra", "DownwardDog", "Child", "Triangle", "WarriorOne", "WarriorTwo", "WarriorTwoBack" ]
    for posePath in posePaths:
        path = os.path.join(datasetPath, posePath)
        path = os.path.join(path, "Images")
        leng = len(os.listdir(path))
        Count += leng * (leng - 1)

    print(Count)

    datasetPath = ".\..\data03\\"
    posePaths = os.listdir(datasetPath)
    for posePath in posePaths:
        if posePath != "MachineData":
            path = os.path.join(datasetPath, posePath)
            path = os.path.join(path, "Images")
            leng = len(os.listdir(path))
            Count += leng * (leng - 1)

    print(Count)
