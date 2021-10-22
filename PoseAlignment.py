import json
import os
import cv2
import numpy as np
import random


def divideChunks(l, n):
    temp = []
    for i in range(0, len(l), n):
        temp.append(l[i : i + n - 1])
    return temp

def get2PointsDistance(p1, p2):
    return np.sqrt(np.sum(np.power((p1 - p2), 2)))


def getHomographyMatrix(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)
    return H

def isRigidTransformation(transformation):
    transposed = transformation.transpose()
    points = {
        np.matrix([0, 0, 1]),
        np.matrix([1920, 0, 1]),
        np.matrix([1920, 1080, 1]),
        np.matrix([0, 1080, 1]),
    }
    points2 = [np.matmul(x,transposed) for x in points]
    print()

def getRansacHomography(p1, p2, eps = 50):
    # filter zeros
    # p1, p2 = zip(*((point1, point2) for point1, point2 in zip(p1, p2) if point1 != [0,0] and point2 != [0,0]))

    inliersCount = 0
    bestInliersCount = -1
    bestmodel = 0
    for cycle in range(0, 1000):
        # setup
        inliersCount = 0
        # chose random 4 points
        possibleInliersIndexes = random.sample(range(0, len(p1)), 4)
        possibleInliersP1 = []
        possibleInliersP2 = []
        for index in possibleInliersIndexes:
            possibleInliersP1.append(p1[index])
            possibleInliersP2.append(p2[index])

        # calculate homography NOTE: so far i am using cv2 homography (its faster and the matrix loo nicer)
        # homography = getHomographyMatrix(possibleInliersP1, possibleInliersP2)
        # homographyTransposed = homography.transpose()
        homography, status = cv2.findHomography(np.matrix(possibleInliersP1), np.matrix(possibleInliersP2))
        if homography is None:
            continue    
        homographyTransposed = homography.transpose()

        # find how many points fit the model with tolerance eps = 20
        for (point1, point2) in zip(p1, p2):
            point3d = np.matrix([point1[0], point1[1], 1])
            transformed = np.matmul(point3d, homographyTransposed)[0:1, 0:2]
            distance = get2PointsDistance(transformed, point2)
            if distance < eps:
                inliersCount += 1
        if inliersCount > bestInliersCount:
            bestInliersCount = inliersCount
            bestmodel = homography
        # repeat and find best fit

    inliersP1 = []
    inliersP2 = []
    for (point1, point2) in zip(p1, p2):
        point3d = np.matrix([point1[0], point1[1], 1])
        transformed = np.matmul(point3d, bestmodel.transpose())[0:1, 0:2]
        distance = get2PointsDistance(transformed, point2)
        if distance < eps:
            inliersP1.append(point1)
            inliersP2.append(point2)
    
    if len(inliersP1) > 4:
        finalHomography, status = cv2.findHomography(np.matrix(inliersP1), np.matrix(inliersP2))
        if finalHomography is not None:
            bestmodel = finalHomography

    additionalInfo = {
        "InliersCount" : bestInliersCount,
    }
    return bestmodel, additionalInfo

###### MAIN ######

main_path = ".\data\\Cobra"
path_to_json = main_path + "\\Jsons\\"
path_to_images = main_path + "\\Processed\\"
path_to_aligned = main_path + "\\Aligned\\"

# Import Json files, pos_json = position JSON
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith(".json")]
print("Found: ", len(json_files), "json keypoint frame files")
images = []
for file in json_files:
    keyPoints = json.load(open(path_to_json + file))["people"][0]["pose_keypoints_2d"]
    points = divideChunks(keyPoints, 3)
    image = {
        "path": file,
        "points": points[:15],
    }
    images.append(image)


firstname = images[0]["path"][:-15]
firstImg = cv2.imread(f"{path_to_images}\\{firstname}_rendered.png")
# vezmu první image a všechny podle něj zarovnám
for image in images[1:]:
    homography, homographyInfo = getRansacHomography(image["points"], images[0]["points"])
    print(image["path"])
    print(homographyInfo['InliersCount'])
    imageName = image["path"][:-15]
    img = cv2.imread(f"{path_to_images}\\{imageName}_rendered.png")
    out = cv2.warpPerspective(img, homography, (1920, 1080))


    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (5, 25)
    fontScale = 1
    color = (255, 0, 0)
    thickness = 1
    image = cv2.putText(out, f"Inlier count: {homographyInfo['InliersCount']}",
     org, font, fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(f"{path_to_aligned}{imageName}_aligned.png", out)

    # cv2.imshow("image window", firstImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow("image window", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()