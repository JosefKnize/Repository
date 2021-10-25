import json
import os
import cv2
import numpy as np
import random

from numpy.core.numeric import cross


def divideChunks(l, n):
    temp = []
    for i in range(0, len(l), n):
        temp.append(l[i : i + n - 1])
    return temp

def get2PointsDistance(p1, p2):
    return np.sqrt(np.sum(np.power((p1 - p2), 2)))

def distancesBetweenPoints(points1, points2):
    return np.sqrt(np.sum(np.power((points1 - points2), 2), axis=1))

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
    points = np.array([[0, 0, 1], [1920, 0, 1], [1920, 1080, 1], [0, 1080, 1],])
    points = np.matmul(points, transformation.transpose())
    vectors = points - np.roll(points, 1, axis=0)
    crossed = np.cross(vectors, np.roll(vectors, 1, axis=0))
    return np.all(crossed[:,2] < 0) or np.all(crossed[:,2] > 0)

def getRansacHomography(p1, p2, eps = 50):
    # rearange points in [x1, y1, 1, x2, y2, 1] format 
    p1 = np.pad(p1, ((0,0),(0,1)), mode='constant', constant_values=1)
    p2 = np.pad(p2, ((0,0),(0,1)), mode='constant', constant_values=1)
    stackedPoints = np.hstack((p1, p2))
    # filter points that openPose didnt find
    filtered  = stackedPoints[np.all(stackedPoints != 0, axis=1)] 

    # ransac
    bestInliersCount = -1
    bestmodel = 0
    for cycle in range(0, 1000):
        # chose random 4 rows
        InlierIndices = np.random.choice(filtered.shape[0], size=4, replace=False)
        PossibleInliers = filtered[InlierIndices, :]

        # calculate model
        homography, status = cv2.findHomography(PossibleInliers[:, [0,1]], PossibleInliers[:, [3,4]])
        if homography is None:
            continue    
        homographyTransposed = homography.transpose()

        # filter degenerative homoraphy -- doesn't seem to do anything
        if not isRigidTransformation(homography):
            continue

        # find how many points fit the model
        inliersCount = 0
        transformed = np.matmul(filtered[:, [0,1,2]], homographyTransposed)

        for transformedPoint, referencePoint in zip(transformed, filtered[:, [3, 4, 5]]):
            if get2PointsDistance(transformedPoint, referencePoint) < eps:
                inliersCount += 1
        if inliersCount > bestInliersCount:
            bestInliersCount = inliersCount
            bestmodel = homography

        # inliersCount = 0
        # # find how many points fit the model
        # for (point1, point2) in zip(p1, p2):
        #     point3d = np.matrix([point1[0], point1[1], 1])
        #     transformed = np.matmul(point3d, homographyTransposed)[0:1, 0:2]
        #     distance = get2PointsDistance(transformed, point2)
        #     if distance < eps:
        #         inliersCount += 1
        # if inliersCount > bestInliersCount:
        #     bestInliersCount = inliersCount
        #     bestmodel = homography
        # repeat and find best fit

    transformed = np.matmul(filtered[:, [0,1,2]], bestmodel.transpose())
    inliers = filtered[distancesBetweenPoints(transformed, filtered[:, [3,4,5]]) < eps, :]
    
    if inliers.shape[0] > 4:
        finalHomography, status = cv2.findHomography(inliers[:,[0,1]], inliers[:,[3,4]])
        if finalHomography is not None:
            bestmodel = finalHomography

    additionalInfo = {
        "InliersCount" : bestInliersCount,
    }
    return bestmodel, additionalInfo

###### MAIN ######

# a = np.array([[1920,1080,1], [-1920,0,1], [1920,-1080,1], [-1920,0,1]])
# b = np.roll(a,-1,axis=0)
# c = np.cross(a,b)

main_path = ".\..\data\\Cobra"
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