from asyncio.windows_events import NULL
import json
from math import fabs
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


# def getHomographyMatrix(p1, p2):
#     A = []
#     for i in range(0, len(p1)):
#         x, y = p1[i][0], p1[i][1]
#         u, v = p2[i][0], p2[i][1]
#         A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
#         A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
#     A = np.asarray(A)
#     U, S, Vh = np.linalg.svd(A)
#     L = Vh[-1, :] / Vh[-1, -1]
#     H = L.reshape(3, 3)
#     return H


def isRigidTransformation(transformation):
    points = np.array(
        [
            [0, 0, 1],
            [1920, 0, 1],
            [1920, 1080, 1],
            [0, 1080, 1],
        ]
    )
    points = np.matmul(points, transformation.transpose())
    vectors = points - np.roll(points, 1, axis=0)
    crossed = np.cross(vectors, np.roll(vectors, 1, axis=0))
    return np.all(crossed[:, 2] < 0) or np.all(crossed[:, 2] > 0)

def IsTransformExtreme(transform):
    # keep aspect ratio
    if abs(abs(transform[0,0]) - transform[1,1]) > 0.3:
        return True
    # keep shear low
    if abs(transform[1,0] + transform[0,1]) > 0.3:
        return True
    # keep rotation low
    if abs(transform[1,0]) > 0.5 or abs(transform[0,1]) > 0.3:
        return True
    # dont flip upside down
    if transform[1,1] < 0:
        return True
    # dont flip around y axis
    if transform[0,0] < 0:
        return True
    return False

def calculateAffineTansformation(p1, p2, eps):
    stackedPoints = np.hstack((p1, p2))

    # filter points that openPose didnt find + convert to float32
    filtered = np.float32(stackedPoints[np.all(stackedPoints != 0, axis=1)])
    if filtered.shape[0] < 3:
        return np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), {"InliersCount": -1}

    # ransac
    bestInliersCount = -1
    bestmodel = None
    for cycle in range(0, 1000):
        # chose random 3 pairs of points
        InlierIndices = np.random.choice(filtered.shape[0], size=3, replace=False)
        PossibleInliers = filtered[InlierIndices, :]

        # calculate model
        homography = cv2.getAffineTransform(PossibleInliers[:, [0, 1]], PossibleInliers[:, [3, 4]])
        if IsTransformExtreme(homography):
            # print(f"Found Extreme transformation \n {homography}")
            continue
        if homography is None:
            continue
        homography = np.vstack((homography, [0, 0, 1]))
        homographyTransposed = homography.transpose()

        # filter degenerative homoraphy -- doesn't seem to do anything
        if not isRigidTransformation(homography):
            continue

        # find how many points fit the model
        inliersCount = 0
        transformed = np.matmul(filtered[:, [0, 1, 2]], homographyTransposed)

        for transformedPoint, referencePoint in zip(transformed, filtered[:, [3, 4, 5]]):
            if get2PointsDistance(transformedPoint, referencePoint) < eps:
                inliersCount += 1
        if inliersCount > bestInliersCount:
            bestInliersCount = inliersCount
            bestmodel = homography

    additionalInfo = {
        "InliersCount": bestInliersCount,
    }
    return bestmodel, additionalInfo


def calculateSimilarityTansformation(p1, p2, eps):
    stackedPoints = np.hstack((p1, p2))

    # filter points that openPose didnt find + convert to float32
    filtered = np.float32(stackedPoints[np.all(stackedPoints != 0, axis=1)])
    if filtered.shape[0] < 3:
        return np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), {"InliersCount": -1}

    # ransac
    bestInliersCount = -1
    bestmodel = None
    for cycle in range(0, 1000):
        # chose random 3 pairs of points
        InlierIndices = np.random.choice(filtered.shape[0], size=3, replace=False)
        PossibleInliers = filtered[InlierIndices, :]

        # calculate model
        homography, _ = cv2.estimateAffinePartial2D(PossibleInliers[:, [0, 1]], PossibleInliers[:, [3, 4]])
        if IsTransformExtreme(homography):
            # print(f"Found Extreme transformation \n {homography}")
            continue
        if homography is None:
            continue
        homography = np.vstack((homography, [0, 0, 1]))
        homographyTransposed = homography.transpose()

        # filter degenerative homoraphy -- doesn't seem to do anything
        if not isRigidTransformation(homography):
            continue

        # find how many points fit the model
        inliersCount = 0
        transformed = np.matmul(filtered[:, [0, 1, 2]], homographyTransposed)

        for transformedPoint, referencePoint in zip(transformed, filtered[:, [3, 4, 5]]):
            if get2PointsDistance(transformedPoint, referencePoint) < eps:
                inliersCount += 1
        if inliersCount > bestInliersCount:
            bestInliersCount = inliersCount
            bestmodel = homography

    additionalInfo = {
        "InliersCount": bestInliersCount,
    }
    return bestmodel, additionalInfo



def getRansacAffineTransformation(p1, p2, eps=50):
    # rearange points in [x1, y1, 1, x2, y2, 1] format
    p1 = np.pad(p1, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    p2 = np.pad(p2, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    p1Mirrored = p1[[0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11]]

    model, info = calculateAffineTansformation(p1, p2, eps)
    # modelMirrored, infoMirrored = calculateAffineTansformation(p1Mirrored, p2, eps)

    # if infoMirrored['InliersCount'] > info['InliersCount']:
    #     # return modelMirrored, infoMirrored
    #     return None, infoMirrored # TODO TODO TODO
    
    return model, info    
 


def getRansacHomography(p1, p2, eps=50):
    # rearange points in [x1, y1, 1, x2, y2, 1] format
    p1 = np.pad(p1, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    p2 = np.pad(p2, ((0, 0), (0, 1)), mode="constant", constant_values=1)
    stackedPoints = np.hstack((p1, p2))
    # filter points that openPose didnt find
    filtered = stackedPoints[np.all(stackedPoints != 0, axis=1)]

    # ransac
    bestInliersCount = -1
    bestmodel = 0
    for cycle in range(0, 1000):
        # chose random 4 rows
        InlierIndices = np.random.choice(filtered.shape[0], size=4, replace=False)
        PossibleInliers = filtered[InlierIndices, :]

        # calculate model
        homography, status = cv2.findHomography(PossibleInliers[:, [0, 1]], PossibleInliers[:, [3, 4]])
        if homography is None:
            continue
        homographyTransposed = homography.transpose()

        # filter degenerative homoraphy -- doesn't seem to do anything
        if not isRigidTransformation(homography):
            continue

        # find how many points fit the model
        inliersCount = 0
        transformed = np.matmul(filtered[:, [0, 1, 2]], homographyTransposed)

        for transformedPoint, referencePoint in zip(transformed, filtered[:, [3, 4, 5]]):
            if get2PointsDistance(transformedPoint, referencePoint) < eps:
                inliersCount += 1
        if inliersCount > bestInliersCount:
            bestInliersCount = inliersCount
            bestmodel = homography

    # calculate and evaluate final homography
    transformed = np.matmul(filtered[:, [0, 1, 2]], bestmodel.transpose())
    inliers = filtered[distancesBetweenPoints(transformed, filtered[:, [3, 4, 5]]) < eps, :]

    if inliers.shape[0] > 4:
        finalHomography, status = cv2.findHomography(inliers[:, [0, 1]], inliers[:, [3, 4]])
        if finalHomography is not None:
            bestmodel = finalHomography

    additionalInfo = {
        "InliersCount": bestInliersCount,
    }
    return bestmodel, additionalInfo


def AlignImagesInFolder(datasetPath, posePath):
    # Loading 
    mainPath = datasetPath + posePath
    jsonPath = mainPath + "\\Jsons\\"
    processedImagesPath = mainPath + "\\Processed\\"
    alignedPath = mainPath + "\\Aligned\\"
    ImagesPath = mainPath + "\\Images\\"
    json_files = [pos_json for pos_json in os.listdir(jsonPath) if pos_json.endswith(".json")]
    print("Found: ", len(json_files), "json keypoint frame files")
    images = []
    for file in json_files:
        js = json.load(open(jsonPath + file))
        if len(js["people"]) == 0:
            os.remove(jsonPath + file)
            print("Removed" + file)
            continue
        keyPoints = js["people"][0]["pose_keypoints_2d"]
        points = divideChunks(keyPoints, 3)
        image = {
            "file": file,
            "ImagePath": ImagesPath + file[:-15] + ".png",
            "points": points[:15],
        }
        images.append(image)

    # # Selecting reference picture
    # bestMeanInliers = -1
    # bestRefImage = images[0]
    # for refImage in random.sample(images, 10 if len(images) > 10 else len(images)):
    #     SumOfInliers = 0
    #     for image in images:
    #         homography, homographyInfo = getRansacAffineTransformation(image["points"], refImage["points"])
    #         # homography, homographyInfo = getRansacHomography(image["points"], refImage["points"])
    #         SumOfInliers += homographyInfo["InliersCount"]
    #     if SumOfInliers / (len(images) - 1) > bestMeanInliers:
    #         bestRefImage = refImage
    #         bestMeanInliers = SumOfInliers / (len(images) - 1)

    # print(f"Best ref image is {bestRefImage['file']} with {bestMeanInliers} MeanInliers.")

    for bestRefImage in images:
        for c, image in enumerate(images):
            homography, homographyInfo = getRansacAffineTransformation(image["points"], bestRefImage["points"])

            imageName = image["file"][:-15]

            if homography is None or homographyInfo["InliersCount"] < 10:
                continue

            # Store Json In Dataset
            data = {}
            data["src"] = image["ImagePath"]
            data["dst"] = bestRefImage["ImagePath"]
            data["transformation"] = homography.tolist()


            with open(datasetPath + "MachineData\\" + ("ValidationSet\\" if c % 8 == 0 else "TrainSet\\") + posePath + "_" + imageName + ".json", "w") as outfile:
                json.dump(data, outfile)

            # Read and Warp
            # img = cv2.imread(f"{processedImagesPath}{imageName}_rendered.png")
            img = cv2.imread(f"{ImagesPath}{imageName}.png")
            out = cv2.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

            # print info in image
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (5, 30)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 1
            # out = cv2.putText(out, f"Inlier count: {homographyInfo['InliersCount']}", org, font, fontScale, color, thickness, cv2.LINE_AA)

            # np.set_printoptions(precision=3)
            # np.set_printoptions(suppress=True)

            # out = cv2.putText(out, f"{homography[0, :]}", (5, 70), font, fontScale, color, thickness, cv2.LINE_AA)
            # out = cv2.putText(out, f"{homography[1, :]}", (5, 110), font, fontScale, color, thickness, cv2.LINE_AA)
            # out = cv2.putText(out, f"{homography[2, :]}", (5, 150), font, fontScale, color, thickness, cv2.LINE_AA)
            # save image
            cv2.imwrite(f"{alignedPath}_{bestRefImage['file'][:-15]}_{imageName}_aligned.png", out)
