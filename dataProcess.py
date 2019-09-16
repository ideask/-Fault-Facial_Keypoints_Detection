import cv2
import os
import numpy as np

folderList = ['I', 'II']

def removeInvalidImg(srcImgList):
    resList= []
    for item in srcImgList:
        picDir = item.split()[0]
        if os.path.isfile(picDir):
            resList.append(item)
    return resList

def loadMetaDataList():
    tmpList = []
    for folderName in folderList:
        picFileDir = os.path.join('data',folderName)
        labelFileDir = os.path.join(picFileDir, 'label.txt')
        with open(labelFileDir) as f:
            lines = f.readlines()
        tmpList.extend(list(map((picFileDir + '\\').__add__, lines)))
    resLines = removeInvalidImg(tmpList)
    return resLines

def loadRectLandMarks(metaDataList):
    truth = {}
    for line in metaDataList:
        line = line.strip().split()
        imgDir = line[0]
        if imgDir not in truth:
            truth[imgDir] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(int,list(map(float, line[5::2]))))
        y = list(map(int,list(map(float, line[6::2]))))
        landMarks = list(zip(x, y))
        truth[imgDir].append((rect, landMarks))
    return truth

# truth = loadRectLandMarks(loadMetaDataList())
# for i in truth:
#     print(i, truth[i])

def drawRectLandMarks(truth):
    for line in truth:
        imgData = cv2.imread(line)
        for rect in truth[line]:
            cv2.rectangle(imgData, (rect[0][0], rect[0][1]), (rect[0][2], rect[0][3]), (0, 255, 0), 3)
            for landMark in rect[1]:
                cv2.circle(imgData, (landMark[0], landMark[1]), 1, (0, 0, 255), 1)
        cv2.imshow(line, imgData)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# truth = loadRectLandMarks(loadMetaDataList())
# drawRectLandMarks(truth)

def expand_roi(x1, y1, x2, y2, imgWidth, imgHeight, ratio):
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    paddingWidth = int(width * ratio)
    paddingHeight = int(height * ratio)
    roiX1 = x1 - paddingWidth
    roiY1 = y1 - paddingHeight
    roiX2 = x2 + paddingWidth
    roiY2 = y2 + paddingHeight
    roiX1 = 0 if roiX1 < 0 else roiX1
    roiY1 = 0 if roiY1 < 0 else roiY1
    roiX2 = imgWidth - 1 if roiX2 >= imgWidth else roiX2
    roiY2 = imgHeight - 1 if roiY2 >= imgHeight else roiY2
    return roiX1, roiY1, roiX2, roiY2, \
            roiX2 - roiX1 + 1, roiY2 - roiY1 + 1

# 蓝色为原来ROI，绿色框是expand后的ROI
def drawRectLandMarksExpandOri(truth, ratio):
    for line in truth:
        imgData = cv2.imread(line)
        imgDataExpandRoi = cv2.imread(line)
        width = imgData.shape[1]
        height = imgData.shape[0]
        for rect in truth[line]:
            x1, y1 = rect[0][0], rect[0][1]
            x2, y2 = rect[0][2], rect[0][3]
            roiX1, roiY1, roiX2, roiY2, roiWidth, roiHeight = expand_roi(x1, y1, x2, y2, width, height, ratio) # ratio usually 0.25
            cv2.rectangle(imgData, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.rectangle(imgDataExpandRoi, (roiX1, roiY1), (roiX2, roiY2), (0, 255, 0), 3)
            for landMark in rect[1]:
                cv2.circle(imgData, (landMark[0], landMark[1]), 1, (0, 0, 255), 1)
                cv2.circle(imgDataExpandRoi, (landMark[0], landMark[1]), 1, (0, 0, 255), 1)
        imgs = np.hstack([imgData, imgDataExpandRoi])
        cv2.imshow("compare_pic", imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# truth = loadRectLandMarks(loadMetaDataList())
# drawRectLandMarksExpandOri(truth, 0.25)

