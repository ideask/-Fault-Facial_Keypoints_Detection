import cv2
import os
import numpy as np
import random as rd


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
        labelFileDir = os.path.join(folderName, 'label.txt')
        with open(labelFileDir) as f:
            lines = f.readlines()
        tmpList.extend(list(map((picFileDir + '/').__add__, lines)))
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

def expandRoi(x1, y1, x2, y2, imgWidth, imgHeight, ratio):
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
            roiX1, roiY1, roiX2, roiY2, roiWidth, roiHeight = expandRoi(x1, y1, x2, y2, width, height, ratio) # ratio usually 0.25
            cv2.rectangle(imgData, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(imgData, 'not expand:', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 199, 0), 1, 1)
            cv2.rectangle(imgDataExpandRoi, (roiX1, roiY1), (roiX2, roiY2), (0, 255, 0), 3)
            cv2.putText(imgDataExpandRoi, 'expand:', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 199, 0), 1, 1)
            for landMark in rect[1]:
                cv2.circle(imgData, (landMark[0], landMark[1]), 1, (0, 0, 255), 1)
                cv2.circle(imgDataExpandRoi, (landMark[0], landMark[1]), 1, (0, 0, 255), 1)
        imgs = np.hstack([imgData, imgDataExpandRoi])
        cv2.imshow("compare_pic", imgs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# truth = loadRectLandMarks(loadMetaDataList())
# drawRectLandMarksExpandOri(truth, 0.25)

def createExpandRoiTxt(metaDataList, ratio, scale): # scale为训练数据所占比例90%或者70%均能接受
    total = len(metaDataList)
    trainNums = int(total * scale) if int(total * scale) < total else total
    testNums = total - trainNums
    print((trainNums,testNums))
    write_lines = []
    for line in metaDataList:
        line = line.strip().split()
        picDir = line[0]
        imgData = cv2.imread(picDir)
        picWidth = imgData.shape[1]
        picHeight = imgData.shape[0]
        rect = list(map(int, list(map(float, line[1:5]))))
        x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
        roiX1, roiY1, roiX2, roiY2, roiWidth, roiHeight = expandRoi(x1, y1, x2, y2, picWidth, picHeight, ratio)
        keyPointList = []
        for item in line[5:]:
            keyPointList.append(float(item))
        keyPointList = np.array(keyPointList)
        keyPointList = keyPointList.reshape(21, 2)
        keyPointList -= np.array([roiX1, roiY1])
        keyPointList = keyPointList.flatten()
        # print(keyPointList)
        write_line = picDir + ' ' + str(roiX1) + ' ' + str(roiY1) + ' ' + str(roiX2) + ' ' + str(roiY2)
        for word in keyPointList:
            write_line += ' ' + str(word)
        write_line += '\n'
        write_lines.append(write_line)
    rd.shuffle(write_lines)
    with open('train.txt', 'w') as wf:
        for wl in write_lines[:trainNums]:
            wf.write(wl)
    with open('test.txt', 'w') as wf:
        for wl in write_lines[trainNums:]:
            wf.write(wl)

createExpandRoiTxt(loadMetaDataList(), 0.25, 0.9)

def testTrainTestTxt(filename):
    truth = {}
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        imgDir = line[0]
        if imgDir not in truth:
            truth[imgDir] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(int, list(map(float, line[5::2]))))
        y = list(map(int, list(map(float, line[6::2]))))
        landMarks = list(zip(x, y))
        truth[imgDir].append((rect, landMarks))
    for line in truth:
        imgData = cv2.imread(line)
        for rect in truth[line]:
            cv2.rectangle(imgData, (rect[0][0], rect[0][1]), (rect[0][2], rect[0][3]), (0, 255, 0), 3)
            for landMark in rect[1]:
                cv2.circle(imgData, (landMark[0] + rect[0][0], landMark[1] + rect[0][1]), 1, (0, 0, 255), 1)
        cv2.imshow(line, imgData)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

testTrainTestTxt('train.txt')