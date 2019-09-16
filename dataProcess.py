import cv2
import os

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

truth = loadRectLandMarks(loadMetaDataList())
drawRectLandMarks(truth)


