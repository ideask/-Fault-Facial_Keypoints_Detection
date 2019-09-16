import cv2
import numpy as np

folder_list = ['I', 'II']
labelfile = 'label.txt'

def loadLabel(folders):
    for folder in folders:
        file_name = './data/' + folder + '/' + labelfile
        with open(file_name) as f:
            lines = f.readlines()
            for line in lines:
                keyPointList = []
                print(line.split(' ')[0])
                filename = line.split(' ')[0]
                x1, y1 = line.split(' ')[1], line.split(' ')[2]
                x2, y2 = line.split(' ')[3], line.split(' ')[4]
                for item in line.split(' ')[5:]:
                    keyPointList.append(int(float(item)))
                keyPointList = np.array(keyPointList)
                keyPointList = np.array(keyPointList).reshape(21, 2)
                #print(keyPointList)

                image = cv2.imread('./data/' + folder + '/' + filename)
                cv2.rectangle(image, (int(float(x1)), int(float(y1))), (int(float(x2)), int(float(y2))), (0, 255, 0), 2)
                for point in keyPointList:
                    cv2.circle(image, (point[0], point[1]), 1, (0, 0, 255), 1) # point_size,point_color,thickness
                cv2.imshow('result.jpg',image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

loadLabel(folder_list)