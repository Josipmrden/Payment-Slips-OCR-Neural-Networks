import cv2
import os
import datasetAdapter


def scale_image(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key= lambda ctr: cv2.boundingRect(ctr)[0])

    area = 0
    x1, y1, w1, h1 = 0, 0, 0, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        tempArea = w*h
        if tempArea > area:
            area = tempArea
            cout = c
            x1 = x
            y1 = y
            w1 = w
            h1 = h

    rect = img[y1:y1+h1, x1:x1+w1]
    return rect




if __name__ == "__main__":
    basePath = "./num_letters/"
    trainPath = basePath + "train/"
    testPath = basePath + "test/"

    print("Tu sam")
    trainDest = basePath + "scaled_train/"
    testDest = basePath + "scaled_test/"

    for dir in os.listdir(trainPath):
        directoryPath = trainDest + dir
        if not os.path.exists(directoryPath):
            os.makedirs(directoryPath)
        for imgfilename in os.listdir(trainPath + dir):
            img = cv2.imread(trainPath + dir + "/" + imgfilename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            scaled = scale_image(gray)
            #cv2.imshow("scaled", scaled)
            #cv2.waitKey()
            replenished = datasetAdapter.adapt_picture(scaled, 20, 28)
            #cv2.imshow("replenished", replenished)
            #cv2.waitKey()
            pathToSave = trainDestletters_cnn.py + dir + "/" + imgfilename
            print(pathToSave)
            cv2.imwrite(pathToSave, replenished)
