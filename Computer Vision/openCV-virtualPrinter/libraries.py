import cv2
import numpy as np

# print on canvas
def printer(imgResult, pts):

    # pts = [(x coordinate, ycoordinate, detected color)]
    for _ in pts:
        cv2.circle(imgResult, (_[0], _[1]), 5, _[2], cv2.FILLED)


def getColors(img, colors, colorsValue, pts):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    i = 0
    # Create mask for all specified color in a frame
    for _ in colors:
        mask = cv2.inRange(imgHSV, np.array(_[0:3]), np.array(_[3:6]))

        #cv2.imshow(str(_[3]), mask)

        x, y = getContours(mask)

        # condition for resisting duplication of same points befor appending
        if (x & y != 0) & ((x,y,colorsValue[i]) not in pts):
            pts.append((x,y,colorsValue[i]))

        i+=1


# Find contours and return x & y coordinate
def getContours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0,0,0,0
    for _ in contours:
        area = cv2.contourArea(_)

        # Minimum area for our marker pen
        if area >=300:
            curveLen = cv2.arcLength(_, True)
            approx = cv2.approxPolyDP(_, curveLen*.02, True)
            x, y, w, h = cv2.boundingRect(approx)

    return x, y


# stack images
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver