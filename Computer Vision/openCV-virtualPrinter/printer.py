from libraries import *


pts = []

# Color range for each color
# Get from colorDetector.py
colors = [[0, 0, 216, 65, 123, 255],  # light
          [0, 0, 214, 7, 255, 255]]  # orange

# printer color value
colorsValue = [(0,255,0),
                (74,74,255)]


cap = cv2.VideoCapture(1)

while True:
    success, frame = cap.read()
    imgResult = frame.copy()

    getColors(frame, colors, colorsValue, pts)
    printer(imgResult, pts)

    cv2.imshow('imgResult', imgResult)
    #print(pts)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break