from libraries import *

# HSV color space 
# Find the object minimum and maximum of hue, sturation, value
# Using trackbars make the object white and rest dark

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,320)
cv2.createTrackbar("Hue Min","TrackBars",0,179,lambda a: a)
cv2.createTrackbar("Hue Max","TrackBars",0,179,lambda a: a)
cv2.createTrackbar("Sat Min","TrackBars",195,255,lambda a: a)
cv2.createTrackbar("Sat Max","TrackBars",255,255,lambda a: a)
cv2.createTrackbar("Val Min","TrackBars",0,255,lambda a: a)
cv2.createTrackbar("Val Max","TrackBars",255,255,lambda a: a)


cap = cv2.VideoCapture(1)

while True:
    success, img = cap.read()
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, s_min, v_min, h_max, s_max, v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    imgResult = cv2.bitwise_and(img,img,mask=mask)

    imgStack = stackImages(.5,([img,imgHSV],[mask,imgResult]))
    cv2.imshow("Stacked Images", imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# h_min, s_min, v_min, h_max, s_max, v_max  
# 0 0 216 65 123 255   // light
# 0 0 214 7 255 255   // orange
