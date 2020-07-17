import cv2
import imutils
import numpy as np

bg = None
cam = cv2.VideoCapture(0)
i = 1
frame_no = 0
top, right, bottom, left = 200, 500, 355, 750
ra_weight = 1
emoji_name = input("Enter emoji name: ")




def run_avg(image, ra_weight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, ra_weight)


def segment(image):
    global bg
    diff = cv2.absdiff(image,bg.astype("uint8"))
    threshold = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (threshold, segmented) 


while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=800)
    frame = cv2.flip(frame, 1)
    clone = frame.copy()
    roi = frame[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if frame_no < 30:
        run_avg(gray, ra_weight)
    else:
        hand = segment(gray)
        if hand is not None:
            (threshold, segmented) = hand
           #cv2.drawContours(clone, [segmented + (right, top)], -1, (255, 255, 0)) #right and top are to move the contours from (0,0)
            cv2.imshow("Thesholded", threshold)
        
    cv2.rectangle(clone, (left, top), (right, bottom), (0,0,0), 2)
    frame_no += 1
    number = str(i-1)
    cv2.putText(clone,number,(0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.imshow("Video Feed", clone)
    
    k = cv2.waitKey(1)
    if k == 13:
        break
    elif k == 32:
        name = str(emoji_name) + '_' + str(i)
        cv2.imwrite('C:/Users/gsnik/Desktop/Hand Recognition/'+name+'.jpg', threshold)
        i += 1

cam.release()
cv2.destroyAllWindows()