import cv2
from tracker import *

tracker = EuclideanDistTracker()

track = cv2.VideoCapture("highway.mp4")

car_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)
while True:
    ret, frame = track.read()
    height, width, _ = frame.shape


    #roi2 = frame[340: 720, 500: 800]
    roi2 = frame[000: 720, 000: 320]

    mask = car_detector.apply(roi2)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for vnd in contours:
        area = cv2.contourArea(vnd)

        if area > 100:
            x, y, w, h = cv2.boundingRect(vnd)

            detections.append([x, y, w, h])

    rectangles_ids = tracker.update(detections)

    for rect_id in rectangles_ids:
        x, y, w, h, id = rect_id
        cv2.putText(roi2, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(roi2, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("roi2", roi2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitkey(30)
    if key == 27:
        break


track.release()
cv2.destroyAllWindows()