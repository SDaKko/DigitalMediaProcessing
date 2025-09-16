import cv2
import numpy as np

def red_object_tracking():

    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Не удалось прочитать кадр")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = cv2.bitwise_or(mask1, mask2)

        red_only = cv2.bitwise_and(frame, frame, mask=red_mask)

        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.dilate(cv2.erode(red_mask, kernel), kernel)
        closing = cv2.erode(cv2.dilate(opening, kernel), kernel)

        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = frame.copy()

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            M = cv2.moments(largest_contour)
            area = M['m00']

        cv2.imshow('video', frame)
        cv2.imshow('Red Mask (Threshold)', red_mask)
        cv2.imshow('Red Only', red_only)
        cv2.imshow('After Opening', opening)
        cv2.imshow('After Closing', closing)


        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

red_object_tracking()
