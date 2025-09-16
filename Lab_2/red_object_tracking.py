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

        cv2.imshow('video', frame)
        cv2.imshow('Red Mask (Threshold)', red_mask)
        cv2.imshow('Red Only', red_only)


        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

red_object_tracking()
