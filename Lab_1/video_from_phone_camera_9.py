# Задание 1
import cv2

# Задание 9
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not(ret):
        break

    cv2.imshow('iVCam iPhone Camera', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()