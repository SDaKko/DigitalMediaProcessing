import cv2

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

        cv2.imshow('video', frame)


        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

red_object_tracking()
