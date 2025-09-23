# Задание 1
import cv2

# Задание 6
def print_cam():
    cap = cv2.VideoCapture(0)

    cap.set(3, 640)
    cap.set(4, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center_x, center_y = w // 2, h // 2
    cross_size = 90
    thickness = 25

    while True:
        ret, frame = cap.read()

        if not (ret):
            break

        h_start_y = center_y - thickness // 2
        h_end_y = center_y + thickness // 2
        h_start_x = center_x - cross_size
        h_end_x = center_x + cross_size

        cv2.line(frame,
                 (center_x - 10, center_y - 100),
                (center_x + 100, center_y + 150),
                (0, 0, 255), 3)
        cv2.line(frame,
                 (center_x - 10, center_y - 100),
                 (center_x - 100, center_y + 150),
                 (0, 0, 255), 3)
        cv2.line(frame,
                 (center_x + 100, center_y + 150),
                 (center_x - 150, center_y),
                 (0, 0, 255), 3)
        cv2.line(frame,
                 (center_x - 100, center_y + 150),
                 (center_x + 150, center_y),
                 (0, 0, 255), 3)
        cv2.line(frame,
                 (center_x - 150, center_y),
                 (center_x + 150, center_y),
                 (0, 0, 255), 3)

        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

print_cam()