# Задание 1
import cv2

# Задание 3
cap = cv2.VideoCapture(r'C:\Videos\video.mp4', cv2.CAP_ANY)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Размер: {width}x{height}")
print(f"FPS: {fps}")
print(f"Всего кадров: {frame_count}")



while True:
    ret, frame = cap.read()

    if not(ret):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    resized_down = cv2.resize(frame, (width // 2, height // 2))
    cv2.imshow('Resized down x2', resized_down)

    # cv2.imshow('video1_hsv', hsv)
    # cv2.imshow('video1_gray', gray)
    # cv2.imshow('video1', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
