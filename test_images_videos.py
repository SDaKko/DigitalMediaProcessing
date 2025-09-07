# Задание 1
import cv2

# Задание 2
# img = cv2.imread(r'C:\Pics\photo.png', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread(r'C:\Pics\zakat.jpg', cv2.IMREAD_COLOR)
# img3 = cv2.imread(r'C:\Pics\waterfall.gif', cv2.IMREAD_COLOR_RGB)
# cv2.namedWindow('output', cv2.WINDOW_NORMAL)
# cv2.imshow('output', img)
# cv2.waitKey(1000)
# cv2.namedWindow('output2', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('output2', img2)
# cv2.waitKey(1000)
# cv2.namedWindow('output3', cv2.WINDOW_FULLSCREEN)
# cv2.imshow('output3', img3)
# cv2.waitKey(1000)

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
    # cv2.imshow('Размеры уменьшены в 2 раза', resized_down)

    # cv2.imshow('video1_hsv', hsv)
    # cv2.imshow('video1_gray', gray)
    # cv2.imshow('video1', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Задание 4
def read_write_file(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
    while (True):
        ret, frame = cap.read()

        if not (ret):
            break

        cv2.imshow('video2', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

input_path = r'C:\Videos\video.mp4'
output_path = r'C:\Videos\output.mp4'
# output_path = r'C:\Videos\output.avi'
# output_path = r'C:\Videos\output.mov'
read_write_file(input_path, output_path)

# Задание 5
img_new = cv2.imread(r'C:\Pics\zakat.jpg')
hsv_img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2HSV)
cv2.imshow("img_new", img_new)
cv2.waitKey(1000)
cv2.imshow("hsv_img_new", hsv_img_new)
cv2.waitKey(1000)
cv2.destroyAllWindows()

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
        horizontal_region = frame[h_start_y:h_end_y, h_start_x:h_end_x]
        blurred_horizontal = cv2.blur(horizontal_region, (25, 25))
        frame[h_start_y:h_end_y, h_start_x:h_end_x] = blurred_horizontal

        # Горизонтальная полоса
        cv2.rectangle(frame,
                      (center_x - cross_size, center_y - thickness // 2),
                      (center_x + cross_size, center_y + thickness // 2),
                      (0, 0, 255), 0)

        # Вертикальная полоса
        cv2.rectangle(frame,
                      (center_x - thickness // 2, center_y - cross_size),
                      (center_x + thickness // 2, center_y - thickness // 2),
                      (0, 0, 255), 0)
        cv2.rectangle(frame,
                      (center_x - thickness // 2, center_y + thickness // 2),
                      (center_x + thickness // 2, center_y + cross_size),
                      (0, 0, 255), 0)

        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

print_cam()

# Задание 7
import datetime
def from_camera(output_video_path):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()

        if not (ret):
            break

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("camera", frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

output_path = r'C:\Videos\from_wcamera.mp4'
from_camera(output_path)