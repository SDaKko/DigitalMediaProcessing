# Задание 1
import cv2

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



