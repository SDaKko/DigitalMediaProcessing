# Задание 1
import cv2
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
