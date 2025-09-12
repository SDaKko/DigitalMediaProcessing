# Задание 1
import cv2

# Задание 2
img = cv2.imread(r'C:\Pics\photo.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r'C:\Pics\zakat.jpg', cv2.IMREAD_COLOR)
img3 = cv2.imread(r'C:\Pics\waterfall.gif', cv2.IMREAD_REDUCED_GRAYSCALE_2)
cv2.namedWindow('output', cv2.WINDOW_NORMAL)
cv2.imshow('output', img)
cv2.waitKey(1000)
cv2.namedWindow('output2', cv2.WINDOW_AUTOSIZE)
cv2.imshow('output2', img2)
cv2.waitKey(1000)
cv2.namedWindow('output3', cv2.WINDOW_FULLSCREEN)
cv2.imshow('output3', img3)
cv2.waitKey(0)