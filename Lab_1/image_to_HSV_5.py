# Задание 1
import cv2

# Задание 5
img_new = cv2.imread(r'C:\Pics\zakat.jpg')
hsv_img_new = cv2.cvtColor(img_new, cv2.COLOR_BGR2HSV)
cv2.imshow("img_new", img_new)
cv2.waitKey(1000)
cv2.imshow("hsv_img_new", hsv_img_new)
cv2.waitKey(0)
cv2.destroyAllWindows()