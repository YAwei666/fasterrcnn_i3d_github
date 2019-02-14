import cv2
img = cv2.imread('/home/wbr/github/maskrcnn-benchmark/dataset/pic/ILSVRC2015_val_00173001.mp4/9_raw.jpeg')
cv2.rectangle(img, (0, 0), (100, 100), (0, 225, 0), 2)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(img)

