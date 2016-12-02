import numpy as np
import cv2, os, sys
from matplotlib import pyplot as plt



try:
    image_dir = os.path.abspath(sys.argv[1])
except:
    print "\nUSAGE: python segment.py image_dir\n"
    sys.exit()


out_dir = "segmented"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
# os.chdir(out_dir)

for subdir, dirs, files in os.walk(os.path.join('..', image_dir)):
    for file in files:
        filename, ext = os.path.splitext(file)
        path_prefix = os.path.split(image_dir)[-1]

        # note: opencv doesn't support PDFs so those need to be changed to images
        if ext == '.tif' or ext == '.jpg':
            # read the image and run the given function on it
            img = cv2.imread(os.path.join(subdir, file))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            print gray
            
            # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            # thresh = cv2.medianBlur(thresh,5)

            # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # read the image and run the given function on it
            # img = cv2.imread(os.path.join(subdir, file))
            # print (os.path.join(out_dir,file))

            # cv2.imwrite(os.path.join(out_dir, file), thresh)
