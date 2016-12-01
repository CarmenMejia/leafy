import numpy as np
import cv2
from matplotlib import pyplot as plt



for subdir, dirs, files in os.walk(os.path.join('..', experiment_dir)):
    for file in files:
        filename, ext = os.path.splitext(file)
        path_prefix = os.path.split(experiment_dir)[-1]
        # note: opencv doesn't support PDFs so those need to be changed to images
        if ext == '.tif' or ext == '.jpg':
            # print filename
            # make a directory for the image
            if not os.path.exists(filename):
                os.mkdir(filename)
            os.chdir(filename)

            # read the image and run the given function on it
            img = cv2.imread(os.path.join(subdir, file))
            cv2.imwrite(file, img)


img = cv2.imread('coins.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
