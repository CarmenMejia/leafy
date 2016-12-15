# by Carmen Mejia
import numpy as np
import cv2, os, sys
import matplotlib as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# reads in the first argument
try:
    image_dir = os.path.abspath(sys.argv[1])
except:
    print "\nUSAGE: python segment.py image_dir\n"
    sys.exit()

# creates directory to put segmented images
out_dir = "segmented"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# for each image, segments by using KMeans with Saturation and value as features
for subdir, dirs, files in os.walk(os.path.join('..', image_dir)):
    for file in files:
        filename, ext = os.path.splitext(file)
        path_prefix = os.path.split(image_dir)[-1]

        if ext == '.tif' or ext == '.jpg':
            # reads in image
            img = cv2.imread(os.path.join(subdir, file))

            # converts to HSV color scale
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            height = hsv.shape[0]
            width = hsv.shape[1]

            hsv = hsv.reshape((height * width, 3))
            s = hsv[:,1]
            v = hsv[:,2]

            # performs KMeans with features
            sv = cv2.merge((s,v))
            sv = sv.reshape((height * width), 2)
            clt = KMeans(n_clusters = 2)
            clt.fit(sv)

            # reshaping the hsv back to the original height and width
            thresh = hsv[:,1].reshape(height, width)

            # creating segmented image
            labels = clt.labels_
            thresh = labels.reshape(height, width)
            if thresh[0,width/2] == 0:
                for i in range(0,height):
                    for j in range(0,width):
                        if (thresh[i,j] == 1):
                            thresh[i,j] = 255
            else:
                for i in range(0,height):
                    for j in range(0,width):
                        if (thresh[i,j] == 0):
                            thresh[i,j] = 255
                        else :
                            thresh[i,j] = 0

            path_prefix = os.path.split(subdir)[-1]

            # saving segmented image
            if not os.path.exists(os.path.join(out_dir,path_prefix)):
                os.mkdir(os.path.join(out_dir,path_prefix))

            cv2.imwrite(os.path.join(out_dir, path_prefix, file), thresh)
