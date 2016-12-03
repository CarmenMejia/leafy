import numpy as np
import cv2, os, sys
import matplotlib as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# modified from Stanford 2013
def kmeans(dataSet, k):
    numFeatures = 2
    centroids = getRandomCentroids(numFeatures, k)
    # Initialize book keeping vars.
    iterations = 0
    oldCentroids = None

    # Run the main k-means algorithm
    while not shouldStop(oldCentroids, centroids, iterations):
        # Save old centroids for convergence test. Book keeping.
        oldCentroids = centroids
        iterations += 1

        # Assign labels to each datapoint based on centroids
        labels = getLabels(dataSet, centroids)

        # Assign centroids based on datapoint labels
        centroids = getCentroids(dataSet, labels, k)

    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids



def shouldStop(oldCentroids, centroids, iterations):
    if iterations > MAX_ITERATIONS: return True
    return oldCentroids == centroids














def kmeans():
    digits = load_digits(2)
    data = scale(digits.datda)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(digits.target))
    labels = digits.target




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

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # s = hsv[:,:,1]
            # v = hsv[:,:,2]

            # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
            # thresh = cv2.medianBlur(thresh,5)

            # ret, thresh = cv2.threshold(hsv[1],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            # ret2, thresh = cv2.threshold(hsv[2],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


            # print hsv[0].shape
            height = hsv.shape[0]
            width = hsv.shape[1]

            hsv = hsv.reshape((height * width, 3))
            s = hsv[:,1]
            v = hsv[:,2]


            sv = cv2.merge((s,v))
            sv = sv.reshape((height * width), 2)
            clt = KMeans(n_clusters = 2)
            clt.fit(sv)

            # reshaping the hsv back to the original height and width
            thresh = hsv[:,1].reshape(height, width)

            labels = clt.labels_
            thresh = labels.reshape(height, width)
            for i in range(0,height):
                for j in range(0,width):
                    if (thresh[i,j] == 1):
                        thresh[i,j] = 255


            cv2.imwrite(os.path.join(out_dir, file), thresh)
