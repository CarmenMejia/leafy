import sys, os, cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

try:
    seg_dir = os.path.abspath(sys.argv[1])
except:
    print "\nUSAGE: python create_labels.py segments_dir\n"
    sys.exit()


out_dir = "test_label"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
# os.chdir(out_dir)
labels = []
features = []

for subdir, dirs, files in os.walk(os.path.join('..', seg_dir)):
    for file in files:
        filename, ext = os.path.splitext(file)
        path_prefix = os.path.split(subdir)[-1]

        img = cv2.imread(os.path.join(subdir, file),0)

        labels.append(path_prefix)


        ret,thresh = cv2.threshold(img,127,255,0)
        (conts, heigh) = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE )
        hull_area = 0.0
        area = 0
        for c in conts:
            hull = cv2.convexHull(c)
            hull_area = hull_area + cv2.contourArea(hull)
            area = area + cv2.contourArea(c)
        # print hull_area
        # print area
        # print ""
        features.append([area/hull_area])
        # features.append(cv2.contourArea(hull)/cv2.contourArea(cnts))
        # height = thresh.shape[0]
        # width = thresh.shape[1]
        # new = np.zeros((height,width,3), np.uint8)
        # cv2.drawContours(new,cnts,-1,(0, 255, 0),3)
        # cv2.imwrite(os.path.join(out_dir, file), new)


# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)

model = KNeighborsClassifier(n_neighbors=4,	n_jobs=-1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
predLabels = model.predict(testFeat)
f1 = metrics.f1_score(testLabels, predLabels,average='weighted')
print("[INFO] accuracy: {:.2f}%".format(acc * 100))
print("[INFO] f1: {:.2f}%".format(f1 * 100))
