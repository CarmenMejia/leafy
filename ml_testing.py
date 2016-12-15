# by Carmen Mejia
import sys, os, cv2
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Caluclates convex hull area over contour area
def area_over_hull_area(img):
    ret,thresh = cv2.threshold(img,100,255,0)
    (conts, heigh) = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE )

    hull_area = 0.0
    area = 0.0
    hulls = []
    for c in conts:
        hull = cv2.convexHull(c)
        hulls.append(hull)
        hull_area = hull_area + cv2.contourArea(hull)
        area = area + cv2.contourArea(c)
    # height = img.shape[0]
    # width = img.shape[1]
    # new = np.zeros((height,width,3), np.uint8)
    # cv2.drawContours(new,conts,-1,(0, 255, 0),3)
    # cv2.drawContours(new,hulls,-1,(0, 0, 255),3)
    # cv2.imshow('cont',new)
    # print area/hull_area
    # cv2.waitKey(0)
    return area/hull_area

# area of fitEllipse over area of contour
def fittingEllipse(img):
    ret,thresh = cv2.threshold(img,100,255,0)
    (conts, heigh) = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE )

    ellipse_area = 0.0
    area = 0.0
    ellipses = []
    for c in conts:
        if (len(c)>4):
            ellipse = cv2.fitEllipse(c)
            ellipses.append(ellipse)
            (x,y), (MA, ma), angle = ellipse
            ellipse_area = ellipse_area + (math.pi * MA *ma)
            area = area + cv2.contourArea(c)
    return area/ellipse_area

# not Used
# conour area over arc length
def arcLenth(img):
    ret,thresh = cv2.threshold(img,100,255,0)
    (conts, heigh) = cv2.findContours(thresh.copy(),cv2.cv.CV_RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_SIMPLE )

    arcLengthTot = 0.0
    areaTot = 0.0
    arcLengths = []
    for c in conts:
        length = cv2.arcLength(c, True)
        arcLengths.append(length)
        arcLengthTot = arcLengthTot + length
        areaTot = areaTot + cv2.contourArea(c)
    return areaTot/arcLengthTot


# not Used
# number of Hough Lines found
# Wasn't finding neough lines, mostly just one or two
# from the leaf's stem
def hough(img):
    edges = cv2.Canny((255-img),50,150,apertureSize = 3)
    print img.shape
    # cv2.imshow('edges', edges)
    # cv2.waitKey(0)
    hLines = cv2.HoughLines(edges, 1,np.pi/180,10)
    height = img.shape[0]
    width = img.shape[1]
    new = np.zeros((height,width,3), np.uint8)
    new[:,:,0] = img
    minLineLength = 2
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines != None:
        for x in range (0, len(lines)):
            for x1,y1,x2,y2 in lines[0]:
                cv2.line(new,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow('hough',edges)
    cv2.waitKey(0)


# gives baseline for the machine learning model
def baseline(totLabels, labels, features):
    # determines which label I have the most of
    maxL = 0
    for i in range(len(totLabels)):
        totLabels[i] = (totLabels[i][0], labels.count(totLabels[i][0]))
        if totLabels[i][1] >= maxL:
            maxL = totLabels[i][1]
            maxi = i

    # splits data into training and testing features
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)


    baseLabels = [totLabels[maxi][0]]*len(testLabels)
    acc = metrics.accuracy_score(testLabels, baseLabels)
    pre = metrics.precision_score(testLabels,baseLabels,average='macro')
    rec = metrics.recall_score(testLabels,baseLabels,average='macro')
    f1 = metrics.f1_score(testLabels, baseLabels,average="macro")
    print("[INFO] zero rule base accuracy: {:.2f}%".format(acc * 100))
    print("[INFO] zero rule base precision: {:.2f}%".format(pre * 100))
    print("[INFO] zero rule base recall: {:.2f}%".format(rec * 100))
    print("[INFO] zero rule base f1: {:.2f}%".format(f1 * 100))

# tests the machine learning model
def test(labels, features):
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(trainFeat, trainLabels)
    predLabels = model.predict(testFeat)
    acc = metrics.accuracy_score(testLabels, predLabels)
    pre = metrics.precision_score(testLabels,predLabels,average='macro')
    rec = metrics.recall_score(testLabels,predLabels,average='macro')
    f1 = metrics.f1_score(testLabels, predLabels,average="macro")

    print("[INFO] accuracy: {:.2f}%".format(acc * 100))
    print("[INFO] precision: {:.2f}%".format(pre * 100))
    print("[INFO] recall: {:.2f}%".format(rec * 100))
    print("[INFO] f1: {:.2f}%".format(f1 * 100))


def main():
    # extract image directionr (should be segmented images)
    try:
        seg_dir = os.path.abspath(sys.argv[1])
    except:
        print "\nUSAGE: python create_labels.py segments_dir\n"
        sys.exit()


    out_dir = "test_label"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    labels = []
    features = []

    totLabels = []

    for subdir, dirs, files in os.walk(os.path.join('..', seg_dir)):
        for file in files:
            # loads image as grayscale
            img = cv2.imread(os.path.join(subdir, file),0)

            path_prefix = os.path.split(subdir)[-1]
            labels.append(path_prefix)

            # gets features
            features.append([area_over_hull_area(img), fittingEllipse(img)])

            # experiment here



            # Used for zero value baseline testing
            if totLabels.count((path_prefix,0)) < 1:
                totLabels.append((path_prefix,0))

    # compute baseline and test restuls and print out
    baseline(totLabels,labels,features)
    print ''
    test(labels,features)


if __name__=="__main__":
    x =  main()
