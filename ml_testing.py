import sys, os, cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# Caluclates convex hull area over contour area
def area_over_hull_area(img):
    ret,thresh = cv2.threshold(img,127,255,0)
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
    cv2.waitKey(0)
    return area/hull_area

# look at number of votes
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

    cv2.imshow('hough',new)
    cv2.waitKey(0)

def corners(img):
    cv2.imshow('orin',img)
    cv2.waitKey(0)

    out = cv2.cornerHarris(img,2,3,0.04)

    dst = cv2.dilate(out,None)
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)

    # find centroids
    ret, labels, stats, centroids = cv2.cv.connectedComponentsWithStats(dst, 4,cv2.CV_325)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    # Now draw them
    res = np.hstack((centroids,corners))
    res = np.int0(res)
    img[res[:,1],res[:,0]]=[0,0,255]
    img[res[:,3],res[:,2]] = [0,255,0]


    cv2.imshow('corner', img)
    cv2.waitKey(0)

def blob(img):
    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = 0
    params.maxThreshold = 256

    params.filterByArea = True
    params.minArea = 3

    # Filter by Circularity
    params.filterByCircularity = False
    # params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    # params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia =False
    # params.minInertiaRatio = 0.5

    detector = cv2.SimpleBlobDetector()

    img = cv2.inRange(img, (100), (255))

    reversemask = 255-img
    keypoints = detector.detect(reversemask)
    cv2.imshow('feed',cv2.drawKeypoints(img,keypoints,(0,255,0)))
    cv2.waitKey(0)
    # cv2.imshow('inv',reversemask)
    # cv2.waitKey(0)
    print keypoints


def baseline(totLabels, labels, features):
    maxL = 0
    for i in range(len(totLabels)):
        totLabels[i] = (totLabels[i][0], labels.count(totLabels[i][0]))
        if totLabels[i][1] >= maxL:
            maxL = totLabels[i][1]
            maxi = i

    # partition the data into training and testing splits, using 75%
    # of the data for training and the remaining 25% for testing
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

def test(labels, features):
    (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
    model = KNeighborsClassifier(n_neighbors=7,	n_jobs=-1)
    model.fit(trainFeat, trainLabels)
    predLabels = model.predict(testFeat)
    acc = metrics.accuracy_score(testLabels, predLabels)
    pre = metrics.precision_score(testLabels,predLabels,average='macro')
    rec = metrics.recall_score(testLabels,predLabels,average='macro')
    f1 = metrics.f1_score(testLabels, predLabels,average="macro")
    precision, recall, fbeta, support = metrics.precision_recall_fscore_support(testLabels, predLabels,1,average='macro')

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

            features.append([area_over_hull_area(img)])

            # experiment here
            blob(img)


            # Used for zero value baseline testing
            if totLabels.count((path_prefix,0)) < 1:
                totLabels.append((path_prefix,0))

    # compute baseline and test restuls and print out
    baseline(totLabels,labels,features)
    print ''
    test(labels,features)


if __name__=="__main__":
    x =  main()
