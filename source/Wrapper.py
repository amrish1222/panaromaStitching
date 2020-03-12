#!/usr/bin/evn python

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import copy


def corner_score(img):
    img  = cv2.GaussianBlur(img,(3,3),0)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray,5,3,0.04)
    
    
    minVal = 0.0005* dst.max()
    
    cScoreMap = np.where(dst <= minVal, 0, dst)
    
    
    x, y  = np.where(cScoreMap > 0)
    
    corners = np.hstack((y.reshape(-1,1),x.reshape(-1,1)))
    
    return cScoreMap, corners

def applyANMS(scoreMap, corners, image, numBest = 150):
    img = image.copy()
    C = scoreMap.copy()
    locmax = peak_local_max(C,min_distance=10)

    nStrong = locmax.shape[0]
    if nStrong < numBest:
        print("Not enough strong corners\n")
        print(nStrong)

    r = [np.Infinity for i in range(nStrong)]
    x=np.zeros((nStrong,1))
    y=np.zeros((nStrong,1))
    ed=0
    for i in range(nStrong):
        for j in range(nStrong):
            if(C[locmax[j][0],locmax[j][1]] > C[locmax[i][0],locmax[i][1]]):
                ed = (locmax[j][0]-locmax[i][0])**2 + (locmax[j][1]-locmax[i][1])**2
            if ed<r[i]:
                r[i] = ed
                x[i] = locmax[i][0]
                y[i] = locmax[i][1]

    ind = np.argsort(r)
    ind = ind[-numBest:]
    x_best=np.zeros((numBest,1))
    y_best=np.zeros((numBest,1))
    for i in range(numBest):
        x_best[i] = np.int0(x[ind[i]])
        y_best[i] = np.int0(y[ind[i]]) 
        cv2.circle(img,(y_best[i],x_best[i]),3,255,-1)
     
    corners = np.hstack((y_best.reshape(-1,1),x_best.reshape(-1,1)))
    return corners,img
  
def ransac(pts1, pts2, threshold = 5):
    Hfinal = np.zeros((3,3))
    maxInliers = 0
    for iters in range(100):
        ndxs = [np.random.randint(0,len(pts1)) for i in range(4)]
        p1 = pts1[ndxs]
        p2 = pts2[ndxs]
        
        H = cv2.getPerspectiveTransform(np.float32(p1), np.float32(p2))
        numInliers = 0
        for pt1, pt2 in zip(pts1, pts2):
            fromPt = np.array(pt1)
            toPt = np.array(pt2)
            newPt = np.dot(H, np.array([fromPt[0],fromPt[1],1]))
            
            if newPt[2]!=0:
                newPt/=newPt[2]
            else:
                newPt/1e-8
            
            diff = np.linalg.norm(toPt - newPt[:2])
            if diff < threshold:
                numInliers+=1
        if maxInliers < numInliers:
            maxInliers = numInliers
            Hfinal = H
            
            if maxInliers > 40 :
                break
       
    return Hfinal

def combine(img2, img1, H):

    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warpdPts = cv2.perspectiveTransform(pts2, H)
    pts = np.vstack((pts1, warpdPts)).reshape(-1,2)
    xmin, ymin = np.int32(pts.min(axis=0))
    xmax, ymax = np.int32(pts.max(axis=0))
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
#    cv2.imshow("Warped img2", result)
#    cv2.waitKey(0)
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1

    return result

def stitch(img1, img2):
    imgs = [img1, img2]
    scoreMapList = []
    cornersList = []
    for img in imgs:
        tempScoreMap, tempcorners = corner_score(img) 
        scoreMapList.append(tempScoreMap)
        cornersList.append(tempcorners)
    
    for corners, img in zip(cornersList, copy.deepcopy(imgs)):
        for corner in corners:
            cv2.circle(img,tuple(corner),2,(255,0,0), -1)
#        cv2.imshow("Detect Corners", img)
#        cv2.waitKey(0)
    
    anmsCornersList = []
    for scoreMap,  corners, img in zip(scoreMapList, cornersList, copy.deepcopy(imgs)):
        newCorners,img = applyANMS(scoreMap, corners, img, 150)
        anmsCornersList.append(newCorners)
#        cv2.imshow("Apply ANMS", img)
#        cv2.waitKey(0)
    
    detector = cv2.ORB_create(nfeatures=1500)
    descriptorsList = []
    kpList = []
    for corners, img in zip( anmsCornersList, copy.deepcopy(imgs)):
        img  = cv2.GaussianBlur(img,(3,3),0)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp = [cv2.KeyPoint(corner[0], corner[1], 5) for corner in corners]
#        kp = detector.detect(gray,None)
        kp, des = detector.compute(gray, kp)
        descriptorsList.append(des)
        kpList.append(kp)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(descriptorsList[0],descriptorsList[1])
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kpList[0],img2,kpList[1],matches[:10], None)
    plt.imshow(img3)
    plt.show()
    
    pts1 = np.float32([kpList[0][m.queryIdx].pt for m in matches])
    pts2 = np.float32([kpList[1][m.trainIdx].pt for m in matches])
    
    H = ransac(pts1, pts2)
    
    print(H)
    
    test = cv2.warpPerspective(img1, H,(img1.shape[1], img1.shape[0]))
    
#    cv2.imshow("Warp", test)
#    cv2.waitKey(0)
    
    combImg = combine(img1, img2,H)
#    cv2.imshow("Final", combImg)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
    return combImg

def main():
    
    imgs = [cv2.imread(file) for file in glob.glob("../data/train/set2/*.jpg")]
    
    result = imgs[0]
    for i in range(1,len(imgs)):
        result = stitch(result, imgs[i])
    result = cv2.resize(result, (1000,1000))
    cv2.imshow("Pano", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
    main()
 
