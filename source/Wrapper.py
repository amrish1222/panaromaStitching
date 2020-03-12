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
    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    
    minVal = 0.001* dst.max()
    
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
           
    return x_best,y_best,img
    

def main():
    
    imgs = [cv2.imread(file) for file in glob.glob("../data/train/set1/*.jpg")]
    
    scoreMapList = []
    cornersList = []
    for img in imgs:
        tempScoreMap, tempcorners = corner_score(img) 
        scoreMapList.append(tempScoreMap)
        cornersList.append(tempcorners)
    
    for corners, img in zip(cornersList, copy.deepcopy(imgs)):
        for corner in corners:
            cv2.circle(img,tuple(corner),2,(255,0,0), -1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    for scoreMap,  corners, img in zip(scoreMapList, cornersList, copy.deepcopy(imgs)):
        x,y,img = applyANMS(scoreMap, corners, img, 200)
        cv2.imshow("anms", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
if __name__ == '__main__':
    main()
 
