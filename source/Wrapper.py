#!/usr/bin/evn python

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


def corner_score(img):
    img  = cv2.GaussianBlur(img,(3,3),0)
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    
    minVal = 0.01* dst.max()
    
    cScoreMap = np.where(dst <= minVal, 0, dst)
    
    
    x, y  = np.where(cScoreMap > 0)
    
    corners = np.hstack((y.reshape(-1,1),x.reshape(-1,1)))
    
    return cScoreMap, corners

def main():
    
    imgs = [cv2.imread(file) for file in glob.glob("../data/train/set1/*.jpg")]
    
    scoreMapList = []
    cornersList = []
    for img in imgs:
        tempScoreMap, tempcorners = corner_score(img) 
        scoreMapList.append(tempScoreMap)
        cornersList.append(tempcorners)
    
    for corners, img in zip(cornersList, imgs):
        for corner in corners:
            cv2.circle(img,tuple(corner),2,(255,0,0), -1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    
if __name__ == '__main__':
    main()
 
