import cv2 as cv
from handDetectionModule import HandDetector
import mediapipe as mp
import math
def relativeDistance(h1,h2,i,j):
    dist1=math.dist(h1[i],h1[j])
    dist2=math.dist(h2[i],h2[j])
    standardDist1=(math.dist(h1[0],h1[5])+math.dist(h1[0],h1[17])+math.dist(h1[0],h1[9])+math.dist(h1[0],h1[13]))/4
    standardDist2=(math.dist(h2[0],h2[5])+math.dist(h2[0],h2[17])+math.dist(h2[0],h2[9])+math.dist(h2[0],h2[13]))/4
    if (dist1/standardDist1)-(dist2/standardDist2)<0.0001*(standardDist2+standardDist1)/2:
        return True
def getimg():
    cap=cv.VideoCapture(0)
    imagelist=[]
    count=0
    obj=HandDetector()
    while count<1:
        sucess,im=cap.read()
        img=cv.flip(im,1)
        if sucess==False:
            print("failed to capture image")
        obj.detect(img)
        cv.imshow("image1",img)
        if cv.waitKey(1) & 0xff==ord('p'):
            imagelist.append(img)
            count+=1
    cap.release()
    cv.destroyAllWindows()
    return imagelist
def handcompartor(img01,img02):
    obj=HandDetector()
    img1=cv.resize(img01,(500,500))
    img2=cv.resize(img02,(500,500))
    h1=obj.getPoints(img1)
    h2=obj.getPoints(img2)    
    matchedpoints=0
    for i in range(0,20):
        print(f"h1: {h1[i]} and h2: {h2[i]}")
        if i==4 or i==8 or i==12 or i==16:
            continue
        if relativeDistance(h1,h2,i,i+1):
                matchedpoints+=1
    if relativeDistance(h1,h2,0,5):
        matchedpoints+=1
    if relativeDistance(h1,h2,0,17):
                matchedpoints+=1
    probabilityMatching=(matchedpoints/18)*100
    return probabilityMatching

if __name__=='__main__':
    image1=getimg()
    image2=getimg()
    h=handcompartor(image1[0],image2[0])
    if h>80:
        print(f"hand matched and prbalilty is {h}")
    else:
        print(f"hand not matched and probabilty is {h}")
        
                
                