import cv2 as cv
import mediapipe as mp
import time
startTime=0

class HandDetector:
    def __init__(self,mode=False,detectionCon=0.5,maxHands=2,trackCon=0.5):
        self.mode=mode
        self.detectionCon=detectionCon
        self.maxHands=maxHands
        self.trackCon=trackCon
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands()
        self.mpDraw=mp.solutions.drawing_utils
    def detect(self,img):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results=self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
             self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
    def getPoints(self,img):
        landmarksHand=[]
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results=self.hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id,points in enumerate(handLms.landmark):
                    h,w,c=img.shape
                    landmarksHand.append((int(points.x*w),int(points.y*h)))
        return landmarksHand
if __name__=="__main__":
    handDetect=HandDetector()
    cap=cv.VideoCapture(0)
    while True:
        sucess,img=cap.read()
        if sucess==True:
            handDetect.detect(img)
            endTime=time.time()
            fps=1/(endTime-startTime)
            startTime=endTime
            cv.putText(img,str(int(fps)),(70,50),cv.FONT_HERSHEY_COMPLEX,3,(160,56,24),3)
            cv.imshow("Detection",img)
            if cv.waitKey(1)&0xff==ord("p"):
                break
        else:
            print("Error in getting photos")
        handDetect.getPoints(img)
    cap.release()
    cv.destroyAllWindows()

