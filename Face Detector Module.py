import cv2
import mediapipe as mp

import time

class FaceDetector():
    def __init__(self,minDetectionConfidence = 0.5):
        self.minDetectionConfid = minDetectionConfidence
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfid)



    def findFaces(self,img,draw=True):
        self.img_resize = cv2.resize(img, (720,480), fx=0.5, fy=0.5)

        self.imgRGB = cv2.cvtColor(self.img_resize,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(self.imgRGB)
        # print(self.results)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img_resize,detection )
                # print(id,detection)
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box
                ih,iw,ic = self.img_resize.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                    int(bboxC.width * iw), int(bboxC.height * ih)
                
                if draw:
                    img = self.fancyDraw(self.img_resize,bbox)
                
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

        return self.img_resize,bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt= 1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, (255, 0, 255), rt)
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)
        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)
        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)
        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)
        return img

    

def main():
    cap = cv2.VideoCapture(r'C:\Users\deepesh.sherwal\Desktop\All data at Desktop\deepeshsherwal-wms-uk-dispatch-be6146ac8088\2.mp4')
    pTime = 0

    detector = FaceDetector()

    while True:
        success,img = cap.read()
        img,bboxs = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),2)

        cv2.imshow("Video Slow-mo",img)
        cv2.waitKey(1)
    

if __name__ == '__main__':
    main()
