import cv2
import mediapipe as mp

import time

cap = cv2.VideoCapture(r'C:\Users\deepesh.sherwal\Desktop\All data at Desktop\deepeshsherwal-wms-uk-dispatch-be6146ac8088\1.mp4')

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()


pTime = 0
while True:
    success,img = cap.read()
    img_resize = cv2.resize(img, (720,480), fx=0.5, fy=0.5)

    imgRGB = cv2.cvtColor(img_resize,cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img_resize,detection )
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih,iw,ic = img_resize.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img_resize, bbox, (255, 0, 255), 2)
            cv2.putText(img_resize, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img_resize,f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,3,(0,255,0),2)

    cv2.imshow("Video Slow-mo",img_resize)
    cv2.waitKey(1)