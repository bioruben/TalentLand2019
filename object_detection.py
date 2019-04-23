import os
import cv2
from imageai.Detection import VideoObjectDetection


#Setup
workPath = os.getcwd()
inputCamera = cv2.VideoCapture(0) 

#Detector
detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(workPath , "yolo.h5"))
detector.loadModel()

videoPath = detector.detectObjectsFromVideo(camera_input=inputCamera,
                                output_file_path=os.path.join(workPath, "camera_detectection")
                                , frames_per_second=2, return_detected_frame=True, log_progress=True)
print(videoPath)
