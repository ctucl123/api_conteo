import cv2
from yolov8 import YOLOv8
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import numpy as np

class ByteTrackConfig:
    def __init__(self):
        self.track_thresh = 0.5  
        self.track_buffer = 30  
        self.match_thresh = 0.8 
        self.mot20 = False
tracker_cfg = ByteTrackConfig()
tracker = BYTETracker(tracker_cfg, frame_rate=30)

video_path = 'video/test1.mp4'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error al abrir el video.")
    exit()
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
contador_objetos = 0
linea_y = 900 
last_ids = []
while cap.isOpened():


    if cv2.waitKey(1) == ord('q'):
        break

    try:
   
        ret, frame = cap.read()
        if not ret:
            break
    except Exception as e:
        print(e)
        continue
    boxes, scores, class_ids = yolov8_detector(frame)

    combined_img = yolov8_detector.draw_detections(frame)
    cv2.imshow("Detected Objects", combined_img)

