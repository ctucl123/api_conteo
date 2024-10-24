import cv2
import torch
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker,STrack
import numpy as np


model = YOLO('models/best.pt')  
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
contador_objetos = 0
linea_y = 900 
last_ids = []
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break
    results = model(frame)
    dets = []
    for result in results[0].boxes:
        x1, y1, x2, y2 = result.xyxy[0]
        conf = result.conf[0]
        dets.append([x1, y1, x2, y2, conf])
    dets = np.array(dets, dtype=np.float32)
    if len(dets) >0:
        online_targets = tracker.update(dets, frame.shape,frame.shape)
        for target in online_targets:
            tlwh = target.tlwh 
            track_id = target.track_id
            x1, y1, w, h = tlwh
            y2 = y1 + h
            cy = int(y1 + h / 2) 
            if cy > linea_y - 5 and cy < linea_y + 15:
                if track_id in last_ids:
                    pass
                else:
                    last_ids.append(track_id)
                    contador_objetos += 1 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()