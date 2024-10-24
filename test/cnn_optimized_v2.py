import cv2
import torch
from ultralytics import YOLO
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
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
last_ids = set()  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Fin del video.")
        break


    results = model(frame, verbose=False)


    boxes = results[0].boxes
    if boxes.shape[0] > 0:
        dets = np.concatenate(
            [boxes.xyxy.cpu().numpy(), boxes.conf.view(-1, 1).cpu().numpy()], axis=1
        ).astype(np.float32)
        

        online_targets = tracker.update(dets, frame.shape, frame.shape)
        
        for target in online_targets:
            tlwh = target.tlwh  
            track_id = target.track_id
            x1, y1, w, h = tlwh
            cy = int(y1 + h / 2) 
        
            if linea_y - 5 < cy < linea_y + 15 and track_id not in last_ids:
                last_ids.add(track_id) 
                contador_objetos += 1
                print(f'Contador: {contador_objetos} - ID: {track_id}')


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
