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

cv2.namedWindow('YOLOv8 + ByteTrack', cv2.WINDOW_NORMAL)
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
        class_id = int(result.cls[0])
        dets.append([x1, y1, x2, y2, conf])
    dets = np.array(dets, dtype=np.float32)
    if len(dets) >0:
        online_targets = tracker.update(dets, frame.shape,frame.shape)
        for target in online_targets:
            tlwh = target.tlwh 
            track_id = target.track_id
            x1, y1, w, h = tlwh
            x2 = x1 + w
            y2 = y1 + h
            cx, cy = int(x1 + w / 2), int(y1 + h / 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            if cy > linea_y - 5 and cy < linea_y + 15:
                if track_id in last_ids:
                    pass
                else:
                    last_ids.append(track_id)
                    contador_objetos += 1


    cv2.line(frame, (0, linea_y), (frame.shape[1], linea_y), (0, 0, 255), 2)

    cv2.rectangle(frame, (40, 200), (550, 320), (0, 0, 0), -1)
    cv2.putText(frame, f'Pasajeros: {contador_objetos}', (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)


    cv2.imshow('YOLOv8 + ByteTrack', frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()