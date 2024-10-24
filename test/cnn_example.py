import cv2
import time
from ultralytics import YOLO
import numpy as np
start_time = time.time()

model = YOLO('models/best.pt')
frame = cv2.imread('images/prueba.png')
height, width, channels = frame.shape
print(f"Resolución de la imagen antes: {width}x{height}")
frame_resized = cv2.resize(frame, (1280, 720))
height, width, channels = frame_resized.shape
print(f"Resolución ahora: {width}x{height}")
results = model(frame_resized,verbose=False)
dets = []
for result in results[0].boxes:
    x1, y1, x2, y2 = result.xyxy[0]
    conf = result.conf[0]
    class_id = int(result.cls[0])
    cv2.rectangle(frame_resized, (int(x1), int(y1) - 30), (int(x2)-60, int(y1)), (0, 0, 0), -1)#rectangulo negro para ver el id
    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (255, 20, 147), 2)#bounding box
    cv2.putText(frame_resized, f'ID: {class_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2) #texto del ID
    print(f"cordenadas : ({x1} , {y1}) - ({x2} , {y2})")
    print(f'probabilidad: {conf}')
    print(f'id: {class_id}')
    dets.append([x1, y1, x2, y2, conf])
    print(f"dato antes de convertir {dets}")
dets = np.array(dets, dtype=np.float32)
print("dato despues de convertir")
print(dets)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de ejecución: {elapsed_time:.4f} segundos")
cv2.imshow('YOLOv8 Detections', frame_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()