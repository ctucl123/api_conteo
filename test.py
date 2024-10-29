import cv2
from yolov8 import YOLOv8
video_path = 'video/test1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video.")
    exit()

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
    cv2.imshow("Detected Objects", frame)
