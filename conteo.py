import cv2
from yolov8 import YOLOv8
video_path = 'video/test1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error al abrir el video.")
    exit()
model_path = "models/best.onnx"
yolov8_detector = YOLOv8(model_path, conf_thres=0.5, iou_thres=0.5)
cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
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

