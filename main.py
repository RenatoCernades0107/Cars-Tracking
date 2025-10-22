import cv2
from ultralytics import YOLO 
import numpy as np
from setline import set_line_from_video

# Thanks SrujanPR Vehicle-Detection-and-Counter-using-Yolo11

cap = cv2.VideoCapture("videos/videosample1.mp4")
frame_width = int(cap.get(3))  # Width
frame_height = int(cap.get(4))  # Height
fps = int(cap.get(cv2.CAP_PROP_FPS))

model = YOLO('yolo11l.pt',  verbose=False)
class_list = model.names

start, end, points = set_line_from_video("videos/videosample1.mp4")

counter = 0
crossed_ids = set()

FRAME_SKIP = 10
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Saltar este frame
    
    # Draw Red Line
    cv2.line(frame, start, end, (0, 0, 255), 3) 

    # Run YOLO Tracking on the Video Frame
    results = model.track(frame, persist=True, classes=[1,2,3,5,6,7], verbose=False)

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Get Bounding Boxes, Class Indices, and Track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        # Loop Through Detections
        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]

            # Draw Detection Box & ID
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if Object Crossed the Line defined by the two points
            c = np.array([cx - points[0][0], cy - points[0][1]])
            u = np.array([points[1][0] - points[0][0], points[1][1] - points[0][1]])
            p = (np.dot(c, u) / np.linalg.norm(u*u)) * u
            z = c - p
            d = np.cross(u, z)
            if d > 0 and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                counter += 1



    # Display the Object Counts on the Frame
    cv2.putText(frame, f"Total Count: {counter}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()