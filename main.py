import cv2

cap = cv2.VideoCapture("videos/videosample1.mp4")

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    raise RuntimeError("Could not open video.")

# Get the first frame of the video 
ret, frame = cap.read()

# Set two points on the frame using mouse clicks
points = []
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (255, 0, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow('Frame', frame)

cv2.imshow('Frame', frame)
cv2.setMouseCallback('Frame', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    

    # Show the frame (for demonstration purposes)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break
    # Process the frame (e.g., object detection, tracking, etc.)

cap.release()