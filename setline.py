import cv2
def set_line_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))  # Width

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

    if not len(points) == 2:
        print("Error: Please select exactly two points.")
        raise ValueError("Two points are required.")
    else:
        m = (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]) if (points[1][0] - points[0][0]) != 0 else 0
        b = points[0][1] - m * points[0][0]
        print(f"Line equation: y = {m}x + {b}")
        start = (0, int(b))
        end = (frame_width, int(m * frame_width + b))

    return start, end, points