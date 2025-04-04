import cv2
import numpy as np
import time

# Initialize video capture
cap = cv2.VideoCapture("Traffic3.mov")

# Create Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

# Traffic light states and durations
TRAFFIC_LIGHT_STATES = ["red", "green", "yellow"]
STATE_DURATIONS = {"green": 10, "yellow": 5, "red": 15}  # Time in seconds
traffic_state_index = 1  # Start with green
traffic_state = TRAFFIC_LIGHT_STATES[traffic_state_index]
last_change_time = time.time()  # Track last state change time
frozen_frame = None  # Store frozen frame when red light is active
resume_detection = False  # Flag to resume detection when red is active
red_light_active = False  # Track if red light is active

# Variables for ROI selection
roi_points = []
roi_selected = False

# Mouse callback function for selecting ROI
def select_roi(event, x, y, flags, param):
    global roi_points, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:  # Capture four points
        roi_points.append((x, y))
    if len(roi_points) == 4:
        roi_selected = True

cv2.namedWindow("Traffic Monitoring")
cv2.setMouseCallback("Traffic Monitoring", select_roi)

# Function to draw traffic light
def draw_traffic_light(frame, state, position):
    x, y, w, h = position[0], position[1], 40, 100
    colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    if state == "green":
        colors[2] = (0, 255, 0)
    elif state == "yellow":
        colors[1] = (0, 255, 255)
    elif state == "red":
        colors[0] = (0, 0, 255)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    for i, color in enumerate(colors):
        center = (x + w // 2, y + 20 + i * 30)
        cv2.circle(frame, center, 10, color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (550, 280))
    h, w = frame.shape[:2]

    # Maintain frozen frames for quadrants
    if frozen_frame is None:
        frozen_frame = frame.copy()

    # Create a blank combined frame
    combined_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    if traffic_state in ["green", "yellow"]:
        combined_frame[:h, :w] = frame  # Q1 (Playing)
        combined_frame[h:, w:] = frame  # Q4 (Playing)
        combined_frame[:h, w:] = frozen_frame  # Q2 (Paused)
        combined_frame[h:, :w] = frozen_frame  # Q3 (Paused)
    else:  # Red Light Active
        combined_frame[:h, :w] = frozen_frame  # Q1 (Paused)
        combined_frame[h:, w:] = frozen_frame  # Q4 (Paused)
        combined_frame[:h, w:] = frame  # Q2 (Playing)
        combined_frame[h:, :w] = frame  # Q3 (Playing)

    # Draw traffic lights in diagonals
    draw_traffic_light(combined_frame[:h, :w], traffic_state, (20, 20))
    draw_traffic_light(combined_frame[h:, w:], traffic_state, (20, 20))

    if traffic_state != "red":
        draw_traffic_light(combined_frame[:h, w:], "red", (20, 20))
        draw_traffic_light(combined_frame[h:, :w], "red", (20, 20))
    else:
        draw_traffic_light(combined_frame[:h, w:], "green", (20, 20))
        draw_traffic_light(combined_frame[h:, :w], "green", (20, 20))

    # Display timer
    remaining_time = STATE_DURATIONS[traffic_state] - int(time.time() - last_change_time)
    cv2.putText(combined_frame, f"Timer: {remaining_time}s", (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Traffic Monitoring", combined_frame)

    # Check for traffic light state transitions
    current_time = time.time()
    if current_time - last_change_time >= STATE_DURATIONS[traffic_state]:
        traffic_state_index = (traffic_state_index + 1) % len(TRAFFIC_LIGHT_STATES)
        traffic_state = TRAFFIC_LIGHT_STATES[traffic_state_index]
        last_change_time = current_time
        frozen_frame = frame.copy()  # Capture a new frozen frame when light changes

    # Detect key press
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
