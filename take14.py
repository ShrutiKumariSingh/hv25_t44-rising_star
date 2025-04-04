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
last_change_time = time.time()

# Object tracking for speed detection
object_tracker = {}
speed_dict = {}
frame_time = 1 / 30  # Assuming 30 FPS
pixels_to_meters = 0.05  # Adjust according to real-world scale

# Store last valid frame
frozen_frame = None

# Function to calculate speed
def calculate_speed(object_id, new_position):
    if object_id in object_tracker:
        old_position = object_tracker[object_id]
        distance_pixels = np.linalg.norm(np.array(new_position) - np.array(old_position))
        speed = (distance_pixels * pixels_to_meters) / frame_time  # Convert to m/s
        speed_dict[object_id] = round(speed, 2)
    object_tracker[object_id] = new_position

# Function to process movement and speed detection
def process_quadrant(frame):
    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w // 2, y + h // 2)
            calculate_speed(i, center)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            speed_text = f"{speed_dict.get(i, 0)} m/s"
            cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    return frame

# Function to draw traffic light indicators
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

    # Maintain frozen frame when red light is active
    if frozen_frame is None:
        frozen_frame = frame.copy()

    # Create the combined frame
    combined_frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

    if traffic_state in ["green", "yellow"]:
        q1 = process_quadrant(frame.copy())  # Q1 (Playing)
        q4 = process_quadrant(frame.copy())  # Q4 (Playing)
        q2 = frozen_frame  # Q2 (Paused)
        q3 = frozen_frame  # Q3 (Paused)
    else:  # Red Light Active
        q1 = frozen_frame  # Q1 (Paused)
        q4 = frozen_frame  # Q4 (Paused)
        q2 = process_quadrant(frame.copy())  # Q2 (Playing)
        q3 = process_quadrant(frame.copy())  # Q3 (Playing)

    # Arrange the quadrants into a final display frame
    combined_frame[:h, :w] = q1
    combined_frame[h:, w:] = q4
    combined_frame[:h, w:] = q2
    combined_frame[h:, :w] = q3

    # Draw traffic lights on respective quadrants
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

    # Show final output
    cv2.imshow("Traffic Monitoring", combined_frame)

    # Traffic light transition logic
    current_time = time.time()
    if current_time - last_change_time >= STATE_DURATIONS[traffic_state]:
        traffic_state_index = (traffic_state_index + 1) % len(TRAFFIC_LIGHT_STATES)
        traffic_state = TRAFFIC_LIGHT_STATES[traffic_state_index]
        last_change_time = current_time
        frozen_frame = frame.copy()  # Capture a new frozen frame when light changes

    # Exit on 'q' key
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
