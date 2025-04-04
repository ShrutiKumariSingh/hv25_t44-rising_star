import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load four videos
cap1 = cv2.VideoCapture("Traffic.mp4")
cap2 = cv2.VideoCapture("Traffic2.mp4")
cap3 = cv2.VideoCapture("Traffic5.mp4")
cap4 = cv2.VideoCapture("Traffic6.mp4")
caps = [cap1, cap2, cap3, cap4]

# Traffic light control
BASE_DURATIONS = {"green": 10}
state_durations = BASE_DURATIONS.copy()
last_change_time = time.time()
time_extended = False
max_object_threshold = 10

# Light state for each quadrant (Q1, Q2, Q3, Q4)
quadrant_lights = ["red", "red", "red", "red"]
active_quadrant_index = 3  # Start with Q4
quadrant_lights[active_quadrant_index] = "green"

# Frozen frame buffer
frame_buffers = [None] * 4

# Object detection
def detect_objects(frame):
    results = model(frame, verbose=False)[0]
    count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame, count

# Draw traffic light on a frame
def draw_traffic_light(frame, state, position):
    x, y, w, h = position[0], position[1], 40, 100
    colors = [(0, 0, 0)] * 3
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

# Draw LIVE label
def draw_live_label(frame):
    label = "LIVE"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (0, 0, 255)
    dot_radius = 6
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x = frame.shape[1] - text_size[0] - 30
    y = 40
    cv2.putText(frame, label, (x, y), font, font_scale, color, thickness)
    cv2.circle(frame, (x - 15, y - 10), dot_radius, color, -1)

# Main loop
target_size = (400, 300)

while True:
    frames = [None] * 4
    object_counts = []
    processed_frames = []

    # Step 1: Read all frames
    for i in range(4):
        ret, frame = caps[i].read()
        if not ret:
            # Use buffer if video ended
            frame = frame_buffers[i] if frame_buffers[i] is not None else np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, target_size)
            if frame_buffers[i] is None:
                frame_buffers[i] = frame.copy()

        # Use live frame if green, else frozen
        if quadrant_lights[i] == "green":
            frame_buffers[i] = frame.copy()
            frames[i] = frame
        else:
            frames[i] = frame_buffers[i]

    # Step 2: Object detection
    for i in range(4):
        if quadrant_lights[i] == "green":
            detected_frame, count = detect_objects(frames[i])
        else:
            detected_frame = frames[i]
            count = 0
        processed_frames.append(detected_frame)
        object_counts.append(count)

    # Step 3: Extend time if congested
    if not time_extended and object_counts[active_quadrant_index] > max_object_threshold:
        state_durations["green"] += 20
        time_extended = True

    # Step 4: Stack final view
    top_row = np.hstack([processed_frames[0], processed_frames[1]])
    bottom_row = np.hstack([processed_frames[2], processed_frames[3]])
    final_display = np.vstack([top_row, bottom_row])

    # Step 5: Draw traffic lights per quadrant
    draw_traffic_light(final_display[:300, :400], quadrant_lights[0], (20, 20))  # Q1
    draw_traffic_light(final_display[:300, 400:], quadrant_lights[1], (20, 20))  # Q2
    draw_traffic_light(final_display[300:, :400], quadrant_lights[2], (20, 20))  # Q3
    draw_traffic_light(final_display[300:, 400:], quadrant_lights[3], (20, 20))  # Q4

    # Step 6: Draw LIVE on active quadrant
    if quadrant_lights[0] == "green": draw_live_label(final_display[:300, :400])
    if quadrant_lights[1] == "green": draw_live_label(final_display[:300, 400:])
    if quadrant_lights[2] == "green": draw_live_label(final_display[300:, :400])
    if quadrant_lights[3] == "green": draw_live_label(final_display[300:, 400:])

    # Step 7: Show timer
    remaining = state_durations["green"] - int(time.time() - last_change_time)
    cv2.putText(final_display, f"Timer: {remaining}s", (20, final_display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display output
    cv2.imshow("Smart Traffic Monitoring", final_display)

    # Step 8: Switch active quadrant
    if time.time() - last_change_time >= state_durations["green"]:
        active_quadrant_index = (active_quadrant_index + 1) % 4
        quadrant_lights = ["red", "red", "red", "red"]
        quadrant_lights[active_quadrant_index] = "green"
        state_durations = BASE_DURATIONS.copy()
        time_extended = False
        last_change_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()