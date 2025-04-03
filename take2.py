import cv2
import matplotlib.pyplot as plt
import numpy as np

# Initialize video capture
cap = cv2.VideoCapture("Traffic.mp4")

# Create Background Subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

def draw_traffic_light(frame, state):
    """Draw a traffic light at the top-left corner of the frame."""
    x, y, w, h = 20, 20, 40, 100  # Position and size of traffic light box
    colors = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]  # Default (all off)
    
    if state == "green":
        colors[2] = (0, 255, 0)  # Green on
    elif state == "yellow":
        colors[1] = (0, 255, 255)  # Yellow on
    elif state == "red":
        colors[0] = (0, 0, 255)  # Red on
    
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)  # Light box
    
    for i, color in enumerate(colors):
        center = (x + w // 2, y + 20 + i * 30)
        cv2.circle(frame, center, 10, color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))  # Resize for better processing

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)

    # Remove noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count and draw bounding boxes around moving objects
    object_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small movements
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            object_count += 1  # Increase object count

    # Determine traffic light state
    if object_count <= 3:
        traffic_state = "green"
    elif object_count <= 6:
        traffic_state = "yellow"
    else:
        traffic_state = "red"
    
    # Draw traffic light
    draw_traffic_light(frame, traffic_state)

    # Display object count on frame
    cv2.putText(frame, f"Moving Objects: {object_count}", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Display results using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.0000001)
    plt.clf()

    # Print the object count to the terminal/output
    print(f"Number of moving objects: {object_count}")

    # Pause video when 10 moving objects are detected
    if object_count >= 10:
        print("Paused: 10 objects detected. Press any key to continue...")
        input()  # Wait for user input to resume

cap.release()
cv2.destroyAllWindows()