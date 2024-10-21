import cv2
import numpy as np
from vidgear.gears import CamGear
from ultralytics import YOLO
import json

# Load YOLOv8 model
model = YOLO('yolo11s.pt')  # Use the correct YOLOv8 model weights

# Load the classes
classes_path = 'data.yaml'
with open(classes_path, 'r', encoding='utf-8') as f:
    classes = [line.strip() for line in f.readlines()]

# Use the camera (or you can use a video file)
stream = CamGear(source=1).start()  # Change to your video file path if needed

# Initialize metrics
true_positives = []
false_positives = []
false_negatives = []
frames = 0
drone_position = np.array([0, 0, 50])  # x, y, height of the drone
object_height = 1.8  # Replace with actual object height if known

# Prepare a list to store detected objects
detected_objects = []

while True:
    frame = stream.read()
    if frame is None:
        break

    height, width = frame.shape[:2]

    # Perform detection
    results = model(frame)

    # Count the number of people detected
    person_count = 0
    detected_class_ids = []

    for result in results:
        boxes = result.boxes  # Accessing boxes directly

        for box in boxes:
            # Access the bounding box coordinates and class ID
            x1, y1, x2, y2 = box.xyxy[0]  # Get the box coordinates
            cls = int(box.cls[0])  # Access the class ID

            if cls == 0:  # If the detected class is 'person'
                detected_class_ids.append(cls)
                person_count += 1

                # Calculate the object's position
                object_position = np.array([(x1 + x2) / 2, (y1 + y2) / 2, object_height])

                # Calculate distance from the drone
                distance = np.linalg.norm(object_position - drone_position)
                print(f'Detected object at ({(x1 + x2) / 2}, {(y1 + y2) / 2}) with distance: {distance:.2f}m')

                # Store detected object information
                detected_objects.append({
                    'class_id': cls,
                    'coordinates': {'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)},
                    'distance': distance
                })

                # Draw the rectangle and distance on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                cv2.putText(frame, f'Distance: {distance:.2f}m', (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Here you can assume a ground truth; this is an example
    ground_truth_class_ids = [0]  # Assume you know how many persons are in the frame

    # Calculate metrics
    frames += 1
    tp = len(set(detected_class_ids) & set(ground_truth_class_ids))
    fp = len(set(detected_class_ids) - set(ground_truth_class_ids))
    fn = len(set(ground_truth_class_ids) - set(detected_class_ids))

    true_positives.append(tp)
    false_positives.append(fp)
    false_negatives.append(fn)

    # Calculate precision, recall, F1 score, and accuracy
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    accuracy = (tp) / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Accuracy calculation

    # Display metrics on frame
    cv2.putText(frame, f'Count: {person_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Precision: {precision:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Recall: {recall:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'F1 Score: {f1:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, f'Accuracy: {accuracy:.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display metrics in the terminal
    print(f'Frame: {frames}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}, Accuracy: {accuracy:.2f}')

    cv2.imshow("CoSI Rescue Vision", frame)

    # Stop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save detected objects to JSON file
output_file = 'detected_objects.json'
with open(output_file, 'w') as f:
    json.dump(detected_objects, f, indent=4)

# Calculate average metrics
avg_precision = np.mean(true_positives) / (np.mean(true_positives) + np.mean(false_positives)) if (np.mean(true_positives) + np.mean(false_positives)) > 0 else 0
avg_recall = np.mean(true_positives) / (np.mean(true_positives) + np.mean(false_negatives)) if (np.mean(true_positives) + np.mean(false_negatives)) > 0 else 0
avg_f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0
avg_accuracy = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives) + np.sum(false_negatives)) if (np.sum(true_positives) + np.sum(false_positives) + np.sum(false_negatives)) > 0 else 0

print(f'Average Precision: {avg_precision:.2f}')
print(f'Average Recall: {avg_recall:.2f}')
print(f'Average F1 Score: {avg_f1:.2f}')
print(f'Average Accuracy: {avg_accuracy:.2f}')

stream.release()
cv2.destroyAllWindows()