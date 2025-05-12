import cv2
import numpy as np

#YOLO Setup
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
class_names_path = "coco.names"

with open(class_names_path, "r") as f:
    yolo_classes = f.read().strip().split("\n")

yolo_net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def detect_yolo(frame, target_class, conf_threshold=0.4, nms_threshold=0.4):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    layer_names = yolo_net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
    layer_outputs = yolo_net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if indices is None or len(indices) == 0:
        return []

    results = []
    for i in indices.flatten():
        target_class = "motorbike" if target_class == "motorcycle" else target_class
        if yolo_classes[class_ids[i]] == target_class:
            results.append((boxes[i], confidences[i], class_ids[i]))
    return results

def match_color(segmented_frame, target_color, match_threshold=0.3):
    if segmented_frame is None or segmented_frame.size == 0:
        return False
    
    hsv = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [((0, 50, 50), (10, 255, 255)), ((170, 50, 50), (180, 255, 255))],
        "green": [((35, 50, 50), (85, 255, 255))],
        "blue": [((100, 50, 50), (140, 255, 255))],
        "yellow": [((25, 50, 50), (35, 255, 255))],
        "orange": [((10, 50, 50), (25, 255, 255))],
        "purple": [((140, 50, 50), (160, 255, 255))],
        "white": [((0, 0, 200), (180, 50, 255))],
        "black": [((0, 0, 0), (180, 255, 50))],
        "gray": [((0, 0, 50), (180, 50, 200))],
    }

    if target_color not in color_ranges:
        return False

    mask = None
    for lower, upper in color_ranges[target_color]:
        current_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

    color_pixels = cv2.countNonZero(mask)
    total_pixels = segmented_frame.shape[0] * segmented_frame.shape[1]
    color_ratio = color_pixels / total_pixels

    return color_ratio >= match_threshold

def process_image_with_classes(image, class_color_list):
    matched_boxes = []

    for target_color, target_class in class_color_list:
        detections = detect_yolo(image, target_class)
        if not detections:
            return "Not found"

        valid_detections = []
        for box, conf, class_id in detections:
            x, y, w, h = box
            # Clamp coordinates to image dimensions
            height, width = image.shape[:2]
            x = max(0, x)
            y = max(0, y)
            x_end = min(x + w, width)
            y_end = min(y + h, height)

            # Skip invalid boxes
            if x_end <= x or y_end <= y:
                continue

            roi = image[y:y+h, x:x+w]

            # Extra check: make sure ROI is not empty
            if roi.size == 0:
                continue

            if target_color == "none":
                valid_detections.append((box, yolo_classes[class_id]))
            else:
                if match_color(roi, target_color):
                    valid_detections.append((box, yolo_classes[class_id]))

        # If no valid detections for this pair, return "Not found"
        if not valid_detections:
            return "Not found"

        matched_boxes.extend(valid_detections)

    # Draw all matched boxes
    for (x, y, w, h), label in matched_boxes:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image


