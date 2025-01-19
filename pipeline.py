import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import base64
import tempfile
import gdown

import logging

logging.getLogger('tensorflow').setLevel(logging.ERROR)

def download_file_from_google_drive(file_id, destination):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, destination, quiet=False)
    except Exception as e:
        st.write(f"Failed to download {destination} using gdown. Error: {e}")
        st.write("Trying to download using curl...")
        os.system(f"curl -L {url} -o {destination}")

FILES_TO_DOWNLOAD = {
    "frozen_inference_graph.pb": "1Uugg-uKvHn-0jeGS8-7yO29IRkaxoitK",
    "yolov3-tiny.weights": "1sjHRq9pBQgVEmmdR2ZZxJiONY5KbYXd7"
}

for filename, file_id in FILES_TO_DOWNLOAD.items():
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}...")
        download_file_from_google_drive(file_id, filename)


# YOLO model setup
weights_path = "yolov3-tiny.weights"
config_path = "yolov3-tiny.cfg"
class_names_path = "coco.names"


with open(class_names_path, "r") as f:
    yolo_classes = f.read().strip().split("\n")


yolo_net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# Mask R-CNN setup
frozen_model_path = "frozen_inference_graph.pb"
label_map_path = "mscoco_label_map.pbtxt"


def load_label_map(label_map_path):
    category_index = {}
    category_id, category_name = None, None
    with open(label_map_path, 'r') as f:
        for line in f:
            if 'id:' in line:
                category_id = int(line.split(':')[-1].strip())
            if 'name:' in line:
                category_name = line.split(':')[-1].strip().replace("'", "")
            if category_id is not None and category_name is not None:
                category_index[category_id] = category_name
                category_id, category_name = None, None
    return category_index


mask_rcnn_classes = load_label_map(label_map_path)


def load_mask_rcnn_model(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


mask_rcnn_net = load_mask_rcnn_model(frozen_model_path)


def detect_yolo(frame, target_class, conf_threshold=0.3, nms_threshold=0.4):
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
        return None


    for i in indices.flatten():
        if yolo_classes[class_ids[i]] == target_class:
            return boxes[i], confidences[i], class_ids[i]


    return None


# def scale_mask(mask, box, frame_shape):
#     x, y, w, h = box
#     mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
#     mask_binary = (mask_resized > 0.5).astype(np.uint8)  # Threshold to binary mask
#     scaled_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
#     scaled_mask[y:y + h, x:x + w] = mask_binary
#     return scaled_mask

def scale_mask(mask, box, frame_shape):
    """
    Scales the predicted mask to the bounding box dimensions and applies it to the frame.


    Args:
        mask (np.ndarray): The predicted mask from Mask R-CNN (multi-channel or single-channel).
        box (tuple): The bounding box coordinates (x, y, w, h).
        frame_shape (tuple): The shape of the original frame (height, width, channels).


    Returns:
        np.ndarray: A binary mask of the same shape as the frame, applied within the bounding box.
    """
    x, y, w, h = box
    height, width = frame_shape[:2]

    print(f"Mask shape before resizing: {mask.shape}")
    print(f"Frame dimensions: {frame_shape}")
    print(f"Bounding box dimensions: {box}")
    
    # Handle multi-channel masks (e.g., mask with shape (H, W, C))
    if mask.ndim > 2:
        # Collapse the multi-channel mask into a single channel by taking the maximum value
        mask = np.max(mask, axis=-1)


    # Resize the mask to the bounding box size
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    print(f"Mask shape after resizing: {mask_resized.shape}")


    # Convert to binary (0 or 1)
    mask_binary = (mask_resized > 0.5).astype(np.uint8)


    # Create an empty binary mask for the full frame
    scaled_mask = np.zeros((height, width), dtype=np.uint8)


    # Clip bounding box coordinates to stay within frame boundaries
    x_start, x_end = max(x, 0), min(x + w, width)
    y_start, y_end = max(y, 0), min(y + h, height)


    # Ensure mask dimensions align with the clipped bounding box
    mask_clipped = mask_binary[:y_end - y_start, :x_end - x_start]


    # Assign the clipped mask to the corresponding region in the scaled mask
    scaled_mask[y_start:y_end, x_start:x_end] = mask_clipped


    return scaled_mask


def detect_mask_rcnn(frame, box, detection_graph, sess):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    w = min(frame.shape[1] - x, w)
    h = min(frame.shape[0] - y, h)
    cropped_frame = frame[y:y + h, x:x + w]


    input_image = np.expand_dims(cropped_frame, axis=0)


    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
    classes_tensor = detection_graph.get_tensor_by_name('detection_classes:0')
    masks_tensor = detection_graph.get_tensor_by_name('detection_masks:0')


    boxes, scores, classes, masks = sess.run(
        [boxes_tensor, scores_tensor, classes_tensor, masks_tensor],
        feed_dict={image_tensor: input_image}
    )


    if masks is not None and masks.shape[0] > 0:
        return scale_mask(masks[0], box, frame.shape)
    return None


def match_color(segmented_frame, target_color, match_threshold=30):
    color_ranges = {
        "red": [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],
        "green": [(35, 50, 50), (85, 255, 255)],
        "blue": [(100, 50, 50), (140, 255, 255)],
        "yellow": [(25, 50, 50), (35, 255, 255)],
        "orange": [(10, 50, 50), (25, 255, 255)],
        "purple": [(140, 50, 50), (160, 255, 255)],
        "white": [(0, 0, 200), (180, 50, 255)],
        "black": [(0, 0, 0), (180, 255, 50)],
        "gray": [(0, 0, 50), (180, 50, 200)],
    }


    if target_color not in color_ranges:
        return False


    try:
        hsv_frame = cv2.cvtColor(segmented_frame, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return False


    ranges = color_ranges[target_color]
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)


    for lower, upper in ranges:
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)
        mask |= cv2.inRange(hsv_frame, lower_bound, upper_bound)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    match_percentage = (np.sum(mask > 0) / mask.size) * 100
    return match_percentage >= match_threshold


def process_video(video_path, target_class, target_color=None, use_color_matching=False):
    cap = cv2.VideoCapture(video_path)
    matching_frames = []


    with mask_rcnn_net.as_default():
        with tf.compat.v1.Session(graph=mask_rcnn_net) as sess:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break


                result = detect_yolo(frame, target_class)
                if result:
                    box, confidence, class_id = result
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


                    if use_color_matching and target_color:
                        mask = detect_mask_rcnn(frame, box, mask_rcnn_net, sess)
                        if mask is not None and match_color(mask, target_color):
                            matching_frames.append(frame)
                    else:
                        matching_frames.append(frame)


    cap.release()
    return matching_frames


def download_link(file_path, download_text):
    with open(file_path, "rb") as f:
        file_bytes = f.read()
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:file/mp4;base64,{b64}" download="{os.path.basename(file_path)}">{download_text}</a>'


def main():
    st.title("Video Content Search")


    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    target_class = st.selectbox("Select Target Class", yolo_classes)
    use_color_matching = st.checkbox("Object Color?")
    target_color = None
    if use_color_matching:
        target_color = st.selectbox("Select Target Color", ["red", "green", "blue", "yellow", "orange", "purple", "white", "black", "gray"])


    if st.button("Process"):
        if uploaded_video:
            st.write("Processing video...")
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video.name.split('.')[-1]}")
            video_path = temp_video.name
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())


            matching_frames = process_video(video_path, target_class, target_color, use_color_matching)


            if not matching_frames:
                st.write("No objects found.")
            else:
                output_path = "output_video.mp4"
                height, width, _ = matching_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
                for frame in matching_frames:
                    out.write(frame)
                out.release()


                st.write("Output Video:")
                st.markdown(download_link(output_path, "Download Link"), unsafe_allow_html=True)


                st.write("Matching Frames:")
                cols = st.columns(4)
                for i, frame in enumerate(matching_frames):
                    col = cols[i % 4]
                    with col:
                        st.image(frame, channels="BGR", caption=f"Frame {i+1}")


if __name__ == "__main__":
    tf.compat.v1.disable_eager_execution()
    main()
