import cv2
import os
import winsound

# Constants
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = (300, 300)
BEEP_FREQUENCY = 1000  # Frequency for the beep sound

# Flag to keep track of beep state
is_beeping = False

# Function to load class names from a file
def load_class_names(filename):
    try:
        with open(filename, 'rt') as f:
            class_names = f.read().rstrip('\n').split('\n')
        return class_names
    except FileNotFoundError:
        print(f"Class names file '{filename}' not found.")
        return []

# Function to load a pre-trained model
def load_model(weights_path, config_path):
    try:
        net = cv2.dnn.readNet(weights_path, config_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except cv2.error:
        print("Error loading the model. Make sure the paths are correct.")
        return None

# Function to preprocess an input image
def preprocess_image(image, size):
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=size, mean=(127.5, 127.5, 127.5), swapRB=True, crop=False)
    return blob

# Function to run inference on the pre-trained model
def run_inference(net, blob):
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function to get detection bounding box coordinates
def get_detection_coordinates(image, detection):
    h, w = image.shape[:2]
    left = int(detection[3] * w)
    top = int(detection[4] * h)
    right = int(detection[5] * w)
    bottom = int(detection[6] * h)
    return left, top, right, bottom

# Function to draw bounding box and label
def draw_bounding_box_and_label(image, top_left, bottom_right, label):
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), thickness=2)
    cv2.putText(image, label, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness=2)

# Function to display the processed image
def display_output_image(image):
    cv2.imshow("Real-Time Object Detection", image)

# Function to format the confidence score as a percentage string
def get_percentage_string(score):
    return f"{int(score * 100)}%"

# Function to ring the bell
def ring_bell():
    winsound.Beep(BEEP_FREQUENCY, 2000)  # Frequency: 1000 Hz, Duration: 2000 ms

# Function to process detections and draw bounding boxes
def process_and_draw_detections(image, detections, class_names):
    h, w = image.shape[:2]

    for detection in detections[0, 0, :, :]:
        score = detection[2]
        class_id = int(detection[1])

        if score > CONFIDENCE_THRESHOLD and class_id <= len(class_names):
            class_name = class_names[class_id - 1]
            left, top, right, bottom = get_detection_coordinates(image, detection)
            label = f"{class_name.upper()}: {get_percentage_string(score)}"
            draw_bounding_box_and_label(image, (left, top), (right, bottom), label)
            if not is_beeping:
                ring_bell()

def main():
    global is_beeping  # Access the global beep state

    # Initialize video capture from camera (0 for default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video capture.")
        return

    class_names = load_class_names('object_detection_classes_coco.txt')
    if not class_names:
        cap.release()
        return

    net = load_model('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
    if net is None:
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        blob = preprocess_image(frame, IMAGE_SIZE)
        detections = run_inference(net, blob)
        process_and_draw_detections(frame, detections, class_names)
        display_output_image(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit
            break
        elif key == ord(' '):  # Press Spacebar to toggle the beep sound
            if is_beeping:
                winsound.Beep(0, 0)  # Stop the beep
                is_beeping = False
            else:
                winsound.Beep(BEEP_FREQUENCY, 2000)  # Start the beep
                is_beeping = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
