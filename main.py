import numpy as np
import cv2

yolo_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


yolo_net = cv2.dnn.readNetFromDarknet("model/yolov3.cfg", "model/yolov3.weights")

yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

np.random.seed(42)
yolo_colors = np.random.uniform(0, 255, size=(len(yolo_classes), 3))

min_confidence = 0.5
nms_threshold = 0.4

# For Detection Objects on Live Camera or Video:
# --------------------------------------------------------
#cap = cv2.VideoCapture(0) # 0 is generally default cam
#cap = cv2.VideoCapture("test_files/video.mp4")


# For Detection Objects on Image:
# --------------------------------------------------------
image_path = "test_files/test_image1.jpg"


while True:

    # For Detection Objects on Live Camera or Video:
    #_, image = cap.read()

    # For Detection Objects on Image:
    image = cv2.imread(image_path)

    height, width = image.shape[0], image.shape[1]
    
    # YOLO için blob oluştur
    yolo_blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(yolo_blob)
    yolo_outs = yolo_net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    
    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > min_confidence:
                # Nesne merkezi ve boyutları
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Köşe koordinatları
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum suppression (çakışan kutuları temizle)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(yolo_classes[class_ids[i]])
            confidence = confidences[i]
            
            color = yolo_colors[class_ids[i]].tolist()
            
            # Dikdörtgen çiz
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # Etiket yaz
            prediction_text = f"{label}: {confidence:.2f}%"
            cv2.putText(image, prediction_text, 
                       (x, y - 10 if y > 20 else y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Sonucu göster
    cv2.imshow("Object Detection", image)
    
    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()