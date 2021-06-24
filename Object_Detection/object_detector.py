import cv2
import numpy as np

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, color):
    label = (str(classes[class_id]) + " " + str(format(confidence, '.2f'))).capitalize()
    font_scale = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    (txt_w, txt_h), base_line = cv2.getTextSize(label, font, font_scale, 2)
    cv2.rectangle(img, (x - 1, y - txt_h - base_line - 2), (x + txt_w, y), color, cv2.FILLED)
    cv2.putText(img, label, (x, y - base_line), font, font_scale, (255, 255, 255), 2)

def detect_objects(image):
    global classes
    objects, positions, coords = [], [], []
    image = cv2.resize(image, (416, 416))

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392   # 1.0/255.0

    with open("./Object_Detection/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    net = cv2.dnn.readNet('./Object_Detection/yolov3.cfg', './Object_Detection/yolov3.weights')
    output_layers = net.getUnconnectedOutLayersNames()
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in indices:
        i = i[0] # indices has dimension Cx1
        x, y, w, h = boxes[i]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h), COLORS[class_id])
        objects.append(classes[class_ids[i]])
        coords.append([x, y, w, h])

        center_x = x + (w / 2)
        if center_x < 133:
            positions.append('left')
        elif center_x > 266:
            positions.append('right')
        else:
            positions.append('front')

    # cv2.imshow('Frame', image)
    # cv2.waitKey(0)

    return objects, positions, image


classes = None
objects, positions, coords = [], [], []

# image = cv2.imread("dog.jpg")
# objects, coords, positions = detect_objects(image)