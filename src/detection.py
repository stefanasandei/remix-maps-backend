import cv2
import numpy as np

def haar(filename):
    img = cv2.imread(filename)

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5,5), 0)
    dilated = cv2.dilate(blur, np.ones((3,3)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel) 

    car_cascade = cv2.CascadeClassifier("./data/cars.xml")
    cars = car_cascade.detectMultiScale(closing, 1.1, 1)

    cnt = 0
    for (x,y,w,h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cnt += 1

    cv2.imwrite(filename, img)

    return cnt

def yolo(filename):
    config_file = './data/yolov3.cfg'
    weights_file = './data/yolov3.weights'
    net = cv2.dnn.readNet(config_file, weights_file)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    image = cv2.imread(filename)
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5: 
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    for i in indices:
        x, y, w, h = boxes[i]
        label = str(class_ids[i])
        confidence = confidences[i]
        color = (0, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(filename, image)

    return len(indices)

def detectCars(filename):
    method = "yolo"

    if method == "haar":
        cnt = haar(filename)
    elif method == "yolo":
        cnt = yolo(filename)

    return cnt

