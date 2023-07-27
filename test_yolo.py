import cv2
import numpy as np

vid = cv2.VideoCapture("http://82.76.145.217/cgi-bin/faststream.jpg?stream=half&fps=30")
  
while(True):
    ret, image = vid.read()
  
    config_file = './data/yolov3.cfg'
    weights_file = './data/yolov3.weights'
    net = cv2.dnn.readNet(config_file, weights_file)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416*2, 416*2), swapRB=True, crop=False)

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

    cv2.imshow('camera', image)

    if cv2.waitKey(33) == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()
