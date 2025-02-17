import cv2
import torch
from models.yolo import Model

def detect_and_show(weights_path, video_path):
    # Load model
    cfg = "yolov5/models/yolov5s.yaml"  # Path to the configuration file used during training
    model = Model(cfg)
    model.load_state_dict(torch.load(weights_path))
    model.conf = 0.25  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1)
    model.classes = None  # (optional list) filter by class, i.e., = [0, 15, 16] for persons, cats and dogs

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Make detections
        results = model(frame)

        # Draw results on frame
        for *box, conf, cls in results.xyxy[0]:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 2)
            frame = cv2.putText(frame, label, (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Show frame
        cv2.imshow('frame', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

weights_path = 'runs/train/exp2/weights/best.pt'
video_path = '/home/mr16294/Desktop/PycharmProjects/pythonProject/output.mp4'
detect_and_show(weights_path, video_path)

