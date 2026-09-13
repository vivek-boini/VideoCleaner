import cv2
import time
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video = "input/test.mp4"

target_classes = {
    0: "person",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe"
}

cap = cv2.VideoCapture(video)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")
print("Checking 2 frames per second")

frame_step = max(1, int(fps / 2))

frame_number = 0
checked = 0
detections = 0

start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_step == 0:
        results = model(
            frame,
            device="cpu",
            classes=list(target_classes.keys()),
            conf=0.35,
            verbose=False
        )

        checked += 1

        detected_names = []

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id in target_classes:
                detected_names.append(
                    f"{target_classes[class_id]} ({confidence:.2f})"
                )

        if detected_names:
            detections += 1
            timestamp = frame_number / fps

            print(
                f"{timestamp:.2f}s - "
                + ", ".join(detected_names)
            )

    frame_number += 1

cap.release()

elapsed = time.time() - start_time

print()
print("TEST COMPLETE")
print(f"Frames checked: {checked}")
print(f"Frames with detections: {detections}")
print(f"Detection time: {elapsed:.2f} seconds")
print(f"Average time per checked frame: {elapsed / checked:.3f} seconds")