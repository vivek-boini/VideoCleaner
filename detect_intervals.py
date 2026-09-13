import cv2
from ultralytics import YOLO

VIDEO = "input/test.mp4"

SAMPLE_FPS = 2
CONFIDENCE = 0.35
PADDING = 0.5
GAP_LIMIT = 0.75

TARGET_CLASSES = {
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

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

frame_step = max(1, round(fps / SAMPLE_FPS))

detections = []

frame_number = 0
checked = 0

print(f"Video duration: {duration:.2f} seconds")
print(f"FPS: {fps}")
print(f"Sampling: {SAMPLE_FPS} FPS")
print()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_number % frame_step == 0:

        result = model(
            frame,
            device="cpu",
            classes=list(TARGET_CLASSES.keys()),
            conf=CONFIDENCE,
            verbose=False
        )[0]

        detected_names = []

        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id in TARGET_CLASSES:
                detected_names.append(TARGET_CLASSES[class_id])

        checked += 1

        if detected_names:
            timestamp = frame_number / fps
            detections.append(timestamp)

            names = sorted(set(detected_names))

            print(
                f"{timestamp:.2f}s -> "
                + ", ".join(names)
            )

    frame_number += 1

cap.release()

print()
print(f"Frames checked: {checked}")
print(f"Detection points: {len(detections)}")

if not detections:
    print()
    print("No people or animals detected.")
    exit()

intervals = []

start = detections[0]
previous = detections[0]

for timestamp in detections[1:]:

    if timestamp - previous <= GAP_LIMIT:
        previous = timestamp
    else:
        intervals.append((start, previous))
        start = timestamp
        previous = timestamp

intervals.append((start, previous))

final_intervals = []

for start, end in intervals:

    start = max(0, start - PADDING)
    end = min(duration, end + PADDING)

    if final_intervals and start <= final_intervals[-1][1]:
        final_intervals[-1] = (
            final_intervals[-1][0],
            max(final_intervals[-1][1], end)
        )
    else:
        final_intervals.append((start, end))

print()
print("DETECTION INTERVALS")
print("===================")

for start, end in final_intervals:
    print(f"REMOVE: {start:.2f}s -> {end:.2f}s")

print()
print("KEEP INTERVALS")
print("==============")

current = 0

for start, end in final_intervals:

    if current < start:
        print(f"KEEP:   {current:.2f}s -> {start:.2f}s")

    current = end

if current < duration:
    print(f"KEEP:   {current:.2f}s -> {duration:.2f}s")