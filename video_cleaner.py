import cv2
import subprocess
import os
from ultralytics import YOLO

INPUT_VIDEO = "input/test.mp4"
OUTPUT_VIDEO = "output/cleaned_test.mp4"

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

os.makedirs("output", exist_ok=True)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

frame_step = max(1, round(fps / SAMPLE_FPS))

detections = []
frame_number = 0

print(f"Video duration: {duration:.2f} seconds")
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

        detected = False

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in TARGET_CLASSES:
                detected = True
                break

        if detected:
            timestamp = frame_number / fps
            detections.append(timestamp)

            print(f"Detected at {timestamp:.2f}s")

    frame_number += 1

cap.release()

print()
print("Creating removal intervals...")

if not detections:

    keep_intervals = [(0, duration)]

else:

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

    removal_intervals = []

    for start, end in intervals:

        start = max(0, start - PADDING)
        end = min(duration, end + PADDING)

        if removal_intervals and start <= removal_intervals[-1][1]:

            removal_intervals[-1] = (
                removal_intervals[-1][0],
                max(removal_intervals[-1][1], end)
            )

        else:

            removal_intervals.append((start, end))

    keep_intervals = []

    current = 0

    for start, end in removal_intervals:

        if current < start:
            keep_intervals.append((current, start))

        current = end

    if current < duration:
        keep_intervals.append((current, duration))


print()
print("KEEP INTERVALS")

for start, end in keep_intervals:
    print(f"{start:.2f}s -> {end:.2f}s")


if not keep_intervals:

    print("Nothing to keep.")
    exit()


print()
print("Creating final video...")

inputs = []

for start, end in keep_intervals:

    inputs.append(
        f"trim=start={start}:end={end},setpts=PTS-STARTPTS"
    )

filter_complex = ";".join(
    f"[0:v]{filter}[v{i}]"
    for i, filter in enumerate(inputs)
)

concat_inputs = "".join(
    f"[v{i}]"
    for i in range(len(inputs))
)

filter_complex += (
    f";{concat_inputs}"
    f"concat=n={len(inputs)}:v=1:a=0[outv]"
)

command = [
    "ffmpeg",
    "-y",
    "-i",
    INPUT_VIDEO,
    "-filter_complex",
    filter_complex,
    "-map",
    "[outv]",
    "-an",
    "-c:v",
    "libx264",
    "-preset",
    "ultrafast",
    "-crf",
    "23",
    OUTPUT_VIDEO
]

subprocess.run(command, check=True)

print()
print("DONE!")
print(f"Output: {OUTPUT_VIDEO}")