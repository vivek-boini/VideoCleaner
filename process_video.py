import cv2
import subprocess
import os
import sys
from ultralytics import YOLO

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

INPUT_DIR = "input"
OUTPUT_DIR = "output"

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
]

if not videos:
    print("No videos found in input folder.")
    sys.exit()

if len(videos) > 1:
    print("Multiple videos found.")
    print("For now, keep only one video in the input folder.")
    sys.exit()

input_video = os.path.join(INPUT_DIR, videos[0])

filename = os.path.splitext(os.path.basename(input_video))[0]

output_video = os.path.join(
    OUTPUT_DIR,
    f"{filename}_cleaned.mp4"
)

print(f"Input: {input_video}")
print(f"Output: {output_video}")
print()

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

frame_step = max(1, round(fps / SAMPLE_FPS))

detections = []
frame_number = 0
checked = 0

print(f"Duration: {duration:.2f} seconds")
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

        detected = False

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id in TARGET_CLASSES:
                detected = True
                break

        checked += 1

        if detected:

            timestamp = frame_number / fps
            detections.append(timestamp)

            print(f"Detected at {timestamp:.2f}s")

    frame_number += 1

cap.release()

print()
print(f"Frames checked: {checked}")
print(f"Detection points: {len(detections)}")

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

    print("No usable video remains.")
    sys.exit()

print()
print("Creating final video...")

filters = []

for i, (start, end) in enumerate(keep_intervals):

    filters.append(
        f"[0:v]trim=start={start}:end={end},"
        f"setpts=PTS-STARTPTS[v{i}]"
    )

concat_inputs = "".join(
    f"[v{i}]"
    for i in range(len(keep_intervals))
)

filters.append(
    f"{concat_inputs}"
    f"concat=n={len(keep_intervals)}:v=1:a=0[outv]"
)

filter_complex = ";".join(filters)

command = [
    "ffmpeg",
    "-y",
    "-i",
    input_video,
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
    output_video
]

subprocess.run(command, check=True)

print()
print("================================")
print("DONE!")
print("================================")
print(f"Output: {output_video}")