import cv2
import os
import subprocess
from ultralytics import YOLO

INPUT_DIR = "input"
OUTPUT_DIR = "output"

SAMPLE_FPS = 2
REFINE_FPS = 4

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

os.makedirs(OUTPUT_DIR, exist_ok=True)

videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
]

if len(videos) == 0:
    print("No video found in input folder.")
    exit()

if len(videos) > 1:
    print("Multiple videos found. Please keep only one video in input folder.")
    exit()

input_video = os.path.join(INPUT_DIR, videos[0])

base_name = os.path.splitext(videos[0])[0]
output_video = os.path.join(
    OUTPUT_DIR,
    f"{base_name}_cleaned.mp4"
)

print(f"Input: {input_video}")
print(f"Output: {output_video}")

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(input_video)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Duration: {duration:.2f} seconds")
print(f"FPS: {fps}")
print(f"Coarse sampling: {SAMPLE_FPS} FPS")
print(f"Refinement sampling: {REFINE_FPS} FPS")

coarse_detections = []

step = 1 / SAMPLE_FPS
t = 0

while t < duration:

    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)

    ret, frame = cap.read()

    if not ret:
        t += step
        continue

    results = model(
        frame,
        device="cpu",
        conf=CONFIDENCE,
        classes=list(TARGET_CLASSES.keys()),
        verbose=False
    )

    found = []

    for result in results:

        if result.boxes is not None:

            for cls in result.boxes.cls:

                class_id = int(cls)
                found.append(TARGET_CLASSES[class_id])

    if found:

        coarse_detections.append(t)

        print(
            f"{t:.2f}s -> "
            f"{', '.join(found)}"
        )

    t += step

cap.release()

print()
print(f"Coarse frames checked: {int(duration * SAMPLE_FPS)}")
print(
    f"Coarse detection points: "
    f"{len(coarse_detections)}"
)

if not coarse_detections:

    print()
    print("No living things detected.")
    print("Keeping entire video.")

    keep_intervals = [(0, duration)]

else:

    detection_ranges = []

    start = coarse_detections[0]
    end = coarse_detections[0]

    for t in coarse_detections[1:]:

        if t - end <= GAP_LIMIT:

            end = t

        else:

            detection_ranges.append(
                (start, end)
            )

            start = t
            end = t

    detection_ranges.append((start, end))

    refine_ranges = []

    for start, end in detection_ranges:

        start = max(0, start - 1.0)
        end = min(duration, end + 1.0)

        if not refine_ranges:

            refine_ranges.append(
                [start, end]
            )

        elif start <= refine_ranges[-1][1]:

            refine_ranges[-1][1] = max(
                refine_ranges[-1][1],
                end
            )

        else:

            refine_ranges.append(
                [start, end]
            )

    refined_detections = []

    cap = cv2.VideoCapture(input_video)

    refine_step = 1 / REFINE_FPS

    for start, end in refine_ranges:

        print()
        print(
            f"Refining: "
            f"{start:.2f}s -> {end:.2f}s"
        )

        t = start

        while t <= end:

            cap.set(
                cv2.CAP_PROP_POS_MSEC,
                t * 1000
            )

            ret, frame = cap.read()

            if not ret:

                t += refine_step
                continue

            results = model(
                frame,
                device="cpu",
                conf=CONFIDENCE,
                classes=list(TARGET_CLASSES.keys()),
                verbose=False
            )

            found = False

            for result in results:

                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    found = True
                    break

            if found:

                refined_detections.append(t)

            t += refine_step

    cap.release()

    all_detections = sorted(
        set(
            coarse_detections
            + refined_detections
        )
    )

    detection_ranges = []

    start = all_detections[0]
    end = all_detections[0]

    for t in all_detections[1:]:

        if t - end <= GAP_LIMIT:

            end = t

        else:

            detection_ranges.append(
                (start, end)
            )

            start = t
            end = t

    detection_ranges.append(
        (start, end)
    )

    remove_intervals = []

    for start, end in detection_ranges:

        start = max(
            0,
            start - PADDING
        )

        end = min(
            duration,
            end + PADDING
        )

        remove_intervals.append(
            [start, end]
        )

    merged = []

    for start, end in remove_intervals:

        if (
            not merged
            or start > merged[-1][1]
        ):

            merged.append(
                [start, end]
            )

        else:

            merged[-1][1] = max(
                merged[-1][1],
                end
            )

    remove_intervals = merged

    keep_intervals = []

    current = 0

    for start, end in remove_intervals:

        if current < start:

            keep_intervals.append(
                (current, start)
            )

        current = end

    if current < duration:

        keep_intervals.append(
            (current, duration)
        )

print()
print("KEEP INTERVALS")
print("==============")

for start, end in keep_intervals:

    print(
        f"KEEP: "
        f"{start:.2f}s -> {end:.2f}s"
    )

if not keep_intervals:

    print()
    print("Nothing remains after removing detected sections.")
    exit()

filters = []

for i, (start, end) in enumerate(keep_intervals):

    filters.append(
        f"[0:v]trim=start={start}:end={end},"
        f"setpts=PTS-STARTPTS[v{i}]"
    )

inputs = "".join(
    f"[v{i}]" for i in range(len(keep_intervals))
)

filters.append(
    f"{inputs}concat=n={len(keep_intervals)}:"
    f"v=1:a=0[outv]"
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

print()
print("Creating cleaned video...")

subprocess.run(command)

if os.path.exists(output_video):

    size_mb = (
        os.path.getsize(output_video)
        / (1024 * 1024)
    )

    print()
    print("DONE!")
    print(f"Output: {output_video}")
    print(f"Size: {size_mb:.2f} MB")

else:

    print()
    print("ERROR: Output video was not created.")