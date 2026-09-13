import cv2
from ultralytics import YOLO

INPUT_VIDEO = "input/test.mp4"

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

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(INPUT_VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print(f"Video duration: {duration:.2f} seconds")
print(f"FPS: {fps}")
print(f"Coarse sampling: {SAMPLE_FPS} FPS")
print(f"Refinement sampling: {REFINE_FPS} FPS")

coarse_times = []
detections = []

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
        detections.append(t)
        print(f"{t:.2f}s -> {', '.join(found)}")

    coarse_times.append(t)
    t += step

cap.release()

print()
print(f"Coarse frames checked: {len(coarse_times)}")
print(f"Coarse detection points: {len(detections)}")

if not detections:
    print()
    print("No living things detected.")
    print(f"KEEP: 0.00s -> {duration:.2f}s")
    exit()

refine_ranges = []

start = detections[0]
end = detections[0]

for t in detections[1:]:
    if t - end <= GAP_LIMIT:
        end = t
    else:
        refine_ranges.append((start, end))
        start = t
        end = t

refine_ranges.append((start, end))

merged_refine_ranges = []

for start, end in refine_ranges:
    start = max(0, start - 1.0)
    end = min(duration, end + 1.0)

    if not merged_refine_ranges or start > merged_refine_ranges[-1][1]:
        merged_refine_ranges.append([start, end])
    else:
        merged_refine_ranges[-1][1] = max(
            merged_refine_ranges[-1][1],
            end
        )

refine_ranges = merged_refine_ranges

refined_detections = []

cap = cv2.VideoCapture(INPUT_VIDEO)

refine_step = 1 / REFINE_FPS

for start, end in refine_ranges:

    refine_start = max(0, start - 1.0)
    refine_end = min(duration, end + 1.0)

    print()
    print(
        f"Refining: {refine_start:.2f}s -> "
        f"{refine_end:.2f}s"
    )

    t = refine_start

    while t <= refine_end:

        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
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
            if result.boxes is not None and len(result.boxes) > 0:
                found = True
                break

        if found:
            refined_detections.append(t)

        t += refine_step

cap.release()

all_detections = sorted(set(detections + refined_detections))

intervals = []

start = all_detections[0]
end = all_detections[0]

for t in all_detections[1:]:

    if t - end <= GAP_LIMIT:
        end = t
    else:
        intervals.append((start, end))
        start = t
        end = t

intervals.append((start, end))

remove_intervals = []

for start, end in intervals:

    start = max(0, start - PADDING)
    end = min(duration, end + PADDING)

    remove_intervals.append((start, end))

merged = []

for start, end in remove_intervals:

    if not merged or start > merged[-1][1]:
        merged.append([start, end])
    else:
        merged[-1][1] = max(merged[-1][1], end)

remove_intervals = merged

keep_intervals = []

current = 0

for start, end in remove_intervals:

    if current < start:
        keep_intervals.append((current, start))

    current = end

if current < duration:
    keep_intervals.append((current, duration))

print()
print("DETECTION INTERVALS")
print("===================")

for start, end in remove_intervals:
    print(f"REMOVE: {start:.2f}s -> {end:.2f}s")

print()
print("KEEP INTERVALS")
print("==============")

for start, end in keep_intervals:
    print(f"KEEP:   {start:.2f}s -> {end:.2f}s")