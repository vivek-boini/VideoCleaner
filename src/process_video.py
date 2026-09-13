import cv2
import os
import subprocess
from ultralytics import YOLO


INPUT_DIR = "input"
OUTPUT_DIR = "output"

SAMPLE_FPS = 3
REFINE_FPS = 4

CONFIDENCE = 0.35
PADDING = 0.5
GAP_LIMIT = 0.75

SMOOTH_GAP = 1.5
MIN_KEEP_DURATION = 2.5

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

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")


os.makedirs(OUTPUT_DIR, exist_ok=True)


videos = [
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(VIDEO_EXTENSIONS)
]

if not videos:
    print("No videos found in input folder.")
    exit()


print(f"Found {len(videos)} video(s).")
print()


model = YOLO("yolo11n.pt")


def merge_intervals(intervals, max_gap=0):

    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda x: x[0])

    merged = [[intervals[0][0], intervals[0][1]]]

    for start, end in intervals[1:]:

        if start <= merged[-1][1] + max_gap:

            merged[-1][1] = max(
                merged[-1][1],
                end
            )

        else:

            merged.append(
                [start, end]
            )

    return merged


def detect_living_things(video_path):

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    print(f"Duration: {duration:.2f}s")
    print(f"FPS: {fps}")
    print(f"Coarse sampling: {SAMPLE_FPS} FPS")
    print(f"Refinement sampling: {REFINE_FPS} FPS")

    coarse_detections = []

    step = 1 / SAMPLE_FPS
    t = 0

    while t < duration:

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            t * 1000
        )

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

                    if class_id in TARGET_CLASSES:
                        found.append(
                            TARGET_CLASSES[class_id]
                        )

        if found:

            coarse_detections.append(t)

            print(
                f"{t:.2f}s -> "
                f"{', '.join(found)}"
            )

        t += step

    cap.release()

    print()
    print(
        f"Coarse detection points: "
        f"{len(coarse_detections)}"
    )

    if not coarse_detections:

        return [(0, duration)], duration

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

    detection_ranges.append(
        (start, end)
    )

    refine_ranges = []

    for start, end in detection_ranges:

        start = max(
            0,
            start - 1.0
        )

        end = min(
            duration,
            end + 1.0
        )

        refine_ranges.append(
            [start, end]
        )

    refine_ranges = merge_intervals(
        refine_ranges
    )

    refined_detections = []

    cap = cv2.VideoCapture(video_path)

    refine_step = 1 / REFINE_FPS

    for start, end in refine_ranges:

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

    remove_intervals = merge_intervals(
        remove_intervals
    )

    print()
    print("REMOVE INTERVALS")
    print("================")

    for start, end in remove_intervals:

        print(
            f"REMOVE: "
            f"{start:.2f}s -> {end:.2f}s"
        )

    print()

    smoothed_remove_intervals = []

    for start, end in remove_intervals:

        if not smoothed_remove_intervals:

            smoothed_remove_intervals.append(
                [start, end]
            )

            continue

        previous_start, previous_end = (
            smoothed_remove_intervals[-1]
        )

        gap = start - previous_end

        if gap <= SMOOTH_GAP:

            smoothed_remove_intervals[-1][1] = end

        else:

            smoothed_remove_intervals.append(
                [start, end]
            )

    remove_intervals = smoothed_remove_intervals

    remove_intervals = merge_intervals(
        remove_intervals
    )

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

    final_keep_intervals = []
    final_remove_intervals = list(remove_intervals)

    for start, end in keep_intervals:

        if end - start < MIN_KEEP_DURATION:

            final_remove_intervals.append(
                [start, end]
            )

        else:

            final_keep_intervals.append(
                (start, end)
            )

    final_remove_intervals = merge_intervals(
        final_remove_intervals
    )

    keep_intervals = []

    current = 0

    for start, end in final_remove_intervals:

        if current < start:

            keep_intervals.append(
                (current, start)
            )

        current = end

    if current < duration:

        keep_intervals.append(
            (current, duration)
        )

    print("SMOOTHED KEEP INTERVALS")
    print("======================")

    for start, end in keep_intervals:

        print(
            f"KEEP: "
            f"{start:.2f}s -> {end:.2f}s "
            f"({end - start:.2f}s)"
        )

    return keep_intervals, duration


def create_cleaned_video(
    input_video,
    output_video,
    keep_intervals
):

    if not keep_intervals:

        print(
            "Nothing remains after "
            "removing detected sections."
        )

        return False

    filters = []

    for i, (start, end) in enumerate(
        keep_intervals
    ):

        filters.append(
            f"[0:v]trim=start={start}:end={end},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )

    inputs = "".join(
        f"[v{i}]"
        for i in range(len(keep_intervals))
    )

    filters.append(
        f"{inputs}concat="
        f"n={len(keep_intervals)}:"
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

    result = subprocess.run(command)

    return result.returncode == 0


successful = 0
failed = 0
skipped = 0


for index, video in enumerate(videos, 1):

    print("=" * 60)
    print(f"VIDEO {index}/{len(videos)}")
    print("=" * 60)

    if "_cleaned" in os.path.splitext(video)[0].lower():

        print(
            f"Skipping already cleaned video: "
            f"{video}"
        )

        skipped += 1
        print()
        continue

    input_video = os.path.join(
        INPUT_DIR,
        video
    )

    base_name = os.path.splitext(video)[0]

    output_video = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_cleaned.mp4"
    )

    if os.path.exists(output_video):

        print(
            f"Output already exists: "
            f"{output_video}"
        )

        print("Skipping this video.")

        skipped += 1
        print()
        continue

    print(f"Input: {input_video}")
    print(f"Output: {output_video}")
    print()

    try:

        keep_intervals, duration = (
            detect_living_things(
                input_video
            )
        )

        print()
        print("Creating cleaned video...")

        success = create_cleaned_video(
            input_video,
            output_video,
            keep_intervals
        )

        if success:

            size_mb = (
                os.path.getsize(output_video)
                / (1024 * 1024)
            )

            print()
            print("DONE!")

            print(
                f"Output: "
                f"{output_video}"
            )

            print(
                f"Size: "
                f"{size_mb:.2f} MB"
            )

            successful += 1

        else:

            print()
            print(
                "FAILED: "
                "Could not create output."
            )

            failed += 1

    except Exception as e:

        print()
        print("FAILED:")
        print(str(e))

        failed += 1

    print()


print("=" * 60)
print("BATCH SUMMARY")
print("=" * 60)

print(
    f"Total videos: {len(videos)}"
)

print(
    f"Successful:   {successful}"
)

print(
    f"Failed:       {failed}"
)

print(
    f"Skipped:      {skipped}"
)