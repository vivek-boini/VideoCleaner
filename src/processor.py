import cv2
import os
import subprocess
from ultralytics import YOLO


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


def detect_living_things(video_path, model):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

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

        t += step

    cap.release()

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

    return keep_intervals, duration


def create_cleaned_video(
    input_video,
    output_video,
    keep_intervals
):
    if not keep_intervals:
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

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return result.returncode == 0


def process_video(input_video, output_video, model):
    keep_intervals, duration = detect_living_things(
        input_video,
        model
    )

    success = create_cleaned_video(
        input_video,
        output_video,
        keep_intervals
    )

    return success, duration