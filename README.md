# VideoCleaner

AI-powered video cleaner that detects people and animals
in short videos and automatically removes those sections
while removing the original audio.

## Current Features

- Human detection
- Animal detection
- Automatic unwanted-section detection
- Automatic video trimming
- Audio removal
- CPU-based processing
- Optimized for short videos

## Tech Stack

- Python
- YOLO11
- OpenCV
- FFmpeg

## Current Pipeline

Video
→ YOLO detection
→ Detection intervals
→ FFmpeg
→ Clean video
