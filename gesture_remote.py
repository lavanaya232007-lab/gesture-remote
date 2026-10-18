"""Gesture remote: control volume and brightness with hand gestures.

Current step: webcam preview with hand landmarks and a Left/Right label per hand.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

# ---- Settings ----
CAMERA_INDEX = 0  # 0 = the laptop's built-in webcam
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "Gesture Remote"

# MediaPipe's Left/Right guess assumes a normal (un-mirrored) camera image. We mirror
# the image before detection, so its guess comes out backwards; True swaps it back.
# Verified on this laptop: without the swap, the real right hand was labeled "Left".
SWAP_HANDEDNESS = True


def create_hand_landmarker():
    """Load the MediaPipe hand model, set up for a video stream with up to 2 hands."""
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_hand(frame, landmarks):
    """Draw the 21 hand points and the lines between them onto the frame."""
    vision.drawing_utils.draw_landmarks(
        frame,
        landmarks,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        vision.drawing_styles.get_default_hand_landmarks_style(),
        vision.drawing_styles.get_default_hand_connections_style(),
    )


def real_hand_label(handedness):
    """Return "Left" or "Right" for your real hand, using MediaPipe's best guess."""
    label = handedness[0].category_name
    if SWAP_HANDEDNESS:
        label = "Left" if label == "Right" else "Right"
    return label


def draw_label(frame, landmarks, label):
    """Write the hand's Left/Right label just above the hand."""
    height, width = frame.shape[:2]
    # Landmark x/y are fractions (0.0 to 1.0) of the image size, so convert to pixels.
    x = int(min(lm.x for lm in landmarks) * width)
    y = int(min(lm.y for lm in landmarks) * height) - 15
    y = max(y, 30)  # keep the text inside the window
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def main():
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}\nSee README.md for the download link.")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Could not open the webcam.")
        return

    landmarker = create_hand_landmarker()
    last_timestamp_ms = -1
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("Could not read a frame from the webcam.")
                break

            # Flip left-right so the preview behaves like a mirror.
            frame = cv2.flip(frame, 1)

            # OpenCV stores colors as BGR, MediaPipe expects RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # VIDEO mode needs a timestamp in milliseconds that always goes up.
            timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # handedness[i] holds MediaPipe's guesses for hand i; the first is the best one.
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                draw_hand(frame, landmarks)
                draw_label(frame, landmarks, real_hand_label(handedness))

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            # Stop on q, or when the window's X button is clicked.
            if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
