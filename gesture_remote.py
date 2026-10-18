"""Gesture remote: control volume and brightness with hand gestures.

Current step: right-hand pinch controls the volume (dry-run by default, --live to change it).
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from pycaw.pycaw import AudioUtilities

# ---- Safety limits ----
MAX_VOLUME = 0.70  # system volume never goes above this (0.0 = silent, 1.0 = 100%)

# ---- Pinch calibration (see README for how to tune) ----
PINCH_RATIO_MIN = 0.35  # this ratio or smaller = pinch closed = lowest level
PINCH_RATIO_MAX = 1.20  # this ratio or bigger = pinch open = highest level

# ---- Smoothing ----
SMOOTHING = 0.2  # 0 to 1: how far to move toward the target each frame (lower = smoother, slower)
VOLUME_MAX_STEP = 0.02  # biggest volume change allowed in one frame
VOLUME_DEADBAND = 0.01  # volume changes smaller than this are ignored (stops flicker)

# ---- Other settings ----
CAMERA_INDEX = 0  # 0 = the laptop's built-in webcam
MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "Gesture Remote"

# MediaPipe's Left/Right guess assumes a normal (un-mirrored) camera image. We mirror
# the image before detection, so its guess comes out backwards; True swaps it back.
# Verified on this laptop: without the swap, the real right hand was labeled "Left".
SWAP_HANDEDNESS = True

# Landmark numbers from MediaPipe's list of 21 hand points.
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_BASE = 9  # the knuckle where the middle finger joins the palm


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


def pinch_ratio(landmarks, width, height):
    """Thumb-to-index distance divided by hand size (wrist to middle-finger base).

    Dividing by hand size keeps the ratio about the same whether your hand is
    close to the camera or far away.
    """
    def distance(a, b):
        # x and y are fractions of the image width/height, so convert to pixels first.
        return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)

    hand_size = max(distance(landmarks[WRIST], landmarks[MIDDLE_BASE]), 1e-6)  # never divide by 0
    return distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_size


def pinch_to_level(ratio, low, high):
    """Turn a pinch ratio into a level: closed pinch = low, open pinch = high."""
    fraction = (ratio - PINCH_RATIO_MIN) / (PINCH_RATIO_MAX - PINCH_RATIO_MIN)
    fraction = max(0.0, min(1.0, fraction))  # beyond the calibrated range: stay at the end
    return low + fraction * (high - low)


class SmoothLevel:
    """Moves a level (like the volume) gently toward a target, one video frame at a time."""

    def __init__(self, low, high, max_step, deadband):
        self.low = low  # bottom of the allowed range
        self.high = high  # top of the allowed range
        self.max_step = max_step  # biggest change allowed in one frame
        self.deadband = deadband  # changes smaller than this are ignored
        self.level = None  # the level we last set (or read from the system)
        self.smoothed = None  # smoothed version of the target

    def start(self, current_level):
        """Begin from the system's current level, so nothing jumps."""
        self.level = current_level
        self.smoothed = current_level

    def update(self, target):
        """Move one frame toward target. Returns the new level, or None if nothing should change."""
        # 1. Smooth: move only part of the way toward the target each frame,
        #    which averages out small shakes in the hand tracking.
        self.smoothed += SMOOTHING * (target - self.smoothed)
        # Holding a full pinch puts the target at an end of the range; finish the
        # move there so the exact end (e.g. 0 or MAX_VOLUME) is reached.
        at_end = target in (self.low, self.high)
        if at_end and abs(target - self.smoothed) < self.deadband:
            self.smoothed = target

        # 2. Ignore tiny changes so the level doesn't flicker. (Not needed at an end:
        #    the target is steady there, and we want to finish the move.)
        difference = self.smoothed - self.level
        if difference == 0 or (abs(difference) < self.deadband and not at_end):
            return None

        # 3. Limit how much the level can change in one frame.
        new_level = self.level + max(-self.max_step, min(self.max_step, difference))

        # 4. Safety: never go above high or below low. If we started outside the range
        #    (e.g. the volume was already 0.90), the level may only move back toward it.
        new_level = min(new_level, max(self.high, self.level))
        new_level = max(new_level, min(self.low, self.level))
        self.level = new_level
        return new_level


def print_status(control_on, ratios, volume):
    """Print one status line, overwriting the previous one ("\\r" goes back to the line start)."""
    def show(value):
        return " -- " if value is None else f"{value:.2f}"

    state = "ON " if control_on else "OFF"
    print(
        f"Control {state} | Right: {show(ratios.get('Right'))} -> volume {show(volume)}"
        f" | Left: {show(ratios.get('Left'))}   ",
        end="\r",
    )


def draw_label(frame, landmarks, label):
    """Write the hand's Left/Right label just above the hand."""
    height, width = frame.shape[:2]
    # Landmark x/y are fractions (0.0 to 1.0) of the image size, so convert to pixels.
    x = int(min(lm.x for lm in landmarks) * width)
    y = int(min(lm.y for lm in landmarks) * height) - 15
    y = max(y, 30)  # keep the text inside the window
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)


def parse_args():
    parser = argparse.ArgumentParser(description="Control volume and brightness with hand gestures.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="really change the system volume (without this, it's a dry-run that changes nothing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not MODEL_PATH.exists():
        print(f"Model file not found: {MODEL_PATH}\nSee README.md for the download link.")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print("Could not open the webcam.")
        return

    # Windows' volume control for the default speakers (pycaw), using the 0.0-1.0 scale.
    volume_endpoint = AudioUtilities.GetSpeakers().EndpointVolume
    volume_control = SmoothLevel(0.0, MAX_VOLUME, VOLUME_MAX_STEP, VOLUME_DEADBAND)

    landmarker = create_hand_landmarker()
    last_timestamp_ms = -1
    control_on = False  # control always starts OFF; space toggles it
    if args.live:
        print("LIVE: when control is ON, your right hand really changes the system volume.")
    else:
        print("DRY-RUN: nothing on your system will be changed (add --live to change it).")
    print("Space = control ON/OFF, q = quit.")
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

            height, width = frame.shape[:2]
            ratios = {}  # pinch ratio of each visible hand, e.g. {"Right": 0.85}
            # handedness[i] holds MediaPipe's guesses for hand i; the first is the best one.
            for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                label = real_hand_label(handedness)
                ratios[label] = pinch_ratio(landmarks, width, height)
                draw_hand(frame, landmarks)
                draw_label(frame, landmarks, f"{label} {ratios[label]:.2f}")

            # Right hand controls the volume, only while control is ON.
            # No right hand visible = the volume is left unchanged.
            if control_on and "Right" in ratios:
                target = pinch_to_level(ratios["Right"], 0.0, MAX_VOLUME)
                new_volume = volume_control.update(target)
                if new_volume is not None and args.live:
                    volume_endpoint.SetMasterVolumeLevelScalar(new_volume, None)

            print_status(control_on, ratios, volume_control.level if control_on else None)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                control_on = not control_on
                if control_on:
                    # Start from the real current volume so nothing jumps.
                    current = volume_endpoint.GetMasterVolumeLevelScalar()
                    volume_control.start(current)
                    if current > MAX_VOLUME:
                        print(f"\nVolume is {current:.2f}, above MAX_VOLUME ({MAX_VOLUME}): "
                              "it will come down gradually.")
            # Stop on q, or when the window's X button is clicked.
            if key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        print()  # move past the status line
        landmarker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
