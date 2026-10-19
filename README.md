# Gesture Remote

Control your Windows laptop's **volume** and **screen brightness** with hand gestures in front of the webcam.

- **Right hand:** pinch (thumb tip to index fingertip) controls the **volume**.
- **Left hand:** the same pinch controls the **screen brightness**.

Closed pinch = lowest level, open pinch = highest level. Everything runs locally on your laptop:
no internet connection, accounts or API keys are needed while it runs.

Built with [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker),
[OpenCV](https://opencv.org/), [pycaw](https://github.com/AndreMiras/pycaw) and
[screen-brightness-control](https://github.com/Crozzers/screen_brightness_control).

## Requirements

- Windows 10/11 laptop with a webcam
- The **built-in laptop screen** (external monitors are not supported for brightness)
- Python 3.13 (tested with 3.13.1)

## Install

Run these in PowerShell:

```powershell
git clone https://github.com/lavanaya232007-lab/gesture-remote.git
cd gesture-remote
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then download the hand-tracking model. It comes from Google's official MediaPipe models
(listed on the [Hand Landmarker page](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker#models)):

```powershell
New-Item -ItemType Directory -Force models
Invoke-WebRequest "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile models\hand_landmarker.task
```

> Don't also install `opencv-python`. MediaPipe already uses `opencv-contrib-python`, and having both breaks `import cv2`.

## Run

**Dry-run** (the default). It shows and prints the values, but **changes nothing** on your system:

```powershell
.\.venv\Scripts\python.exe gesture_remote.py
```

**Live**. It really changes the volume and brightness:

```powershell
.\.venv\Scripts\python.exe gesture_remote.py --live
```

### Keys

Click the video window first, so it receives your key presses.

| Key | Action |
|---|---|
| `space` | Turn control ON / OFF (it always starts OFF) |
| `q` | Quit (closing the window also quits) |

The terminal shows one status line that updates in place, for example:

```
Control ON  | Right 0.85 -> volume 0.42 | Left 0.45 -> brightness  52
```

## Safety

- Starts in **dry-run**. Nothing changes unless you add `--live`.
- Control starts **OFF**. Press space to turn it on.
- Volume never goes above `MAX_VOLUME = 0.70` (Windows shows 70).
  If the volume is already higher when you turn control on, it comes down gradually.
- Brightness never goes below `MIN_BRIGHTNESS = 10` percent, so the screen never goes dark.
  If it's already lower when you turn control on, it comes up gradually.
- When control turns on, both levels start from their **current** values and move gradually:
  values are smoothed, each frame can only change a little, and tiny changes are ignored so nothing flickers.
- If a hand isn't visible, its setting stays unchanged.
- **On quit, volume and brightness are left as they are.** They are not restored to the values from before.
- The script never mutes or unmutes.

## Calibrate

The script measures a **pinch ratio** for each hand:

```
pinch ratio = (thumb tip to index fingertip distance) / (wrist to middle-finger base distance)
```

Dividing by the size of your hand means the ratio stays about the same whether your hand is close to the camera or far away.

To tune it for your hands:

1. Run the dry-run. Each hand's label shows its ratio (for example **Right 0.85**). The terminal shows it too.
2. Pinch **closed** (thumb touching index finger) and note the number for each hand.
3. Open the pinch **fully** and note the number for each hand.
4. At the top of `gesture_remote.py`, set:
   - `PINCH_RATIO_MIN` a little **above** your highest "closed" number, so a closed pinch reliably reaches the lowest level.
   - `PINCH_RATIO_MAX` a little **below** your lowest "open" number, so an open pinch reliably reaches the highest level.

Both hands share these two values. The defaults (`0.35` and `1.20`) came from measured closed values of 0.30 / 0.20
and open values of 1.55 / 1.30.

Other settings at the top of the file:

| Setting | What it does |
|---|---|
| `SMOOTHING` | 0 to 1. Lower = smoother but slower to follow your hand |
| `VOLUME_MAX_STEP`, `BRIGHTNESS_MAX_STEP` | Biggest change allowed per video frame |
| `VOLUME_DEADBAND`, `BRIGHTNESS_DEADBAND` | Changes smaller than this are ignored |
| `SWAP_HANDEDNESS` | Swaps the Left/Right labels (see Troubleshooting) |
| `CAMERA_INDEX` | Which camera to use (0 = built-in webcam) |

## Troubleshooting

| Problem | Fix |
|---|---|
| Space or `q` does nothing | Click the video window first. Key presses go to whichever window is active, often the terminal. |
| Your real right hand is labeled "Left" | Flip `SWAP_HANDEDNESS` (`True` / `False`). MediaPipe's label assumes an un-mirrored image, and the preview is mirrored. |
| Volume number changes but you hear no difference | Windows is muted. Unmute it; the script never changes mute. |
| `Model file not found` | Download the model (see Install). |
| `Could not open the webcam.` | Close other apps using the camera (Camera, Teams, Zoom, browser tabs). Check **Settings → Privacy & security → Camera**: "Camera access" and "Let desktop apps access your camera" must be On. If you have more than one camera, try another `CAMERA_INDEX`. |
| Brightness doesn't change | Only built-in laptop screens are supported (Windows "wmi" method). Test with: `.\.venv\Scripts\python.exe -c "import screen_brightness_control as sbc; print(sbc.get_brightness(method='wmi'))"` |
| Video stutters while brightness is moving | Normal: each brightness change takes Windows about 40 ms. It goes back to normal when your hand is still. |
| VS Code says `Import "cv2" could not be resolved` | The code is fine. Point VS Code at the project's Python: **Ctrl+Shift+P → Python: Select Interpreter →** `.venv\Scripts\python.exe`. |
| `import cv2` fails or behaves strangely | Make sure `opencv-python` isn't installed next to `opencv-contrib-python`: `.\.venv\Scripts\python.exe -m pip uninstall opencv-python` |

## Project files

```
gesture_remote.py        the whole program (settings at the top)
requirements.txt         Python libraries, pinned to tested versions
models/                  hand_landmarker.task goes here (downloaded, not stored in Git)
```
