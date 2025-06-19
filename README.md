# Customs checker

**Customs checker** is a satirical Python application designed to mimic and expose the reductive logic of skin-tone-based profiling, often seen in border control or surveillance systems. It uses real-time face detection, skin color sampling, and visual classification to simulate how simplistic algorithms can be used to enforce biased decisions under the guise of automation.

> **Disclaimer:** This project is a **joke**. It is **not intended** for production use. It aims to illustrate, through mockery, the dangers of building tech that encodes human bias.

---

## Features

- Real-time webcam face detection (OpenCV Haar cascade)
- 3-second facial analysis lock-in with live face tracking
- Skin tone sampling from central facial region
- Euclidean distance-based Lab color classification:
  - Closer to light → "white"
  - Closer to dark → "black"
  - In between → "neutral"
- Dynamic GUI (Tkinter):
  - Status panel for face detection and color class
  - Color category reference display (white/dark tone swatches)
  - Hints for lighting, alignment, and detection feedback
- Pop-up alerts with colored backgrounds
- Retake test functionality to restart the flow

---

## Installation

**Dependencies:**

- Python 3.10+
- `opencv-python`
- `numpy`
- `Pillow`

**Install via pip:**

```bash
pip install opencv-python numpy pillow
````

---

## Usage

```bash
python main.py
```

## Technical Design

### Face Detection

```python
cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

* Works on grayscale frame.
* Bounding box used to extract face region.
* Only one face (largest) is processed.

### Color Sampling

* Samples central 50×50 pixel region within face bounding box.
* Computes average BGR → RGB → Hex.
* Converts to Lab space using OpenCV:

  ```python
  cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)
  ```

### Classification Logic

* Reference Lab values:

  * White skin ≈ `[80, 0, 0]`
  * Black/brown skin ≈ `[35, 0, 0]`
* Compares distance:

  ```python
  np.linalg.norm(color_lab - reference)
  ```
* Classification thresholds:

  * `white` if much closer to white ref
  * `black` if much closer to dark ref
  * `neutral` otherwise

---

## Ethical Context

This mock system replicates, with minimal code, the kind of bias found in real-world border security algorithms. It demonstrates:

* How appearance-based heuristics can be trivially implemented
* That automation is not inherently objective
* The absurdity of drawing social conclusions from color proximity

By making it explicit and slightly absurd, the project aims to critique rather than endorse such profiling.

---

## Known Limitations

* Face detection uses Haar cascades — outdated and fragile under occlusion or poor lighting.
* Skin color classification is uncalibrated and naive.
* RGB→Lab conversion is lighting-dependent.
* Ethnic labeling based solely on tone is reductive and intentionally oversimplified.

---

## Project Structure

```
- main.py                 
- haarcascade_frontalface_default.xml
```