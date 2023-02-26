# 3D Hand Tracking Demos
Biggest names in the tech industry (Microsoft, Facebook, SnapChat) have released commercial products that support 3D hand tracking without any controllers OR markers.
This library will allow users to demo multiple existing open-source 3D hand tracking solutions for monocular RGB cameras.

## TODO List
In descending order of priority

### Hand Detectors
- [X] Standalone hand detection demos
- [ ] Standardise hand detector outputs by creating a wrapper (possibly shared parent)
- [ ] Crop images based on hand detector output (Might be difficult with BlazePalm output being non-axis-aligned)
- [ ] Figure out the best way to incorporate detector within the architecture so that it's switched on for some models (e.g. Meshformer) and switched off for others e.g. E2E Mediapipe tracking, minimal-hand

### Keypoint Estimators
- [ ] Implement Minimal Hand
- [ ] Implement Meshformer
- [ ] Implement official C++ GANerated Hands Tracker

### GUI Improvements
- [ ] Allow users to drag-and-drop custom models (impose limitations on model's I/O)
- [ ] Create a 3D representation of the estimated keypoints/mesh
- [ ] Change the default font

### General Stuff
- [ ] Arguments for the architecture/designed patterns used
- [ ] Possible issues with LICENSE with GANerated Hands and MANO mesh
- [ ] There will be issues with filepaths on Windows

## References
Hand Detection Models
1. https://github.com/aashish2000/hand_tracking
2. https://github.com/cansik/yolo-hand-detection


Keypoint Estimation Models:
1. https://google.github.io/mediapipe/solutions/hands