# Snap&Style CV models

## Files and expected use

- `models/face_landmarker.task` is loaded by MediaPipe Tasks in image mode. It detects up to two faces and provides facial landmarks used by the face-shape heuristic and facial sampling regions.
- `models/resnet18.onnx` is a 19-class face-parsing model. The code expects a 512×512 RGB image normalized with ImageNet mean and standard deviation. Snap&Style uses class 1 for skin, class 16 for clothing, and class 17 for hair.

The exact binary checksums are recorded in `models/SHA256SUMS` and verified by `cv_preflight.py`.

## Provenance and licensing

The current repository does not record where these two binaries were downloaded or their redistribution licenses. Fill in the source URL, original model name/version, author, license, and conversion/export steps before distributing or deploying them. A checksum proves file identity; it does not establish ownership or permission to redistribute.

## Limitations

- Face-shape categories are hand-tuned geometric estimates, not identity or medical classifications.
- Hair color describes the photographed appearance. Lighting, dye, reflections, segmentation, coverings, and exposure affect it.
- Skin tone is an apparent camera shade estimate. Skin undertone from one uncalibrated image is especially uncertain.
- Confidence values are heuristic reliability scores, not calibrated probabilities.
- The models have not been evaluated here on a documented, representative labeled dataset. Real-world accuracy should be measured before making product claims.
