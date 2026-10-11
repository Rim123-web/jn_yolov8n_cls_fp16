# Publication Results: YOLOv8n-cls TensorRT FP16-enabled on Jetson Nano

| Item | Result |
|---|---:|
| Images | 261 |
| Accuracy | 0.850575 |
| Macro precision | 0.880051 |
| Macro recall | 0.793250 |
| Macro F1 | 0.824559 |
| Weighted F1 | 0.842516 |
| Engine latency | 22.811 ms |
| End-to-end latency | 201.587 ms |
| Engine FPS | 43.838 |
| Transfer-inclusive FPS | 38.913 |
| Wall-clock pipeline FPS | 4.948 |
| Average Power TOT | 5.047 W |
| Gross energy/image | 0.990 J |
| Idle-corrected energy/image | 0.370 J |

# Methodology

- Device: NVIDIA Jetson Nano Developer Kit / Tegra X1.
- Model: YOLOv8n-cls classification from `/home/jetson/Downloads/classification_benchmark/yolov8n-cls/yolov8n_cls_tensorrt_fp32_640x640_batch1_warmup10_20260717_161313/model/yolov8n_cls_inference_only.pt` (trusted `.pt` checkpoint).
- Source SHA-256: `6b756b9724c65520f10586087e119afa6cccc28a258916390d5eb641b8f83a54`.
- Source artifact: exact validated static ONNX copied byte-for-byte from the final FP32 package; no ONNX export or graph rewrite.
- ONNX: `/home/jetson/Downloads/classification_benchmark/yolov8n-cls/yolov8n_cls_tensorrt_fp16_640x640_batch1_warmup10_20260717_170948/onnx/yolov8n_cls_source.onnx`; SHA-256 `341afc3adf731e3ff5ac1ccc0c5893d6eed45eca47d9eb80d536ddcbde48ea1c`; opset 17; exact FP32 ONNX parity PASS.
- ONNX input/output: `images` FLOAT `[1,3,640,640]` to `output0` FLOAT `[1,3]`; embedded preprocessing false; embedded softmax true.
- TensorRT FP16-enabled execution (`--fp16`, `--noTF32`, INT8 false, no calibration, no DLA); no pure/all-layer FP16 claim.
- Fresh engine build duration: 466.085176 s; input/output binding dtypes `DataType.FLOAT` / `DataType.FLOAT`.
- Layer inspection: UNAVAILABLE_WITH_DOCUMENTED_LIMITATION; FP32 fallback is permitted and documented where exact layer precision is unavailable.
- Input: batch 1, RGB `[1,3,640,640]`.
- Preprocessing: PIL decode and RGB conversion, bilinear shortest-edge resize to 640,
  center crop 640x640, divide by 255, identity mean `[0,0,0]` and std `[1,1,1]`, HWC-to-CHW.
- Antialias setting: `None` in the recovered torchvision transform; no EXIF transpose, letterbox, or padding.
- Output handling: consume embedded-softmax probabilities directly, then argmax; no external softmax.
- Warm-up: exactly 10 engine executions, excluded from all official timing and monitoring.
- Measured pass: 261 validated images exactly once.
- CPU stages: `time.perf_counter_ns()`.
- H2D, TensorRT, D2H: CUDA events with synchronization at stage boundaries.
- Power boundary: Jetson module-level jtop Power TOT (POM_5V_IN), sampled at 0.5 s.
- Energy: trapezoidal integration of timestamped Power TOT samples.
- FP32/FP16 protocol parity: PASS.
- Cross-model FP16 context: PASS_WITH_DOCUMENTED_NATIVE_PREPROCESSING_DIFFERENCE with model-native preprocessing differences documented.
