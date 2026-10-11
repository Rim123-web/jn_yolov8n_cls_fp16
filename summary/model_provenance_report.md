# Historical YOLOv8n-cls Audit

- Executed source: `/home/jetson/Downloads/classification_jetson/yolov8n_cls_fp32_core4_benchmark/scripts/run_yolov8n_cls_fp32_core4_benchmark.py`
- Architecture: YOLOv8n-cls, three classes.
- Original ONNX: `/home/jetson/Downloads/classification_benchmark/yolov8n-cls/yolov8n_cls_tensorrt_fp32_640x640_batch1_warmup10_20260717_161313/onnx/yolov8n_cls_source.onnx` (`341afc3adf731e3ff5ac1ccc0c5893d6eed45eca47d9eb80d536ddcbde48ea1c`).
- Historical model/ONNX pair is repeated with identical hashes across FP32, FP16, and INT8 packages.
- Preprocessing: PIL RGB, bilinear shortest-edge resize to 640, center crop 640x640, /255, identity mean/std, contiguous NCHW FP32.
- Output: embedded Softmax probabilities; no external Softmax is applied.
- Preprocessing ambiguity: none.

# ONNX Validation

- onnx.checker: PASS
- Fixed FP32 input: [1,3,640,640]
- FP32 output: [1,3]
- ONNX Runtime CPU dummy inference: unavailable on this Jetson and not forced
- Embedded Softmax: True
