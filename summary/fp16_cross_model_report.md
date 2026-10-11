# ResNet50 vs MobileNetV3-Small vs YOLOv8n-cls TensorRT FP16-enabled

- Protocol parity: **PASS_WITH_DOCUMENTED_NATIVE_PREPROCESSING_DIFFERENCE**
- YOLOv8n-cls accuracy: 0.850575; macro F1: 0.824559.
- YOLOv8n-cls engine speedup vs ResNet50: 8.204x.
- YOLOv8n-cls engine speedup vs MobileNetV3-Small: 1.191x.
- Model-native preprocessing differs and is disclosed; device, dataset, shape, batch, warm-up, execution count, timing, and telemetry boundaries match.
