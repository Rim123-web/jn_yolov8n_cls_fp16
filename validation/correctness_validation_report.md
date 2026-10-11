# Source, FP32, and FP16 Correctness Validation

- Status: **PASS**
- Samples: 20 across all three classes
- FP32/FP16 top-1 agreement: 1.000000
- Source/FP16 agreement: 1.000000
- Maximum FP32/FP16 probability error: 0.0179914832
- Minimum FP32/FP16 cosine similarity: 0.999518812
- Runtime isolation: FP32 TensorRT and FP16 TensorRT ran in separate fresh subprocesses.
- Source validation mode: REUSED_VALIDATED_IMMUTABLE_FP32_REFERENCE_AFTER_CURRENT_CPU_PYTORCH_NONFINITE_OUTPUT
- Limitation: the current isolated CPU PyTorch worker produced non-finite outputs, so exact source outputs from the immutable validated FP32 package were reused after all parity gates passed.
- ONNX checker/parser: PASS; ONNX Runtime unavailable and safely excluded.
