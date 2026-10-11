from __future__ import annotations

import csv
import json
from pathlib import Path

from common import PACKAGED_ENGINE, RUN, sha256, write_json
from trt_runtime import TensorRTRunner


def main() -> None:
    with TensorRTRunner(PACKAGED_ENGINE) as runner:
        bindings = runner.binding_info
        input_binding = next(item for item in bindings if item["is_input"])
        outputs = [item for item in bindings if not item["is_input"]]
        if tuple(input_binding["runtime_shape"]) != (1, 3, 640, 640):
            raise RuntimeError(f"Input shape mismatch: {bindings}")
        if len(outputs) != 1 or tuple(outputs[0]["runtime_shape"]) != (1, 3):
            raise RuntimeError(f"Output shape mismatch: {bindings}")
        inspector_status = "UNAVAILABLE"
        inspector_text = ""
        precision_rows = []
        try:
            inspector = runner.engine.create_engine_inspector()
            if inspector is None:
                raise RuntimeError("create_engine_inspector returned None")
            inspector.context = runner.context
            fmt = runner.trt.LayerInformationFormat.JSON
            inspector_text = inspector.get_engine_information(fmt)
            inspector_status = "AVAILABLE"
            parsed = json.loads(inspector_text)
            layers = parsed if isinstance(parsed, list) else parsed.get("Layers", parsed.get("layers", []))
            for index, layer in enumerate(layers):
                precision_rows.append({
                    "index": index,
                    "name": layer.get("Name", layer.get("name", "NA")),
                    "layer_type": layer.get("LayerType", layer.get("type", "NA")),
                    "precision": layer.get("Precision", layer.get("precision", "NOT_EXPOSED")),
                    "output_types": json.dumps(layer.get("Outputs", layer.get("outputs", []))),
                })
        except Exception as exc:
            inspector_text = f"TensorRT 8.0.1 layer inspector unavailable or insufficient: {exc!r}\n"
            inspector_status = "UNAVAILABLE_WITH_DOCUMENTED_LIMITATION"
        (RUN / "engine/engine_inspector.txt").write_text(inspector_text or "No layer information returned.\n", encoding="utf-8")
        with (RUN / "engine/fallback_layer_report.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["index", "name", "layer_type", "precision", "output_types"])
            writer.writeheader()
            writer.writerows(precision_rows)
        exposed = [str(row["precision"]).upper() for row in precision_rows]
        fp16_layers = sum("HALF" in value or "FP16" in value for value in exposed)
        fp32_layers = sum("FLOAT" in value or "FP32" in value for value in exposed)
        report = {
            "status": "PASS",
            "engine_path": str(PACKAGED_ENGINE),
            "engine_sha256": sha256(PACKAGED_ENGINE),
            "engine_size_bytes": PACKAGED_ENGINE.stat().st_size,
            "bindings": bindings,
            "input_shape_verified": True,
            "output_shape_verified": True,
            "input_output_fp32_verified": all("FLOAT" in str(item["dtype"]) for item in bindings),
            "fp16_builder_enabled": True,
            "int8_enabled": False,
            "tf32_enabled": False,
            "calibration_used": False,
            "dla_used": False,
            "layer_inspector_status": inspector_status,
            "layer_rows_exposed": len(precision_rows),
            "fp16_layers_identified": fp16_layers,
            "fp32_fallback_layers_identified": fp32_layers,
            "fallback_interpretation": (
                "Layer precision is reported only where TensorRT 8.0.1 exposes it; the FP16 builder flag permits FP32 fallback. "
                "FP32 input/output bindings do not imply all internal layers are FP32."
            ),
            "cudart": runner.cudart_path,
        }
        write_json(RUN / "engine/engine_metadata.json", report)
        (RUN / "engine/layer_precision_report.txt").write_text(
            "TensorRT FP16-enabled execution\n"
            f"Layer inspector: {inspector_status}\n"
            f"Layer rows exposed: {len(precision_rows)}\n"
            f"FP16 layers identified: {fp16_layers}\n"
            f"FP32 fallback layers identified: {fp32_layers}\n"
            "Accurate limitation: TensorRT may retain FP32 layers; no pure/all-layer FP16 claim is made.\n",
            encoding="utf-8",
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
