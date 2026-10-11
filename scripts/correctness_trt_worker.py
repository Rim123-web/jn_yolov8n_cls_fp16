from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import load_image_order, output_probabilities, preprocess
from trt_runtime import TensorRTRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    indices = [int(value) for value in json.loads(args.indices)]
    samples = load_image_order()
    outputs = []
    with TensorRTRunner(Path(args.engine)) as runner:
        if tuple(runner.input_shape) != (1, 3, 640, 640):
            raise RuntimeError(f"{args.label} engine shape mismatch: {runner.binding_info}")
        if not all("FLOAT" in item["dtype"].upper() for item in runner.binding_info):
            raise RuntimeError(f"{args.label} binding dtype mismatch: {runner.binding_info}")
        for index in indices:
            tensor = preprocess(samples[index]["path"])
            raw, _ = runner.infer(tensor, timed=False)
            outputs.append(output_probabilities(raw.reshape(-1)))
    array = np.asarray(outputs, dtype=np.float32)
    if array.shape != (len(indices), 3) or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid {args.label} output array: {array.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    print(json.dumps({"status": "PASS", "backend": args.label, "shape": list(array.shape)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
