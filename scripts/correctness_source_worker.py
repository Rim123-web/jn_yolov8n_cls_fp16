from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import PACKAGED_MODEL, load_image_order, load_model, output_probabilities, preprocess


def unwrap(result) -> np.ndarray:
    if isinstance(result, (tuple, list)):
        result = result[0]
    if isinstance(result, dict):
        result = next(result[key] for key in ("output", "probabilities", "logits", "pred") if key in result)
    return result.detach().cpu().numpy().reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    indices = [int(value) for value in json.loads(args.indices)]
    samples = load_image_order()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model, _, _ = load_model(PACKAGED_MODEL)
    nonfinite_parameters = [name for name, value in model.named_parameters() if not torch.isfinite(value).all()]
    nonfinite_buffers = [name for name, value in model.named_buffers() if not torch.isfinite(value).all()]
    if nonfinite_parameters or nonfinite_buffers:
        raise RuntimeError(
            f"Non-finite model state: parameters={nonfinite_parameters}, buffers={nonfinite_buffers}"
        )
    outputs = []
    with torch.inference_mode():
        for index in indices:
            tensor = preprocess(samples[index]["path"])
            if not np.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite input tensor for {samples[index]['path']}")
            raw = unwrap(model(torch.from_numpy(tensor)))
            if not np.isfinite(raw).all():
                raise RuntimeError(
                    f"Non-finite source output for {samples[index]['path']}: {raw.tolist()}; "
                    f"input_min={float(tensor.min())}, input_max={float(tensor.max())}, input_mean={float(tensor.mean())}"
                )
            outputs.append(output_probabilities(raw))
    array = np.asarray(outputs, dtype=np.float32)
    if array.shape != (len(indices), 3) or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid source output array: {array.shape}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)
    print(json.dumps({"status": "PASS", "backend": "source_pytorch", "shape": list(array.shape)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
