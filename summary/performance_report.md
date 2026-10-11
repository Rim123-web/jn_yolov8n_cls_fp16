# Performance

{
  "measured_images": 261,
  "warmup_count": 10,
  "mean_engine_latency_ms": 22.81140761539854,
  "mean_transfer_inclusive_latency_ms": 25.698645162742256,
  "mean_end_to_end_latency_ms": 201.58743890804598,
  "engine_fps": 43.83771562281686,
  "transfer_inclusive_fps": 38.912557205536814,
  "latency_based_pipeline_fps": 4.960626542093972,
  "official_measured_wall_time_seconds": 52.745892150000145,
  "wall_clock_pipeline_fps": 4.948252638475834,
  "engine_percentiles_ms": {
    "p50": 21.62661361694336,
    "p90": 21.72901153564453,
    "p95": 25.38880157470703,
    "p99": 54.15899047851491
  },
  "end_to_end_percentiles_ms": {
    "p50": 103.094556,
    "p90": 164.038506,
    "p95": 216.045512,
    "p99": 2339.0288763999934
  },
  "fps_definitions": {
    "engine_fps": "1000 / mean TensorRT execution-only CUDA-event latency",
    "transfer_inclusive_fps": "1000 / mean(H2D + TensorRT + D2H)",
    "latency_based_pipeline_fps": "1000 / mean complete file-open-to-classification latency",
    "wall_clock_pipeline_fps": "N / official measured process wall time"
  }
}
