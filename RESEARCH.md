# Moka

This repository distills the existing KataGo b6c96 teacher into a small, fixed 9×9 policy/value network for the miniature board.

The deployment target is not ONNX. The browser artifact uses per-output-channel INT8 weights and a dedicated Worker, so the page does not need the 3.9 MB teacher or the 12.2 MB ONNX Runtime WASM binary.

## Architecture

- 12 input planes: current/opponent stones, one- and two-liberty groups, ko, two recent moves, two recent passes, and perspective komi
- 32-channel 3×3 stem
- four 32-channel residual blocks
- 82-way policy head, including pass
- current-player value head
- 109,569 parameters
- hard export budget: 200 KB

MLX trains the float model on Apple Silicon. KataGo runs through native ONNX Runtime only while generating distillation targets. Export quantizes each convolution and linear output channel independently.

## Setup

From this directory:

```sh
uv sync
```

The project pins Python 3.12 through `uv`; the system Python is not used.

## Train and export

Generate teacher positions:

```sh
uv run go-generate --positions 50000
```

Collect teacher labels on positions reached by the current Moka checkpoint:

```sh
uv run go-collect \
  --checkpoint checkpoints/moka-model.safetensors \
  --positions 10000 \
  --output data/on-policy.npz
```

Train Moka:

```sh
uv run go-train \
  --data data/katago-distillation.npz \
  --supplemental-data data/on-policy.npz \
  --epochs 30
```

Export the browser artifact:

```sh
uv run go-export
```

Measure compiled MLX inference:

```sh
uv run go-benchmark
```

For a fast pipeline check, use 512 positions and two epochs:

```sh
uv run go-generate --positions 512 --output data/smoke.npz
uv run go-train --data data/smoke.npz --epochs 2 --checkpoint checkpoints/smoke.safetensors
uv run go-export --checkpoint checkpoints/smoke.safetensors --output dist/smoke
```

Generate search-improved labels with the native 9×9 specialist:

```sh
uv run go-search-generate \
  --checkpoint checkpoints/moka-dgrpo-prefix-v1.safetensors \
  --games 128 \
  --visits 64 \
  --output data/moka-search-distillation.npz
```

## Browser runtime decision

| Path                      |                 Download cost | Main-thread impact             | Expected fit                                              |
| ------------------------- | ----------------------------: | ------------------------------ | --------------------------------------------------------- |
| Current ONNX Runtime WASM |    about 16 MB raw with b6c96 | low after initialization       | Too large                                                 |
| Custom Worker JavaScript  |           model plus a few KB | none during inference          | Default                                                   |
| Custom WASM kernels       | model plus roughly tens of KB | none when hosted by the Worker | Add only if measured faster                               |
| WebGPU                    |        model plus shader code | low                            | Dispatch overhead is likely dominant for this 9×9 network |

`web/runtime.ts` is the reference implementation and dequantizes the roughly 100 KB artifact once inside the Worker. This increases Worker memory to about 400 KB but keeps the network payload small. `web/client.ts` transfers feature and policy buffers rather than cloning them.

Run the Worker benchmark from the repository root:

```sh
nr arena
```

Then open `http://localhost:4174/web/benchmark.html`.

The playable model arena is at `http://localhost:4174/web/arena.html`. It can run 100 color-balanced games concurrently, enforces captures, suicide, pass, and simple ko, and reports every winner, score margin, game length, termination reason, and opening-position win prediction. Run history includes Brier scores for both value heads.

## On-policy distillation

`go-collect` implements a compact DAgger-style correction loop:

1. Roll out Moka's policy with a small amount of KataGo intervention.
2. Ask KataGo for a full 82-move policy and current-player value at every visited state.
3. Give the largest sample weight to middle-difficulty positions where Moka has useful room to improve.
4. Mix those samples with the original KataGo self-play replay set during training.

The validation and test sets remain complete held-out games from the primary dataset. Supplemental on-policy positions contribute only to training, so test comparisons remain stable.

The Rust crate contains the first raw WASM kernels without `wasm-bindgen` runtime overhead. Keep it behind a benchmark: a 9×9 model has small tensors, so crossing the JS/WASM boundary for every layer can erase SIMD gains. WebGPU should also remain optional until an end-to-end trace beats the Worker path on both Apple Silicon and a mid-range mobile device.

Build the WASM prototype:

```sh
rustup target add wasm32-unknown-unknown
cargo build --manifest-path wasm/Cargo.toml --target wasm32-unknown-unknown --release
```

## Quality gates

A production run should use at least 50,000 diverse positions. Before replacing KataGo on the site, hold out complete games rather than random positions and record:

- teacher top-move agreement
- policy cross entropy
- value mean absolute error
- legal-move rate
- 9×9 games against the teacher at several temperatures
- cold download, initialization, p50, and p95 move latency

The smoke model proves the pipeline and size budget only. It is not a strength benchmark.

## Tests

```sh
uv run python tests/board.py
uv run python tests/features.py
uv run python tests/symmetry.py
uv run python tests/collect.py
uv run python tests/policy.py
uv run python tests/search-generate.py
```
