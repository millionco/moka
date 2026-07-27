# Experiment log

This notebook records measurements, failures, interpretations, and next experiments for the Moka 9×9 Go model. Results are not promoted from hypothesis to conclusion without a reproducible measurement.

## 2026-07-26 — Baseline and deployment budget

### Question

Can the miniature board use a competent learned policy without downloading the current multi-megabyte KataGo and ONNX Runtime payload?

### Environment

- Apple Silicon arm64
- macOS 26.3.2
- Node.js 25.8.2
- MLX 0.32.0
- Python 3.12.13, provisioned by `uv`
- ONNX Runtime 1.28.0 for teacher inference
- KataGo b6c96 teacher

### Baseline measurements

| Artifact                        |   Raw bytes | Gzip bytes |
| ------------------------------- | ----------: | ---------: |
| KataGo b6c96 ONNX               |   4,131,496 |  3,823,306 |
| ONNX Runtime SIMD/threaded WASM |  12,810,620 |  3,283,537 |
| ONNX Runtime WebGPU/JSEP WASM   | about 25 MB |  6,081,900 |

The existing browser path therefore has a cold payload of about 16.9 MB raw before JavaScript glue. Switching ONNX Runtime from WASM to WebGPU increases the runtime payload.

### Teacher contract inspection

The b6c96 ONNX file contains 1,026,510 parameters and 150 graph nodes.

Inputs:

- `input_spatial`: batch × 22 × height × width
- `input_global`: batch × 14

Outputs:

- `policy`: batch × 1 × height × width
- `policy_pass`: batch × 1
- `value`: batch × 3
- `score_value`: batch × 2
- `ownership`: batch × 1 × height × width

On an empty 9×9 board, `policy` and `policy_pass` behaved as logits rather than normalized probabilities. The first two `value` outputs were treated as current-player win/loss logits; the third was effectively disabled near −5000.

### Hypothesis

A fixed-board residual student with approximately 90,000 parameters can retain useful local Go structure while fitting below 100 KB with per-output-channel INT8 weights. A dedicated Worker can keep all convolution work off the main thread without shipping a general tensor runtime.

### Design selected

- 10 relative input planes
- 32-channel 3×3 stem
- three 32-channel residual blocks
- four-channel policy head producing 82 moves
- two-channel value head
- 90,497 parameters

The float32 parameter payload is 361,988 bytes before serialization. The export budget is 200,000 bytes.

## 2026-07-26 — Toolchain setup

### Attempt that did not work

The system Python is 3.9.6. Current MLX requires Python 3.10 or newer, so the system interpreter was rejected as the project runtime.

### Successful setup

`uv sync` provisioned Python 3.12.13 and installed MLX 0.32.0, native ONNX Runtime 1.28.0, and NumPy 2.5.1 in an isolated environment.

### Additional constraint discovered

MLX does not provide the needed quantized `Conv2d` deployment path for this use case. Training remains in MLX, while export performs custom symmetric INT8 quantization per output channel. The browser dequantizes once into Worker-owned float arrays.

## 2026-07-26 — Rules and feature validation

### Protocol

Unit tests covered:

- surrounded-stone capture
- suicide rejection
- pass legality
- empty-board area score with seven-point komi
- student and teacher feature shapes
- side-to-move perspective
- pass history not creating a spatial recent-move marker

### Result

Six tests passed.

### Attempt that did not work

Python's `unittest discover` did not import filenames containing hyphens. The repository requires kebab-case, so the tests were changed to the valid single-word filenames `board.py` and `features.py` and are run explicitly.

## 2026-07-26 — Smoke distillation

### Protocol

```sh
uv run go-generate --positions 256 --output data/smoke.npz
uv run go-train --data data/smoke.npz --epochs 2 --batch-size 64 --checkpoint checkpoints/smoke.safetensors
uv run go-export --checkpoint checkpoints/smoke.safetensors --output dist/smoke
```

Teacher positions came from temperature-sampled KataGo self-play. The student optimized policy cross entropy plus weighted value mean-squared error.

### Observations

| Epoch | Training loss | Validation loss | Top-move agreement | Value MAE |
| ----: | ------------: | --------------: | -----------------: | --------: |
|     1 |        4.6035 |          4.4869 |               0.0% |    0.4082 |
|     2 |        4.4589 |          4.4329 |               8.0% |    0.3402 |

The exported weights were 92,912 bytes. The JSON manifest was 2,421 bytes. The weights compressed to 91,645 bytes with gzip.

### Interpretation

The losses decreased, the export remained within budget, and the whole pipeline executed successfully. The sample contains only 256 correlated self-play positions, so the move agreement is not evidence of useful playing strength.

### Conclusion

The engineering pipeline is viable. Model quality remains untested.

## 2026-07-26 — Inference performance

### Compiled MLX protocol

The smoke checkpoint ran 10 warmup inferences followed by 100 measured inferences with a batch size of one on a zero-valued 9×9 input.

| Metric |   Result |
| ------ | -------: |
| Mean   | 0.369 ms |
| p50    | 0.311 ms |
| p95    | 0.833 ms |

### Plain JavaScript protocol

The TypeScript runtime was compiled to JavaScript and run under Node.js. It loaded the quantized artifact, dequantized once, completed five warmups, and measured 50 single-position inferences.

| Metric |   Result |
| ------ | -------: |
| Mean   | 2.873 ms |
| p50    | 2.857 ms |
| p95    | 3.103 ms |

### Browser Worker protocol

The Vite benchmark loaded the manifest and weights over localhost in the in-app Chromium browser. Inference ran in a dedicated module Worker. Ten warmups preceded 100 measured request/response cycles, so results include Worker messaging.

| Metric         |    Result |
| -------------- | --------: |
| Initialization | 13.400 ms |
| Mean           |  3.213 ms |
| p50            |  3.200 ms |
| p95            |  3.500 ms |

### Interpretation

The Worker path is fast relative to the board's 500 ms move cadence and removes inference from the main thread. The approximately 0.3 ms difference between Node and the browser Worker is small and includes messaging overhead.

### Conclusion

Use the custom Worker JavaScript runtime as the first production path. Its measured latency is already well below the interaction budget.

## 2026-07-26 — WASM and WebGPU investigation

### WASM prototype

The raw Rust crate currently implements allocation, ReLU, and dense kernels without `wasm-bindgen`.

```sh
rustup target add wasm32-unknown-unknown
cargo build --manifest-path wasm/Cargo.toml --target wasm32-unknown-unknown --release
```

| Artifact       |   Raw bytes | Gzip bytes |
| -------------- | ----------: | ---------: |
| Prototype WASM | about 9,300 |      4,343 |

### What is not yet proven

The WASM crate does not yet implement the full convolutional network, and it has not beaten the 3.2 ms Worker JavaScript median. A small standalone kernel can look fast while losing end-to-end after memory copies and repeated JS/WASM crossings.

### WebGPU hypothesis

WebGPU is expected to be slower on this fixed 9×9 network because device initialization and multiple shader dispatches are large relative to roughly 90,000 weights and 81 spatial points. The available ONNX Runtime WebGPU path also increases compressed runtime cost to about 6.1 MB before the model.

### Conclusion

Do not ship WebGPU by default. Do not ship WASM based on artifact size alone. Add either backend only after a complete policy/value inference benchmark beats the Worker JavaScript path on desktop and mobile hardware.

## Next experiments

1. Generate at least 50,000 teacher positions, splitting validation by complete game rather than random position.
2. Record teacher top-move agreement, policy cross entropy, value MAE, and legal-move rate.
3. Run paired games against KataGo across deterministic and temperature-sampled settings.
4. Measure INT8 export drift against the MLX checkpoint on at least 1,000 held-out positions.
5. Implement one fused WASM convolution/residual path and compare complete inference, including transfers.
6. Test p50 and p95 Worker latency on a mid-range mobile device.
7. Replace the website's ONNX path only after the student clears agreed strength and latency gates.

## 2026-07-26 — Vite arena and first head-to-head

### Question

Can the compact student complete legal games against KataGo in the browser, and what obvious strength or behavioral failures appear?

### Training protocol

The arena student used 10,000 temperature-sampled KataGo self-play positions, 20 MLX epochs, and a batch size of 256.

```sh
uv run go-generate --positions 10000 --output data/arena.npz
uv run go-train --data data/arena.npz --epochs 20 --batch-size 256 --checkpoint checkpoints/arena.safetensors
uv run go-export --checkpoint checkpoints/arena.safetensors --output dist/arena
```

### Training observations

Validation loss declined from 4.5286 to 2.6347. Top-move agreement increased from 11.9% to 48.6%. Value MAE declined from 0.6216 to 0.2035. The exported model remained 92,912 bytes.

The validation split is position-level, so nearby positions from the same self-play game can occur in both training and validation sets. These metrics are useful for pipeline iteration but probably overestimate generalization.

### Vite integration failure

The first arena load failed before play:

```text
no available backend found
Failed to fetch dynamically imported module:
/onnxruntime/ort-wasm-simd-threaded.mjs?import
```

Vite recognized the ONNX Runtime module inside `public/` as an attempted source import and refused to transform it. A development-only Vite middleware now exposes the teacher runtime under `/teacher-runtime/` with explicit JavaScript and WASM content types. Reloading after that change initialized both engines successfully.

### Arena protocol

- shared TypeScript rules engine
- captures, suicide, pass, and simple ko enforced
- area scoring
- seven-point komi
- deterministic highest-policy legal move
- passing disabled before move 20 for both engines
- maximum 120 moves
- colors swapped for the second game

### Results

| Game | Black        | White        | Result         | Length |
| ---: | ------------ | ------------ | -------------- | -----: |
|    1 | Moka         | KataGo b6c96 | KataGo by 6.0  |     42 |
|    2 | KataGo b6c96 | Moka         | KataGo by 74.0 |    120 |

Series: KataGo 2, Moka 0.

### Interpretation

The first game was competitive by final area margin. The color-swapped game was not. It reached the move cap instead of two passes, showing that the student has not learned reliable game termination and probably cycles through low-value legal moves in settled positions.

Two deterministic games are insufficient for a strength estimate. They are sufficient to reject the hypothesis that the 10,000-position student is already a dependable replacement for KataGo.

### Next arena changes

1. Group dataset samples by game and hold out complete games.
2. Increase late-game and pass examples rather than only increasing total positions.
3. Distill ownership or score in addition to win/loss value.
4. Add positional superko or state-repeat detection to the arena.
5. Run at least 100 color-balanced games with stochastic openings.
6. Report win rate with a confidence interval and separate move-cap adjudications.

## 2026-07-26 — Web KaTrain audit

### Question

Can Web KaTrain provide reusable rules tests, compact training data, or a materially smaller teacher?

### Sources inspected

- Web KaTrain engine documentation and worker integration
- game, liberty, scoring, and pass-history tests
- bundled KataGo model and SGF assets
- upstream commit `7a0a4876ed0577bac3e511df4938ba5223446e6a`

### What transferred

Pass history was a real omission in the first student representation. Two explicit recent-pass planes increased the feature count from 10 to 12. Shared-liberty, enclosed-area, and recent-pass regression cases were adapted under the upstream MIT license.

### What did not transfer

The bundled model is roughly 3.6 MB and the available SGFs are 19×19. Neither meets the 9×9, hundreds-of-kilobytes deployment goal. The useful output was rules coverage, not a deployment artifact.

## 2026-07-26 — Symmetry and capacity experiment

### Protocol

- 10,000 KataGo self-play positions
- complete-game validation split
- rotations and reflections sampled during training
- residual block count increased from three to four
- 30 epochs, batch size 256

### Observations

Validation loss reached 2.683, top-move agreement reached 46.8%, and value MAE reached 0.329. The INT8 artifact increased from 92,912 to 112,432 bytes. Compiled MLX inference remained below one millisecond at p95.

### Interpretation

Board symmetry and one extra residual block improved fit while remaining comfortably inside the 200 KB budget. The validation split still needed a stable untouched test bucket before further iteration.

## 2026-07-26 — Training-method audit from `~/Developer/brain`

### Question

Which recent post-training ideas fit a 109,569-parameter Go policy/value network?

### Sources inspected

- self-distillation cluster: SDFT, SDPO, and on-policy self-distillation
- Goldilocks RL
- dense credit-assignment work: RLTT, ADMIRE, ICA, and OpAgent
- search self-play

### Selected translation

The applicable mechanism is DAgger-style on-policy distillation. The student generates its own state distribution, while KataGo supplies dense full-policy and value targets on those states. Goldilocks sampling translates into a bell-shaped sample weight that emphasizes middle-difficulty disagreements instead of already-solved or overwhelming positions. Original teacher self-play positions remain as replay.

### Ideas deferred

SDFT's privileged-context self-teacher does not map directly to a small convolutional network. MASPO and GRPO solve scalar-reward optimization problems that are unnecessary while exact KataGo policy targets are available. Search self-play becomes useful after the policy is strong enough to produce informative games without extensive teacher correction.

## 2026-07-26 — On-policy distillation experiment

### Hypothesis

Offline teacher self-play misses states induced by student mistakes. Adding weighted teacher labels on the student's own trajectories should improve held-out policy and value quality without changing model size.

### Baseline protocol

- primary dataset: 10,000 positions
- stable split by game ID bucket: 7,816 train, 1,053 validation, 1,131 test
- four residual blocks and symmetry augmentation
- 30 epochs

### Baseline result

| Metric                  | Result |
| ----------------------- | -----: |
| Test loss               | 2.9013 |
| Test top-move agreement |  41.7% |
| Test value MAE          | 0.4257 |

### On-policy collection

- 10,000 positions reached by the baseline student
- KataGo full-policy and value labels at every position
- 15% teacher intervention during rollouts
- Goldilocks sample weighting by probability assigned to KataGo's top move
- mean sample weight 1.116

The baseline agreed with KataGo's top move on only 36.9% of its own visited states. This confirms a distribution shift between teacher self-play and student rollouts.

### Candidate result

| Metric                  |      Baseline | On-policy candidate |      Change |
| ----------------------- | ------------: | ------------------: | ----------: |
| Test loss               |        2.9013 |              2.5608 |      −11.7% |
| Test top-move agreement |         41.7% |               46.3% | +4.6 points |
| Test value MAE          |        0.4257 |              0.2922 |      −31.4% |
| INT8 weights            | 112,432 bytes |       112,432 bytes |   unchanged |

### Conclusion

The hypothesis was supported on all untouched test metrics. The on-policy checkpoint replaced the arena artifact.

## 2026-07-26 — 100-game arena and value calibration

### Protocol

- 50 deterministic four-move openings
- each opening played twice with colors swapped
- deterministic highest-policy legal move
- 120-move cap
- opening-position win probability recorded from both models

### Result

| Metric             | Previous student | On-policy student |
| ------------------ | ---------------: | ----------------: |
| Moka wins          |                1 |                 2 |
| KataGo wins        |               99 |                98 |
| Move caps          |               27 |                23 |
| Moka Brier score   |     not recorded |             0.259 |
| KataGo Brier score |     not recorded |             0.348 |
| Run time           |     not recorded |      19.4 seconds |

### Interpretation

The direction is positive but the student remains far weaker than KataGo. Brier scores are measured against this arena's area-scored outcomes after fixed openings. They should not be interpreted as a general comparison of value-head quality because the arena omits positional superko and KataGo search.

## 2026-07-26 — Deferred website runtime

### Question

Can the 112 KB student replace the 4.1 MB ONNX teacher on the main site without affecting initial-page performance?

### Implementation

- model initialization is scheduled 1.5 seconds after hydration through an idle callback
- the 112,432-byte INT8 artifact is stored as a deterministic gzip file
- the browser streams, decompresses, and verifies the model before transferring it to a dedicated worker
- the worker is compiled into a 4,267-byte JavaScript asset
- model files use immutable one-year browser caching

### Result

| Artifact         |  Uncompressed |  Transferred |
| ---------------- | ------------: | -----------: |
| INT8 weights     | 112,432 bytes | 99,718 bytes |
| Inference worker |   4,267 bytes |  1,611 bytes |

A clean browser reload advanced from queued to move 5 within four seconds, including the intentional 1.5-second deferral. No browser warnings or errors were emitted. The earlier TypeScript worker asset failed because browsers classify `.ts` as an MPEG transport stream; precompiling it to JavaScript eliminated that failure.

### Conclusion

The main page now uses only the student model, keeps inference off the main thread, and avoids loading model code during the initial render. Gzip saves 11.3% on the already-quantized weights.

## 2026-07-26 — 9×9 specialist distillation

### Hypothesis

A teacher finetuned exclusively on diverse 9×9 positions should provide a stronger signal than the general b6c96 teacher without increasing Moka's deployed size.

### Teacher

- KataGo `kata9x9-b18c384nbt-20231025`
- 18 nested bottleneck blocks and 384 trunk channels
- 28,864,714 parameters
- SWA checkpoint evaluated through PyTorch on Apple Metal
- 20,000 teacher-policy self-play positions across 322 games
- policy-surprise replay weighting: half uniform and half proportional to teacher–Moka KL divergence

The teacher checkpoint and PyTorch are local training dependencies only. Neither enters the website bundle.

### Result

| Metric                  | On-policy baseline | Specialist candidate |
| ----------------------- | -----------------: | -------------------: |
| Test loss               |             2.5608 |               2.4843 |
| Test top-move agreement |              46.3% |                51.0% |
| Test value MAE          |             0.2922 |               0.2775 |
| INT8 weights            |      112,432 bytes |        112,432 bytes |
| 100-game arena          |               2–98 |                 2–98 |

### Interpretation

The stronger teacher improved all fixed test metrics but not game strength. Greedy play compounds a small number of policy errors over a complete game, so static teacher self-play does not cover the states induced by those errors.

## 2026-07-26 — Search and specialist DAgger

### Search probes

| Method                          | Games | Moka–KataGo | Move caps |
| ------------------------------- | ----: | ----------: | --------: |
| Greedy policy                   |   100 |        2–98 |        26 |
| 16-visit PUCT                   |    20 |        1–19 |        10 |
| Eight-candidate value lookahead |    20 |        1–19 |         8 |

Neither low-budget search method justified its latency. PUCT was too shallow for an 82-action root, while one-step lookahead exposed insufficient value accuracy.

### On-policy correction

Moka agreed with the 9×9 specialist on 43.3% of 15,000 positions reached by Moka rollouts. Fifteen percent of rollout moves used teacher intervention, and all positions were relabeled with the full specialist policy and value. Whole games assigned to validation or test buckets were excluded during training.

### Result

| Metric                  | Specialist | Specialist DAgger |
| ----------------------- | ---------: | ----------------: |
| Test loss               |     2.4843 |            2.4735 |
| Test top-move agreement |      51.0% |             49.1% |
| Test value MAE          |     0.2775 |            0.2707 |
| 100-game arena          |       2–98 |              9–91 |
| Move caps               |         26 |                20 |

### Interpretation

The lower static top-move agreement but higher arena win rate confirms that correcting Moka's own state distribution matters more than matching average held-out positions. A second DAgger iteration is warranted.
