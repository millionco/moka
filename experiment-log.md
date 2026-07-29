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

A fixed-board residual Moka network with approximately 90,000 parameters can retain useful local Go structure while fitting below 100 KB with per-output-channel INT8 weights. A dedicated Worker can keep all convolution work off the main thread without shipping a general tensor runtime.

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
- Moka and teacher feature shapes
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

Teacher positions came from temperature-sampled KataGo self-play. Moka optimized policy cross entropy plus weighted value mean-squared error.

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
7. Replace the website's ONNX path only after Moka clears agreed strength and latency gates.

## 2026-07-26 — Vite arena and first head-to-head

### Question

Can Moka complete legal games against KataGo in the browser, and what obvious strength or behavioral failures appear?

### Training protocol

The arena Moka checkpoint used 10,000 temperature-sampled KataGo self-play positions, 20 MLX epochs, and a batch size of 256.

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

The first game was competitive by final area margin. The color-swapped game was not. It reached the move cap instead of two passes, showing that Moka has not learned reliable game termination and probably cycles through low-value legal moves in settled positions.

Two deterministic games are insufficient for a strength estimate. They are sufficient to reject the hypothesis that the 10,000-position Moka checkpoint is already a dependable replacement for KataGo.

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

Pass history was a real omission in Moka's first representation. Two explicit recent-pass planes increased the feature count from 10 to 12. Shared-liberty, enclosed-area, and recent-pass regression cases were adapted under the upstream MIT license.

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

The applicable mechanism is DAgger-style on-policy distillation. Moka generates its own state distribution, while KataGo supplies dense full-policy and value targets on those states. Goldilocks sampling translates into a bell-shaped sample weight that emphasizes middle-difficulty disagreements instead of already-solved or overwhelming positions. Original teacher self-play positions remain as replay.

### Ideas deferred

SDFT's privileged-context self-teacher does not map directly to a small convolutional network. MASPO and GRPO solve scalar-reward optimization problems that are unnecessary while exact KataGo policy targets are available. Search self-play becomes useful after the policy is strong enough to produce informative games without extensive teacher correction.

## 2026-07-26 — On-policy distillation experiment

### Hypothesis

Offline teacher self-play misses states induced by Moka's mistakes. Adding weighted teacher labels on Moka's own trajectories should improve held-out policy and value quality without changing model size.

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

- 10,000 positions reached by the baseline Moka checkpoint
- KataGo full-policy and value labels at every position
- 15% teacher intervention during rollouts
- Goldilocks sample weighting by probability assigned to KataGo's top move
- mean sample weight 1.116

The baseline agreed with KataGo's top move on only 36.9% of its own visited states. This confirms a distribution shift between teacher self-play and Moka rollouts.

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

| Metric             | Previous Moka | On-policy Moka |
| ------------------ | ------------: | -------------: |
| Moka wins          |             1 |              2 |
| KataGo wins        |            99 |             98 |
| Move caps          |            27 |             23 |
| Moka Brier score   |  not recorded |          0.259 |
| KataGo Brier score |  not recorded |          0.348 |
| Run time           |  not recorded |   19.4 seconds |

### Interpretation

The direction is positive but Moka remains far weaker than KataGo. Brier scores are measured against this arena's area-scored outcomes after fixed openings. They should not be interpreted as a general comparison of value-head quality because the arena omits positional superko and KataGo search.

## 2026-07-26 — Deferred website runtime

### Question

Can the 112 KB Moka model replace the 4.1 MB ONNX teacher on the main site without affecting initial-page performance?

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

The main page now uses only Moka, keeps inference off the main thread, and avoids loading model code during the initial render. Gzip saves 11.3% on the already-quantized weights.

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

## 2026-07-27 — Fixed protocol and compact-network sweep

### Protocol

- 100 deterministic development games from 50 four-move openings
- each opening played with colors swapped
- KataGo b6c96 ONNX opponent
- whole-game split by `game_id % 10`
- validation bucket 0, test bucket 1
- no development-arena opening inserted into training
- final opening-offset arena reserved until a development candidate exceeds 50 wins

The 100-game development arena is used for method rejection. Test metrics are reported after validation-based checkpoint selection and are never optimized directly.

### Results

| Candidate                         | Parameters | Moka–KataGo | Conclusion |
| --------------------------------- | ---------: | ----------: | ---------- |
| Legacy static Moka                | about 110k |        2–98 | rejected   |
| Legacy specialist DAgger          | about 110k |        9–91 | rejected   |
| 12-block nested, soft targets     |    104,129 |        9–91 | rejected   |
| 12-block nested DAgger            |    104,129 |       10–90 | rejected   |
| 12-block nested, 50% hard targets |    104,129 |   **16–84** | incumbent  |
| 12-block nested, 80% hard targets |    104,129 |       14–86 | rejected   |
| Context input planes              |    105,281 |        5–95 | rejected   |
| Recurrent tied-depth trunk        |    104,129 |       12–88 | rejected   |
| Spatial policy and ownership head | about 100k |        8–92 | rejected   |
| Wide bottleneck trunk             |    190,929 |        8–92 | rejected   |

The 50% hard-target nested checkpoint is `moka-nested-hard50-v1.safetensors`. It won five games as black and eleven as white, with nine move caps.

### Interpretation

The bottleneck architecture helped only when paired with a mixed hard/soft teacher objective. Larger capacity, more history, recurrence, ownership, and a fully convolutional policy head did not transfer into game strength.

## 2026-07-27 — Data weighting, divergence, and search sweep

### Distillation variants

| Method                          | Moka–KataGo |
| ------------------------------- | ----------: |
| Opponent-aware DAgger fine-tune |       12–88 |
| Phase weighting                 |       13–87 |
| Error-focused replay            |       11–89 |
| Teacher top-four policy         |       12–88 |
| 50% reverse KL                  |       15–85 |
| 25% reverse KL                  |       12–88 |
| Pure on-policy reverse KL       |        7–93 |
| α–β divergence, α=0.8, β=0.3    |        5–95 |

α–β divergence reached 52.5% held-out top-move agreement but only five arena wins. The result reinforces that average one-step agreement is a weak selection proxy for an autoregressive game policy.

### Search variants

| Method                         | Result |
| ------------------------------ | -----: |
| PUCT, 32 visits                |   5–15 |
| PUCT, 64 visits                |   6–14 |
| PUCT, 128 visits               |   7–13 |
| PUCT, 256 visits               |   7–13 |
| One-step value lookahead       |   3–17 |
| Flat terminal rollouts         |    1–9 |
| Handcrafted tactical reranking |  13–87 |

PUCT plateaued because the learned value remained inaccurate and the root branching factor was large. Handcrafted capture, atari, liberty, and self-atari bonuses also reduced strength.

## 2026-07-27 — Auxiliary and direct-outcome experiments

| Method                                 | Moka–KataGo | Notes                        |
| -------------------------------------- | ----------: | ---------------------------- |
| Frozen-trunk outcome value calibration |        4–16 | 64-visit probe               |
| PPO, policy head only, iteration 1     |       16–84 | changed color balance        |
| PPO, policy head only, iteration 2     |       13–87 | fresh games                  |
| PPO, full network, iteration 1         |       15–85 | 10⁻⁵ learning rate           |
| PPO, full network, iteration 2         |       10–90 | 2,000 fresh games            |
| Two-ply planning head, weight 0.25     |       13–87 | training-only auxiliary head |
| Two-ply planning head, weight 0.05     |       10–90 | conservative ablation        |

Neither the learned value nor direct PPO outcome optimization improved the incumbent. The planning head increased held-out agreement to 56.7% but harmed games, again separating static imitation from sequential strength.

## 2026-07-27 — Capacity and representation checks

### Capacity

A 995,745-parameter, 20-block assistant was trained with the same data and target mix. After a lower-learning-rate continuation it reached 58.1% held-out move agreement, but scored only 9–91. A 103,233-parameter depthwise-expanded network scored 3–97. Capacity alone is not the limiting variable.

### Retrieval and ensembling

- a five-checkpoint probability ensemble scored 9–91
- a 4,000-entry symmetry-canonicalized nearest-neighbor memory achieved 9.9% validation agreement versus Moka's 56.1%
- every validation-selected retrieval distance threshold reduced hybrid agreement

The errors are correlated across neural checkpoints, while compressed state retrieval does not generalize teacher moves.

### Tactical input planes

Eight computed planes encoded three-liberty groups, area ownership, captures, post-move liberties, ataris, and self-atari. They added only 2,304 stem parameters but scored 5–95. The feature branch was removed.

## 2026-07-27 — GRPO

### Motivation

PPO depended on Moka's inaccurate value baseline. GRPO can replace the critic with relative rewards among sampled continuations from the same start.

### Protocol

- 1,024 fresh games per iteration
- same-color groups of eight
- win/loss reward plus a bounded ±0.1 area-margin term
- reward normalization within each group
- full-network clipped update
- sampled-action KL weight 0.04
- entire groups assigned to train, validation, or test buckets

All 128 first-iteration groups had nonzero reward variance.

### Result

| Iteration            | Moka–KataGo |
| -------------------- | ----------: |
| GRPO 1               |       15–85 |
| GRPO 2, fresh groups |       14–86 |

### Conclusion

GRPO produced a balanced policy but did not exceed the 16-win incumbent. A second fully on-policy iteration regressed, so coefficient tuning against the development arena was not pursued.

## 2026-07-27 — Data scale and continuation

### Question

Does substantially more fresh specialist-teacher self-play improve the fixed-size network, and can a conservative continuation preserve the incumbent's opponent-specific behavior?

### Protocol

- 300,000 new positions from the 9×9 b18c384 specialist
- seed 101, disjoint game-based validation and test buckets
- 100,000 prior specialist positions and 50,000 opponent-trajectory positions added only through their training buckets
- 12-block, 104,129-parameter nested network
- 50% hard and 50% soft policy objective

The first candidate trained from a fresh initialization for 15 epochs. The second continued the 16-win incumbent for three epochs at a learning rate of 0.0001.

### Result

| Candidate               | Test agreement | Moka–KataGo |
| ----------------------- | -------------: | ----------: |
| 16-win incumbent        | not remeasured |       16–84 |
| Fresh 400k-position run |   not selected |        5–95 |
| Low-rate continuation   |          57.4% |       12–88 |

The continuation's test loss was 2.1769 and its value MAE was 0.3581. It won three games as black and nine as white, with ten move caps.

### Conclusion

Data scale alone did not address sequential error accumulation. A conservative continuation retained more strength than fresh training but still regressed despite high one-step agreement. Neither candidate replaced the incumbent.

## 2026-07-27 — Batched symmetry augmentation

### Question

Can training preprocessing be accelerated without changing the sampled symmetries?

### Protocol

A 256-position synthetic batch used independently sampled rotations and flips. The original per-position path and the new grouped batch path each processed the same transformations for 100 repetitions. Exact array equality was tested against the scalar implementation.

### Result

| Implementation |    Time |
| -------------- | ------: |
| Scalar loop    | 0.240 s |
| Batched        | 0.055 s |
| Speedup        |   4.40× |

### Conclusion

The batched implementation is exactly equivalent on the test cases and removes a measurable Python preprocessing bottleneck. It changes training throughput, not the training distribution.

## 2026-07-27 — Four-bit quantization

### Question

Can Moka transfer substantially fewer than 100 KB while preserving the float checkpoint's policy?

### Protocol

The exporter packed two signed-offset INT4 weights per byte. Groupwise scales were tested at 16, 32, and 64 weights. Drift was measured on 2,000 held-out positions against the unquantized checkpoint. The production INT8 artifact was not changed.

### Result

| Quantization                                |       Raw size | Top-move agreement | Policy KL |
| ------------------------------------------- | -------------: | -----------------: | --------: |
| INT4, per output channel                    |       57,852 B |              48.2% |     0.500 |
| INT4, groups of 32                          |       70,348 B |             64.75% |     0.159 |
| INT4, groups of 16                          |       83,896 B |              65.7% |     0.130 |
| Mixed INT8 heads/stem, INT4 trunk, group 64 | about 79,664 B |             79.65% |     0.052 |

MSE-optimized clipping reduced the mixed-precision policy KL to 0.042 but did not materially improve top-move agreement.

### Conclusion

Naive all-layer INT4 is too destructive. Mixed precision approaches the requested transfer size but still changes roughly one in five greedy moves, so it is not release quality. INT4 remains an experimental exporter option; the website continues to serve the verified INT8 model.

## 2026-07-27 — Distillation-guided GRPO

### Motivation

The first GRPO experiment compared continuations only from the empty board and relied on sparse terminal rewards. The revised method gives each same-color group a shared, teacher-sampled prefix and combines group-relative outcome optimization with dense teacher policy distillation on every Moka state.

### Protocol

- 2,048 games from 128 independent prefixes
- prefix lengths sampled from 0, 4, 8, 12, and 16 moves
- eight same-color continuations per prefix
- 256 complete groups, with 204 training, 26 validation, and 26 test groups
- no group split across data partitions
- one clipped full-network update at a learning rate of 0.00001
- old-policy KL weight 0.04 and teacher-distillation weight 0.2

The dataset contained 174,543 positions and 87,070 Moka actions. Every group had nonzero shaped-reward variance.

### Result

| Candidate                  | Moka–KataGo | Black wins | White wins | Move caps |
| -------------------------- | ----------: | ---------: | ---------: | --------: |
| Hard-target incumbent      |       16–84 |          5 |         11 |         9 |
| Distillation-guided GRPO 1 |   **19–81** |          8 |         11 |         7 |
| Distillation-guided GRPO 2 |       17–83 |          9 |          8 |         9 |
| b18-guided GRPO            |       16–84 |          7 |          9 |         6 |

### Conclusion

Dense teacher guidance and diverse shared prefixes converted GRPO from a regression into a modest improvement. A second iteration on 2,048 fresh games regressed. Replacing dense b6c96 guidance with the stronger raw 9×9 b18c384 policy on 1,024 fresh games also regressed. The checkpoint `moka-dgrpo-prefix-v1.safetensors` therefore remains the experimental incumbent. It remains below the website promotion threshold.

## 2026-07-27 — Fixup initialization

### Question

Does KataGo-style Fixup initialization improve Moka's 12-block residual trunk without adding deployed parameters?

### Protocol

The first three convolutions in each four-convolution residual branch were scaled by \(12^{-1/6}\), and the final expansion convolution was initialized to zero. The architecture, training data, objective, parameter count, and arena protocol otherwise matched the compact hard-target baseline.

### Result

| Metric                  | Result |
| ----------------------- | -----: |
| Test top-move agreement |  46.9% |
| Test value MAE          | 0.4504 |
| 100-game arena          |   4–96 |
| Move caps               |     29 |

### Conclusion

Fixup substantially regressed both policy quality and game strength. The experimental code path was removed after the measurement.

## 2026-07-27 — Native MCTS policy-improvement targets

### Question

Can low-visit native KataGo search provide a stronger dense target than either raw-policy distillation or terminal-reward GRPO?

### Protocol

- native 9×9 b18c384nbt KataGo specialist
- 64 whole Moka-versus-b6c96 trajectories
- 32 visits at every trajectory state
- 5,994 searched positions
- edge-visit policy targets and side-to-move root values
- whole-game split before training

The initial bridge incorrectly treated KataGo's returned GTP rows as top-origin. KataGo's explicit tuple input uses top-origin coordinates, but GTP output uses a bottom-origin row number. This vertically mirrored the policy targets without mirroring their features.

### Validation

| Target set | Mean probability on occupied points | Positions with occupied-point mass |
| ---------- | ----------------------------------: | ---------------------------------: |
| Mirrored   |                              40.05% |                              4,922 |
| Corrected  |                               0.00% |                                  0 |

The parser now has explicit `A9`, `J9`, `A1`, and `J1` regression cases. The saved target policies were repaired by a vertical board-axis flip without changing features, values, game IDs, or split assignments.

### Result

| Candidate                         | Search test agreement | Moka–KataGo |
| --------------------------------- | --------------------: | ----------: |
| Mirrored search-only              |                 21.9% |       0–100 |
| Corrected search-only             |                 51.1% |        5–95 |
| Corrected search plus 150k replay |                 52.2% |       12–88 |
| Distilled-GRPO incumbent          |        not remeasured |       19–81 |

The replay candidate used eightfold search-target sample weights, 100,000 specialist states, 50,000 opponent-trajectory states, and a learning rate of 0.00001.

### Conclusion

The coordinate invariant caught a severe data-generation defect, but corrected low-visit search imitation still did not improve game strength. Replay reduced catastrophic forgetting but did not eliminate it. None of the search candidates replaced the incumbent.

## 2026-07-27 — Process-aware GRPO

### Question

Can dense per-action teacher values fix GRPO's trajectory-level credit assignment without discarding the stable terminal objective?

### Protocol

- 1,024 fresh games in same-color groups of eight
- shared teacher-sampled prefixes
- whole-group train, validation, and test assignment
- one full-network epoch at a learning rate of 0.00001
- old-policy KL weight 0.04
- teacher policy-distillation weight 0.2
- teacher action advantage \(Q(s,a)-V(s)=-V(s')-V(s)\), clipped to \([-1,1]\)

Three variants were evaluated. Additive process GRPO adds one quarter of the action advantage to the normalized terminal advantage. Contribution-weighted GRPO instead uses the action advantage to scale the terminal advantage between 0.5× and 1.5×. Sample-routed GRPO applies 1.5× teacher distillation to below-group-mean trajectories and 0.5× to above-group-mean trajectories.

### Result

| Candidate                                  | Moka–KataGo | Black wins | White wins | Move caps |
| ------------------------------------------ | ----------: | ---------: | ---------: | --------: |
| Distilled-GRPO incumbent                   |       19–81 |          8 |         11 |         7 |
| Additive teacher action advantage          |       14–86 |          7 |          7 |         9 |
| Contribution weighting                     |       15–85 |          8 |          7 |         9 |
| Sample routing                             |       16–84 |          8 |          8 |        11 |
| Sample routing plus contribution weighting |       18–82 |         10 |          8 |         7 |

### Conclusion

Adding a noisy process reward directly was harmful. Preserving the terminal advantage's sign while reallocating its magnitude was safer. Sample routing and contribution weighting interacted positively but did not yet replace the incumbent. A fresh 2,048-game scale run is warranted; coefficient tuning against the fixed development arena is not.

### Scale and strict-routing follow-up

The combined soft-routing and contribution-weighting method was repeated on 2,048 fresh games. It scored 17–83. Strict sample routing, where actual Moka wins receive only GRPO and losses receive only teacher distillation, scored 15–85 both with and without contribution weighting. Only 8.1% of Moka action positions came from winning trajectories, so strict routing largely reduced to imitation. None of these candidates replaced the incumbent.

## 2026-07-27 — Greedy-distribution DAgger

### Question

Did prior on-policy collectors miss the deployed state distribution because they sampled Moka at temperature 0.85 while the arena and website use greedy moves?

### Protocol

- 50,000 positions from deterministic Moka-versus-b6c96 games
- b18c384nbt policy and value labels at every reached state
- no teacher interventions
- four-move openings beginning at offset 10,000
- 1,000 training openings checked against all 50 development openings: zero overlap
- whole-game train, validation, and test assignment
- conservative two-epoch continuation with 150,000 replay positions

### Result

Moka agreed with b18 on 47.9% of collected greedy-trajectory positions. The continued checkpoint reached 49.2% held-out agreement and scored 15–85 in the fixed arena, with seven black wins, eight white wins, and twelve move caps.

For an upper-bound check, the uncompressed b18 raw policy scored 17–3 against b6c96 across 20 color-balanced development games.

### Conclusion

The b18 teacher is strong enough to dominate b6c96, but deterministic state coverage alone did not transfer that strength into Moka. Greedy-distribution mismatch was real but not the sole bottleneck. The distilled-GRPO checkpoint remains the incumbent.

## 2026-07-27 — Auxiliary softened-policy head

### Question

Can KataGo's training-only softened-policy head force Moka's trunk to learn useful information about runner-up moves without increasing the deployed model?

### Protocol

- separate training-only policy convolution and linear head
- target equal to the teacher policy raised to \(1/4\) and renormalized
- unchanged deployed main policy and value heads
- auxiliary parameters stripped before arena evaluation
- 100,000 specialist states, 50,000 deterministic greedy DAgger states, and 50,000 opponent-trajectory states

The official nominal 8× auxiliary loss caused the newly initialized auxiliary head to dominate a pretrained continuation. Its first export also exposed an MLX non-strict-load assumption: reloading a checkpoint with extra tensors into the base model did not transfer the matching tensor tree correctly. The exporter now directly copies the 108 non-auxiliary parameter tensors into a fresh nested model.

### Result

| Candidate                           | Test agreement | Moka–KataGo | Move caps |
| ----------------------------------- | -------------: | ----------: | --------: |
| 8× auxiliary, invalid stripped file |          54.7% |       0–100 |       100 |
| 0.5× gradient-balanced auxiliary    |          56.5% |       17–83 |        11 |
| Distilled-GRPO incumbent            | not remeasured |       19–81 |         7 |

The invalid 0–100 result is an export-pipeline failure, not a model-strength measurement. After exact tensor-tree transfer, the balanced candidate produced normal outputs and game lengths.

### Conclusion

The separate soft head improved held-out one-step agreement but did not beat the incumbent. The official 8× coefficient does not transfer directly to a newly attached head on a pretrained 104k model. The deployed checkpoint stayed at 104,129 parameters.

## 2026-07-27 — Moka autoresearch harness

The Apple-Silicon harness adapts `karpathy/autoresearch` and `trevin-creator/autoresearch-mlx` to Moka:

- one mutable `autoresearch/experiment.py`
- fixed evaluator and immutable data-split rules
- experiment-content hashes and never-repeat ledger
- three training seeds by default
- bootstrap-confidence keep/discard gate
- ten-minute run timeout
- exact nested-model load, parameter-count, and checkpoint-size validation
- offset-1000 final arena excluded from automation

The baseline reproduced 19–81 with eight black wins, eleven white wins, and seven move caps on all three runs. Each checkpoint contained exactly 104,129 parameters.

## 2026-07-27 — Selective high-visit reanalysis

### Question

Can KataGo's playout-cap randomization idea transfer to distillation by spending the same approximate teacher-search budget on fewer, more surprising positions?

### Protocol

- 64 deterministic Moka-versus-b6c96 games from the separate distillation opening range
- 25% of turns selected for native b18c384 reanalysis
- half of selected turns sampled uniformly and half ranked by policy KL plus value disagreement
- 128 visits per selected turn
- 1,035 searched positions, including 818 training-eligible positions
- whole-game split with buckets 0 and 1 excluded from training
- sixteenfold replay of selected positions alongside 100,000 specialist and 50,000 opponent-trajectory positions
- two continuation epochs at a learning rate of 0.00001

### Validation

All target policies were finite and normalized. Occupied intersections received zero probability. The teacher policy had a mean support of 11.7 moves. The incumbent agreed with the selected teacher moves on 34.4% of training-eligible positions, with a mean policy KL of 1.30.

### Result

The candidate reached 38.0% move agreement on the selected test bucket and scored 16–84 in the fixed arena, with six black wins, ten white wins, and nine move caps.

### Conclusion

Selective deep search produced materially harder, cleaner labels than shallow search everywhere, but conservative replay still regressed from the 19–81 incumbent. This rejects the current policy-only use of selective reanalysis, not KataGo's broader method. Ownership, score, or searched child-value targets may be necessary for the selected states to teach a 104k network more than the improved visit distribution alone.

## 2026-07-27 — Training-only ownership supervision

### Question

Can KataGo's highest-impact auxiliary target improve Moka's shared trunk without increasing the deployed model?

### Protocol

- 50,000 deterministic Moka-versus-b6c96 positions from the separate distillation opening range
- b18c384 policy, value, ownership, short-value, and score predictions
- 100,000 specialist and 50,000 opponent-trajectory replay positions
- correctly rotated and reflected ownership maps under D4 augmentation
- training-only 1×1 ownership head stripped before evaluation
- two continuation epochs at a learning rate of 0.00001
- ownership loss weight 0.25

### Pipeline findings

Two existing defects were found before accepting a result. Spatial targets had not previously followed board symmetry augmentation, and auxiliary targets in a primary dataset were ignored while only supplemental datasets could contribute them. The first two candidate files therefore reproduced the no-auxiliary control exactly and are not ownership experiments. Regression coverage now verifies the spatial transformation, and primary auxiliary arrays are loaded explicitly.

### Result

The corrected ownership candidate contained the same 104,129 deployed parameters and scored 10–90 in the fixed arena, with two black wins, eight white wins, and six move caps.

### Conclusion

Ownership supervision at weight 0.25 destabilized the pretrained trunk and is rejected. The result does not invalidate ownership as a target; its contribution was roughly 14% of the initial combined objective, which may be too abrupt for a conservative continuation. A single gradient-balanced lower-weight follow-up is justified. Score supervision remains isolated because the teacher's raw score mean has rare estimates near ±96 points.

The gradient-balanced ownership follow-up used weight 0.05 and scored 14–86, with six black wins, eight white wins, and eleven move caps. It also failed to replace the incumbent, so no further ownership-weight tuning was performed.

### Search follow-up

The unchanged incumbent was probed with 16-simulation PUCT and four-candidate one-ply value lookahead. Both scored 4–16 over the same 20 development games, showing no directional gain over raw policy play. An eight-rollout probe was stopped because its latency was already unsuitable for the browser. The current value head is therefore not a useful search evaluator.

## 2026-07-27 — Searched child-Q auxiliary

### Question

Can current KataGo's per-child searched value targets teach the trunk more than root visit policies alone?

### Protocol

- the same 1,035-position selective 128-visit corpus as the policy-only experiment
- root-perspective child win values and edge-visit weights for every explored move
- about 11.7 searched children per position
- training-only four-channel Q head stripped before deployment
- Q loss weight 0.1
- sixteenfold selective-position replay alongside 150,000 established replay positions

### Result

The stripped candidate retained 104,129 parameters and scored 15–85, with seven black wins, eight white wins, and eleven move caps.

### Conclusion

Searched child-Q supervision did not replace the incumbent. Together with the soft-policy and ownership results, the tested training-only KataGo auxiliaries improve target richness but do not overcome Moka's sequential policy errors at the current continuation scale.

## 2026-07-27 — Wide distilled-GRPO

### Question

Did the earlier 190,929-parameter wide network fail because it received only supervised distillation rather than grouped outcome optimization?

### Protocol

- 2,048 fresh games and 256 same-color groups of eight
- shared b6c96-sampled prefixes
- zero zero-variance groups
- 171,836 positions with whole-group validation and test buckets
- one full-network distilled-GRPO epoch at a learning rate of 0.00001

### Result

The wide candidate scored 12–88, with three black wins, nine white wins, and nine move caps. This improved the earlier wide supervised result of 8–92 but remained below the compact 19–81 incumbent.

### Conclusion

Grouped outcome training helps the wider architecture, but extra capacity is not sufficient. The wide checkpoint is rejected before quantization work because its float policy is already weaker.

## 2026-07-27 — N-distill trajectory correction

### Question

Can the expected next-state distillation correction from Czarnecki et al. prevent the oscillation observed across repeated on-policy distilled-GRPO iterations?

### Protocol

- teacher cross-entropy evaluated at every next state on the stored student trajectories
- correction return equal to negative future next-state cross-entropy
- same-start, same-color group normalization
- correction weight 0.2, matched to the direct distillation coefficient
- original 2,048-game corpus from the 16-win hard-target baseline
- fresh second-iteration corpus from the 19-win distilled-GRPO incumbent

### Result

| Corpus           | Ordinary distilled-GRPO | N-distill |
| ---------------- | ----------------------: | --------: |
| First iteration  |                   19–81 |     16–84 |
| Second iteration |                   17–83 |     17–83 |

### Conclusion

The correction did not improve either iteration and did not prevent the measured second-iteration regression. The implementation is retained behind an explicit flag with a regression test, but no checkpoint is promoted.

## 2026-07-27 — List-wise policy distillation

### Question

Does a Plackett–Luce objective over the teacher's top eight moves transfer candidate ordering better than marginal KL and mixed hard/soft targets?

### Protocol

- sequential top-eight ranking likelihood over all remaining moves
- list-wise weight 0.1
- 50,000 deterministic greedy-trajectory positions
- 150,000 specialist and opponent-trajectory replay positions
- two continuation epochs at a learning rate of 0.00001

### Result

The candidate scored 16–84, with seven black wins, nine white wins, and eight move caps.

### Conclusion

List-wise ranking did not replace the incumbent. It is a distinct, tested objective, but the current failure mode is not explained solely by losing the teacher's relative action ordering.

## 2026-07-27 — Intermediate attention transfer

### Question

Can Moka learn b18's spatial representation directly without storing or deploying the teacher's 384-channel trunk?

### Protocol

- b18 `trunkfinal` channel energy collapsed to one 9×9 spatial map
- per-position L2 normalization
- 50,000 deterministic Moka-versus-b6c96 positions
- Moka attention derived directly from its existing 32-channel trunk
- attention-transfer loss weight 0.1
- no auxiliary deployed parameters
- three training seeds

### Result

| Seed | Moka–KataGo | Black wins | White wins | Move caps |
| ---: | ----------: | ---------: | ---------: | --------: |
|  293 |       19–81 |          8 |         11 |         5 |
|  307 |       14–86 |          6 |          8 |         6 |
|  311 |       11–89 |          2 |          9 |         8 |

The mean was 14.7 wins, so the incumbent tie at seed 293 was not robust. A fresh 2,048-game distilled-GRPO continuation from that tied seed scored 15–85.

### Conclusion

Attention transfer was the first output-independent representation method to tie the incumbent once, but the effect did not reproduce and did not combine positively with grouped outcome optimization. No attention checkpoint is promoted.

### Inference-temperature probe

The incumbent scored 3–17 at temperature 0.25 and 0–20 at temperatures 0.5, 0.85, and 1.2 over the same 20-game screen. The rejected stochastic runtime branch was removed; greedy play remains the deployment policy.

## 2026-07-27 — Forced counterfactual regret

### Question

Can Moka learn specifically from actions that its policy would choose but b18 would reject, using a searched value for both alternatives rather than ordinary policy imitation?

### Protocol

- four independent 64-game corpora from opening offset 10,000
- 128-visit b18 analysis at each selected root
- a second 128-visit analysis after Moka's forced greedy move
- 4,140 selected positions and 3,272 training-eligible positions
- 429 training positions with regret of at least 0.2
- 160 training positions where b18's preferred continuation was winning and Moka's forced continuation was losing
- smooth regret weighting, sparse critical weighting, and four-corpus critical replay

### Result

| Candidate                        | Moka–KataGo |
| -------------------------------- | ----------: |
| Smooth regret weighting          |       14–86 |
| Single-corpus critical weighting |       13–87 |
| Four-corpus critical replay      |       14–86 |

A training-only child-Q head fit to the combined searched targets produced a valid 130,911-parameter auxiliary checkpoint and a normal 104,129-parameter stripped checkpoint. Using the Q head directly at inference scored 0–100 with 98 move caps. Reranking the policy's top two, four, or eight actions with the Q head also scored 0–20 in screens.

### Conclusion

Forced counterfactual search isolated genuine high-regret errors, but heavily replaying those sparse states damaged the surrounding policy distribution. The learned Q head was not calibrated well enough to control play. No runtime Q path or checkpoint is retained.

## 2026-07-27 — Whole-game elite distillation

### Question

Does discarding every b18-versus-b6 trajectory that b18 fails to win preserve more of the specialist's playing strength than distilling from unfiltered positions?

### Protocol

- 2,048 deterministic b18-versus-b6 games from opening offset 30,000
- alternating b18 color and greedy play for both networks
- b18 policy and value labels at every reached state
- complete trajectories accepted only when b18 won
- 1,884 accepted games before balancing
- 1,840 exactly color-balanced games and 142,065 positions after balancing
- 1,469 training games, 183 validation games, and 188 test games
- zero overlap between training openings and the 100 development openings
- two-epoch continuation with 150,000 architecture-matched replay positions

### Result

The selected checkpoint reached 47.4% validation move agreement and 46.1% test move agreement on the elite corpus. It scored 15–85 in the fixed development arena, with eight black wins, seven white wins, and twelve move caps.

### Conclusion

b18 won 92.0% of the generated games, confirming that the teacher signal was strong. Filtering away its losing trajectories still did not transfer that strength into the compact network and reduced arena performance. The 19–81 distilled-GRPO checkpoint remains the incumbent.

## 2026-07-27 — Shared-depth continuation

### Question

Can executing the same 12-block trunk twice trade browser compute for effective depth without adding parameters or transfer bytes?

### Protocol

- unchanged 104,129-parameter nested network
- two passes through the weight-shared trunk
- 300,000 specialist positions
- 142,065 elite positions
- 50,000 deterministic on-policy positions
- eight continuation epochs with validation move-agreement selection

### Result

The selected recurrent checkpoint reached 56.1% validation move agreement and 56.0% test agreement. Despite those strong static metrics, it scored 5–95 in the fixed arena, with one black win, four white wins, and nine move caps.

### Conclusion

Weight sharing increased effective depth but amplified the same sequential policy failures seen in earlier high-agreement models. The recurrent checkpoint is rejected.

## 2026-07-27 — Batched PUCT and subtree reuse

### Question

Does Moka become competitive when the compact policy/value network is used as a search evaluator at a budget large enough to cover the 82-action root?

### Protocol

- incumbent `moka-dgrpo-prefix-v1.safetensors`
- PUCT with Moka policy priors and current-player values
- batched independent leaf evaluation with virtual reservations
- exact subtree reuse after Moka's selected move and KataGo's observed reply
- batch sizes 32 and 8
- simulation counts from 128 through 1,024
- fixed 20-game screens followed by one 100-game development arena

### Result

| Configuration                        | Moka–KataGo | Move caps |  Time |
| ------------------------------------ | ----------: | --------: | ----: |
| 128 visits, sequential               |        5–15 |         1 | 122 s |
| 512 visits, sequential               |        8–12 |         0 | 479 s |
| 512 visits, batch 32                 |        5–15 |         3 | 146 s |
| 512 visits, batch 8                  |        8–12 |         1 | 209 s |
| 512 visits, batch 8, subtree reuse   |       10–10 |         0 | 218 s |
| 1,024 visits, batch 8                |       10–10 |         1 | 480 s |
| 1,024 visits, batch 8, subtree reuse |       10–10 |         1 | 474 s |

The full 100-game development arena at 512 visits with batch 8 and subtree reuse scored 36–64, with 21 black wins, 15 white wins, and nine move caps. It took 1,318 seconds. This nearly doubled the greedy incumbent's 19 wins.

Batch 8 preserved the sequential 512-visit strength while reducing latency 2.3-fold. Batch 32 introduced too much parallel-search approximation. Subtree reuse provided a further two-win screen improvement at almost no additional wall time.

### Conclusion

Search is the first tested inference method to produce a large and robust playing-strength gain. It remains below the promotion threshold and is far too slow for the current browser animation. The final offset remains sealed.

## 2026-07-27 — Outcome-calibrated search value

### Question

Can a larger, current-incumbent outcome corpus fix the value bias limiting deep PUCT?

### Protocol

- 8,192 fresh Moka-versus-b6 games
- 717,717 positions
- shared randomized prefixes and alternating colors
- 6,552 training games, 820 validation games, and 820 test games
- terminal outcome and area-margin targets
- frozen stem, trunk, and policy head
- only the value convolution and linear layers updated

Every split was essentially 50% positive by position. Verification measured a maximum policy-logit difference of exactly zero between the incumbent and calibrated checkpoint.

### Result

The calibrated value head reached 0.4504 validation MAE and 0.4759 test MAE. At 512 visits with subtree reuse it scored 7–13, below the original value head's 10–10 on the same screen.

### Conclusion

Lower outcome MAE did not improve action ranking inside PUCT. The calibrated checkpoint is rejected, and the production policy remains byte-identical.

## 2026-07-27 — Expert Iteration

### Question

Can Moka distill its own stronger MCTS visit distribution and compound the search improvement in an AlphaZero-style policy-iteration step?

### Protocol

- 64 complete Moka-versus-b6 games from opening offset 40,000
- 256-visit MCTS targets on Moka turns
- b6 policy targets on opponent turns
- b6 value targets throughout
- 5,013 positions
- whole-game split masking
- sixteenfold training-only replay of the small search corpus
- 100,000 specialist and 50,000 opponent-trajectory replay positions

### Result

The selected checkpoint reached 56.3% validation and 56.2% test move agreement. It scored 11–89 greedily and 8–12 with 512-visit search on the fixed screen.

### Conclusion

The MCTS targets did not produce a stronger standalone policy or a stronger next search iteration. The checkpoint is rejected.

## 2026-07-27 — Search leaf and exploration ablations

### Independent validation protocol

Search-only hyperparameters were selected on opening offset 20,000, separate from training corpora, development offset 0, and sealed final offset 1,000.

### PUCT exploration

At 256 visits, exploration coefficients 0.5, 1.0, 3.0, and 6.0 scored 7, 4, 6, and 4 wins respectively over 20 games. The selected 0.5 coefficient regressed to five wins at 512 visits and was rejected.

### Area-value blending

A constant area-margin blend scored nine wins at weight 0.5 versus seven for the zero-weight control at 256 visits, but regressed to seven wins at 512 visits. Pure area value scored two wins. A phase-aware blend beginning at move 40 scored seven and six wins at maximum weights 0.5 and 1.0. All area blends were rejected.

### Policy-guided leaf rollouts

Hybrid PUCT with 64 tree simulations and deterministic policy rollouts scored 6–14 at rollout depth 8 and 4–16 at depth 16. Move caps increased to four and five. The ordinary value bootstrap remains stronger.

### Wide evaluator

The 190,929-parameter wide distilled-GRPO checkpoint scored 5–15 under the same 512-visit search, so extra evaluator capacity did not justify mixed INT4 deployment.

### Conclusion

The default exploration coefficient, original learned value, compact incumbent, and batch-8 subtree-reuse search remain the best measured combination. None clears the development gate.

## 2026-07-27 — Root allocation, symmetry, and adaptive depth

### Deterministic sequential halving

The root-allocation component of Gumbel AlphaZero was tested without stochastic Gumbel noise. Top-prior sets of eight and sixteen actions received explicit visits and were repeatedly halved by `log(prior) - Q`.

At 256 visits on search-validation offset 20,000, the eight-action variant scored 3–17 and the sixteen-action variant scored 4–16. Both eliminated move caps but were substantially weaker than the 7–13 ordinary-PUCT control. Moka's Q estimates are not accurate enough for aggressive elimination.

### Dihedral symmetry inference

All eight rotations/reflections were implemented with tested inverse policy transforms. The eight-way probability/value ensemble scored 2–18 greedily and 6–14 under 256-visit search. Every fixed orientation independently scored the same 2–18 aggregate. Symmetry averaging and fixed canonical orientation were rejected.

### Adaptive simulation budgets

The adaptive method starts with a fixed budget and extends only roots whose top-two visit margin remains below a threshold.

| Validation slice | Base / maximum visits | Margin | Moka–KataGo |  Time |
| ---------------- | --------------------: | -----: | ----------: | ----: |
| Offset 20,000    |           256 / 1,024 |    10% |        9–11 | 172 s |
| Offset 20,000    |           512 / 2,048 |    10% |       10–10 | 351 s |
| Offset 21,000    |           256 / 1,024 |    10% |        8–12 | 184 s |
| Offset 21,000    |           256 / 1,024 |    20% |        9–11 | 222 s |

The fresh offset-21,000 slice was reserved for confirmation after the first validation improvement. The broader margin improved one win over its 10% control but did not cross 50%.

### Conclusion

Adaptive depth is more efficient than uniformly increasing visits and repeatedly approached parity, but it did not satisfy the independent validation gate. Sequential halving and symmetry inference were clear regressions. None advances to development or the sealed final offset.

## 2026-07-27 — KataGo value-perspective audit

### Initial hypothesis

KataGo's checked-in analysis configuration contains `reportAnalysisWinratesAs = BLACK`, which appeared to imply that the generator had interpreted Black-perspective root and move values as side-to-move values.

### Falsification

The generator had already overridden the file with `reportAnalysisWinratesAs=SIDETOMOVE` when launching KataGo. Official analysis documentation states that all returned values follow this setting.

A direct native probe used the same White-to-move position with both reporting modes:

| Reporting mode | Current player | Root win rate | Best-move win rate |
| -------------- | -------------- | ------------: | -----------------: |
| Black          | White          |        0.0832 |             0.0828 |
| Side to move   | White          |        0.9164 |             0.9172 |

The override therefore supersedes the checked-in configuration exactly as intended. Root values and `moveInfos` Q values were already from the root player's perspective. Only a separately queried child root or a value attached to child-state features should be negated.

### Invalid branch

An attempted correction multiplied side-to-move values by board color, double-converting every White root. The resulting datasets, repaired aggregate, and all derivative Q, pointwise-value, and pairwise-ranking checkpoints are invalid measurements. Their arena scores ranged from 2–18 to 12–8 on small slices, but none may be used as evidence for or against the underlying methods.

The invalid 512-game generation was stopped after 240 analyzed games, before it wrote an output file. The correction utility and its command were removed. Regression coverage again asserts parent-to-child negation.

### Conclusion

The original selective-search policy, value, Q, and counterfactual datasets retain their intended perspective semantics. The next larger teacher corpus must use the restored side-to-move path and a disjoint opening range.

## 2026-07-27 — Fourfold selective reanalysis and autoresearch gate

### Corpus

The incumbent generated 512 deterministic games from opening offset 50,000. Selective native b18 analysis used 128 visits on 25% of positions, split evenly between uniform coverage and policy/value disagreement.

The resulting 4,238,527-byte corpus contained:

- 8,639 searched roots from 512 whole games
- 6,830 training roots, 806 validation roots, and 1,003 test roots
- 101,072 searched Q edges and 101,339 child states
- zero policy mass on occupied intersections
- finite, normalized policy, value, Q, and child arrays
- SHA-256 `f9d04e2c463736a742653d1a91b9f9109da4b00f8abbe185661cfd5e2f28dc05`

### Three-seed ablations

The MLX autoresearch harness trained each recipe with seeds 1, 2, and 3, validated the 104,129-parameter checkpoint, and ran the fixed 100-game development arena.

| Recipe                                         | Seed wins  |  Mean | Bootstrap P(better) | Decision |
| ---------------------------------------------- | ---------- | ----: | ------------------: | -------- |
| Full network plus training-only Q head         | 19, 14, 11 | 14.67 |               0.000 | discard  |
| Full network, policy targets only              | 19, 14, 12 | 15.00 |               0.000 | discard  |
| Frozen trunk/value, two-layer policy head only | 23, 18, 23 | 21.33 |               0.963 | keep     |
| Final policy projection only                   | 16         | 16.00 |         fast reject | discard  |

All supervised recipes used two epochs at learning rate 0.00001, 50,000 opponent-trajectory replay positions, and eightfold replay of training-eligible searched roots. The Q auxiliary used the same data and was stripped before evaluation.

### Seed averaging

The accepted head-only seeds shared byte-identical trunk and value weights. Averaging only their policy-head tensors avoided choosing a seed from arena outcomes. The averaged checkpoint scored 22–78 on the full greedy development arena, with eight Black wins, fourteen White wins, and nine caps.

At 512 visits on fresh offset 30,000, the averaged head regressed to 5–15 versus 9–11 for the original policy. It is therefore a greedy development candidate, not a search candidate.

Policy-head interpolation at 25%, 50%, and 75% was screened on offset 31,000. The incumbent and full soup each won one game; every intermediate blend won zero. Interpolation was rejected without confirmation.

## 2026-07-27 — Second selective DAgger and grouped outcomes

### On-policy reanalysis

The 22-win seed-averaged policy generated 256 fresh deterministic games at opening offset 60,000. The validated corpus contained 4,377 roots, 3,469 training roots, 403 validation roots, 505 test roots, and 50,967 searched Q edges. Its SHA-256 was `7100b2c523419f6d86457c3ae309d30600a3d6f342260c4cca9d1e660bdd966c`.

A balanced replay recipe used fourfold first-corpus replay and eightfold second-corpus replay, giving about 27,000 rows from each generation. Its first seed scored 20 wins, below the accepted 21.33 mean, and the autoresearch harness fast-rejected it.

### Distilled GRPO consolidation

Fresh grouped self-play from the 22-win policy produced 2,048 games, 185,416 positions, 92,567 Moka actions, and 256 whole groups. All groups had nonzero reward variance. Training, validation, and test contained 73,765, 8,906, and 9,896 Moka actions. The corpus SHA-256 was `78969eb6fd0fbc66d985f842e592f6260270451bc32202d457109da1cd12e9eb`.

One full-network distilled-GRPO epoch scored 13–87 on its first seed. Freezing the trunk and value head scored 14–86. Both were fast-rejected.

### Conclusion

Selective searched policy targets transfer only when representation drift is prevented. The gain is small but statistically accepted across training seeds. A second supervised DAgger iteration and fresh grouped outcome consolidation did not compound it.

## 2026-07-27 — Valid b18 search-value heads

### Child-state targets

Fitting only the incumbent value head to 101,339 b18-searched child states reached 0.3424 validation MAE and 0.3227 test MAE while preserving policy logits exactly. Despite the good pointwise fit, it scored 3–17 under 256-visit search on offset 33,000, versus 7–13 for the incumbent.

### Root targets

Fitting only the value head to 8,639 b18 root values reached 0.3727 validation MAE and 0.3312 test MAE. Policy logits again remained byte-identical.

| Fresh slice   | Visits | Incumbent | Root-value candidate |
| ------------- | -----: | --------: | -------------------: |
| Offset 34,000 |    256 |      5–15 |                 7–13 |
| Offset 35,000 |    512 |      6–14 |                 11–9 |

The candidate crossed 50% on the independent larger-budget confirmation, with seven Black wins, four White wins, and two caps. It advanced to the full 100-game 512-visit development arena. The offset-1,000 final arena remains sealed.

### Full development result

The root-value checkpoint scored 34–66 on the full 100-game 512-visit development arena, with 14 Black wins, 20 White wins, ten caps, and a 1,251.5-second runtime. This is below the original value head's 36–64 result, so the 11–9 confirmation did not generalize and the checkpoint is rejected.

A value-coefficient screen on fresh offset 36,000 used five color-paired openings per point. The incumbent scored 3–7. Candidate coefficients 0.25, 0.5, and 1.0 also scored 3–7; 1.5 scored 5–5; 2.0 scored 2–8; and 3.0 scored 4–6. The original 1.5 remained best, so scale retuning supplied no new candidate.

Value-head weight interpolation was screened on offset 37,000, where the incumbent scored 6–4. The 25% and 50% blends tied 6–4; the 75% blend and full b18 head scored 5–5. No blend advanced.

## 2026-07-27 — Exact searched-child ranking

### Data integrity

A fresh 256-game corpus from opening offset 70,000 added explicit root-row IDs to every searched child. It contained 4,224 roots, 49,462 children, and exact monotonic coverage of all roots. Whole-game splits contained 3,372 training roots, 377 validation roots, and 475 test roots.

A minimum teacher value gap of 0.1 yielded 3,994 sibling pairs: 3,019 train, 485 validation, and 490 test. The corpus SHA-256 was `bea320f3e6670f1a991ab1b8054538a8c26a2461582e094ce12249fc117152d7`.

### Selection correction

The first pairwise run incorrectly selected checkpoints by pointwise validation MAE. Its candidate reduced held-out weighted order accuracy from 59.16% to 57.99%.

The trainer now reports and selects by held-out pairwise MSE whenever pairwise ranking is active. Validation pairwise MSE was best after the first epoch and worsened on every later epoch. The selected epoch still regressed on the test pairs:

| Metric                  | Incumbent | Selected ranking candidate |
| ----------------------- | --------: | -------------------------: |
| Test pairwise MSE       |    0.2594 |                     0.2608 |
| Weighted order accuracy |    59.16% |                     58.45% |
| Gap correlation         |     0.268 |                      0.279 |

The small correlation increase did not compensate for worse magnitude error and order accuracy. The checkpoint was rejected before arena compute.

### Conclusion

Pointwise b18 value fitting improved held-out MAE but not broad search strength. Exact pairwise ranking also failed its held-out ordering gate. The original value head and 512-visit search remain the search incumbents at 36 wins. The sealed final arena was never opened.

## 2026-07-27 — Phase diagnostics, regret replay, and global pooling

### Scalar diagnostics

A diagnostic arena recorded only per-move scalar errors and outcomes, never board features or training examples. Relative to the 19-win greedy incumbent, the 22-win seed-averaged head improved policy KL in every phase. Its losses were concentrated in the middle game: 44 losing games had their largest value error there, versus 29 in the opening and five in the endgame. The first value error of at least 0.2 also occurred most often in the opening or middle game.

### Regret-directed reanalysis

Two disjoint middle-game corpora were generated from opening offsets 80,000 and 90,000. Teacher reanalysis ranked positions by disagreement between parent value and searched child value while retaining uniform coverage. The larger corpus contained 4,016 roots, split by whole game into 3,208 train, 400 validation, and 408 test rows, with SHA-256 `2641fc68965355143328145af6092801cb73d4575afbef213d67323891218b57`.

Head-only correction from the averaged policy did not pass the fixed autoresearch gate. The 256-game recipe scored 21 wins on its first registered seed. Adding a policy-preservation penalty scored 20. The 512-game recipe also scored 21. The predeclared 21.33-win incumbent mean was not weakened after observing these outcomes, so all three runs were rejected.

### Global pooling

A zero-initialized global-context adapter was added to the nested network. It pools channel-wise means and maxima into eight hidden channels and contributes residual policy and value outputs. The initialized model is exactly output-identical to the averaged head, has 105,561 parameters, and occupies 434,330 bytes in float form.

Adapter-only generic training scored 17 wins on its first seed. Targeted middle-game training scored nine in a manual development screen. Full low-rate co-adaptation with policy preservation scored 20 and was fast-rejected. Global pooling therefore did not improve this capacity regime.

## 2026-07-27 — Opponent-aware search pruning

Search can now optionally restrict opponent nodes to Moka's highest-policy replies while leaving every Moka decision branch available. This uses no KataGo query or hidden benchmark information during play and defaults to disabled.

On 20 paired fresh games from opening offset 38,000 at 256 visits, the unrestricted control scored 9–11 in 123.5 seconds. Top-two pruning scored 8–12 in 97.6 seconds. Top-four pruning tied the control at 9–11 in 107.4 seconds. Pruning reduced runtime but supplied no strength gain in the first screen.

At 512 visits on opening offset 39,000, the unrestricted control scored 8–12 in 287.9 seconds. Top-four pruning also scored 8–12 in 232.8 seconds. Top-eight pruning scored 9–11 in 226.4 seconds. The single-win top-eight edge requires independent confirmation and is not yet accepted.

On independent opening offset 40,000, the 512-visit unrestricted control again scored 8–12, in 279.8 seconds. Top-eight pruning scored 12–8, evenly split between Black and White, in 234.3 seconds. Across the two fresh 512-visit slices, top-eight pruning improved 17 wins to 21 while reducing runtime. It advanced to the full fixed-development arena; the sealed final remained unopened.

Doubling top-eight search to 1,024 visits on the same slice scored 11–9 in 481.4 seconds, one win below its 512-visit result. More visits did not improve the candidate and were rejected.

### Root-only policy hybrid

The averaged policy head improves greedy play but regresses when used throughout search. A root-only evaluator therefore supplied its priors only at each real move's root, while every descendant used the incumbent policy and value. The two checkpoints differ in 26,782 policy parameters; the trunk and value head are identical.

On opening offset 41,000, the unrestricted incumbent scored 8–12 and the root-only hybrid scored 9–11. Combining the root-only hybrid with top-eight opponent pruning scored 12–8. A same-opening top-eight-only control was required before attributing the gain to the extra head.

Top-eight pruning alone scored 14–6 on the same openings, two wins better than the combined hybrid. The root-only distilled prior was therefore rejected. Across offsets 39,000–41,000, top-eight pruning scored 35 of 60 versus 25 of 60 for unrestricted controls.

The distilled head was also screened only at opponent-to-move nodes, where it ranked the eight retained replies while the incumbent handled Moka nodes. On offset 42,000 this scored 7–13 versus 10–10 for the incumbent-only top-eight search. Both dual-head paths were removed after rejection.

### Full development confirmation

Top-eight pruning scored 47–53 on the fixed 100-game development arena at 512 visits, with 20 Black wins, 27 White wins, four move caps, and an 1,123.5-second runtime. This improves the unrestricted search incumbent by eleven wins, from 36 to 47, but remains below the majority threshold. The sealed final arena was not opened.

On opening offset 43,000, a local width screen confirmed eight retained replies. Widths six, eight, ten, and twelve scored 9, 13, 11, and 9 wins respectively. Width eight remained the only candidate.

With width eight fixed, exploration constants 1.0, 1.5, and 2.0 scored 9, 10, and 7 wins on opening offset 44,000. The original 1.5 remained best.

Global PUCT value weights 0.75, 1.0, and 1.25 scored 9, 8, and 8 wins on opening offset 45,000. The one-win edge at 0.75 was too small to advance.

Opponent-node value weights 0, 0.5, and 1.0 scored 8, 8, and 11 wins on opening offset 46,000. Treating KataGo as less adversarial worsened play, so the original value weight remained shared by both sides and the experimental option was removed.

### On-policy top-eight distillation

The next corpus uses a disjoint training opening range beginning at 100,000. Top-eight search determines the visited trajectory, while native b6 policy and value targets label every retained state. Whole-game IDs remain available for train, validation, and test splits. No fixed-development or sealed-final position is written into the corpus.

The 128-game, 256-visit collection produced 9,210 positions in a 1,353,359-byte archive. Whole-game splits contained 7,370 train, 864 validation, and 976 untouched test rows. Policies were finite and normalized, with zero mass on occupied points; values spanned the valid -1 to 1 interval. The corpus SHA-256 was `4972409c8aa0f03e57255951b2eeb4f4420ef8e37ce731c020e1a97aebc7bd5c`.

Three frozen-trunk policy-head seeds replayed the new training buckets eightfold alongside 100,000 strong-teacher and 50,000 opponent-trajectory rows. Validation move agreement was 55.3%, 55.4%, and 55.6%; untouched corpus-test agreement was 52.4%, 52.0%, and 52.2%. Non-policy tensors remained byte-identical. The three policy heads were averaged equally before arena evaluation.

At top-eight/512-visit search on opening offset 47,000, both the original checkpoint and the seed-averaged on-policy head scored 11–9. The candidate supplied no search gain and was rejected.

Uncertainty-adaptive widening retained at least eight replies and up to sixteen when cumulative opponent-policy mass was low. On opening offset 48,000, fixed top-eight scored nine wins; mass thresholds 0.75, 0.90, and 0.95 scored 8, 10, and 10. The one-win edges were insufficient to advance, and the adaptive controls were removed.

### Color-conditional value calibration

The global value-weight screen hid a color interaction: weight 0.75 improved Black by three wins while hurting White by two. Applying 0.75 only when Moka was Black and retaining 1.0 for White scored 12–8 on opening offset 49,000 versus 11–9 for the fixed top-eight control. Black improved from six to seven wins and White remained at five. Across offsets 45,000 and 49,000, Black-only 0.75 improved 9 of 20 Black wins to 13 of 20 without modifying White. It advanced to a second full fixed-development confirmation.

The full fixed-development confirmation remained 47–53. Black regressed from 20 to 19 wins, White changed from 27 to 28, and move caps increased from four to seven. Runtime was 1,048.6 seconds. Black-only weight 0.75 was rejected, and the sealed final arena remained unopened.

The opposite Black-only weight of 1.25 scored 8–12 on opening offset 50,000 versus 10–10 for the control, with Black falling from five to four wins. Color calibration was closed and its runtime control was removed.

### Value-head interactions with top-eight search

On opening offset 51,000, the incumbent value head scored 12–8 under top-eight search. Blends containing 25%, 50%, and 75% of the b18 root-value head scored 11, 10, and 9 wins; the full root-value head scored four. Opponent pruning did not rescue the previously rejected root-value fit.

On opening offset 52,000, the incumbent scored 11 wins. Searched-child value, exact sibling-ranking value, 8,192-game outcome value, and the earlier outcome value scored 8, 9, 8, and 9 wins. No alternative value target improved top-eight search.

On opening offset 53,000, the incumbent scored ten wins. Expert-iteration, second search fine-tuning, selective-search policy, second prefix DGRPO, and b18 elite candidates scored 8, 7, 10, 9, and 10 wins. No prior policy family displaced the incumbent under top-eight search.

The eight fixed dihedral orientations were screened on opening offset 54,000. Their wins were 10, 8, 8, 7, 5, 7, 4, and 10. No single orientation improved the canonical evaluator.

Eight-way symmetry averaging scored 6–4 on ten games from opening offset 55,000, versus 4–6 for the canonical control. Batching the eight transformed positions kept runtime close: 144.1 seconds versus 136.1 seconds. The ensemble advanced to an independent 20-game confirmation.

On independent opening offset 56,000, symmetry-averaged top-eight search scored 13–7 versus 9–11 for the canonical control. It won six games as Black and seven as White with no move caps, in 271.5 seconds. Across both fresh slices, the ensemble improved 13 of 30 control wins to 19 of 30 and advanced to the full fixed-development arena.

### Development majority

The frozen symmetry-averaged top-eight candidate scored 51–49 on the full fixed 100-game development arena, with 28 Black wins, 23 White wins, five move caps, and an 1,183.2-second runtime. This is the first verified development majority against KataGo b6c96. The exact checkpoint, 512-visit budget, width eight, and eight-symmetry averaging settings were frozen before opening the sealed final arena once at offset 1,000.

### Sealed final majority

The frozen candidate scored 52–48 on the sealed 100-game arena at opening offset 1,000, with 26 Black wins, 26 White wins, eight move caps, and a 1,349.0-second runtime. The checkpoint and all search settings were fixed before this arena was opened once. No final-arena position entered training, method selection, or hyperparameter selection.

This independent majority confirms that the development result generalized. Moka uses no KataGo query during play: its 104,129-parameter nested checkpoint supplies every policy and value evaluation, while KataGo b6c96 supplies only the opposing moves.

## 2026-07-27 — Website rollback and exact-artifact verification

The nested search evaluator was removed from the website after the raw browser policy produced visibly pathological edge play and non-finite value displays. Its reported 52–48 result required 512 visits, top-eight opponent pruning, and eight-way symmetry averaging; none of those mechanisms existed in the browser runtime. The claim therefore described the search system rather than the deployed player.

Production returned to `on-policy-v1`. Its browser weights have SHA-256 `0a2352de4302048c9bfe4679556358c27b5aeca317da2152b0412024d7244e58`, matching the previously deployed artifact byte for byte.

The original 100-game browser arena was rerun with the exact rolled-back INT8 artifact and KataGo b6c96. It reproduced 2–98, 23 move caps, Moka Brier score 0.259, and KataGo Brier score 0.348. Moka won games 71 and 88. The unquantized MLX checkpoint scored 1–99 on the same deterministic openings and won only game 71. The one-game difference is quantization-induced policy drift, so the website's 2-of-100 statement is accurate specifically for the browser artifact users receive.

Future promotion requires evaluation of the exact exported browser files. A candidate must produce finite policy and value outputs across the deployment test corpus, pass fixed human-play probes, and improve raw no-search play. Search-only strength cannot be presented as browser strength unless the same search system ships in the browser.

## 2026-07-27 — Browser-targeted policy repair

The exact nested INT8 browser artifact scored 16–84 on the original 100-game arena and 10–90 on a fresh 100-game arena at opening offset 120,000. This established the raw deployment baseline independently of the float checkpoint and the search system.

The seed-averaged selective policy head scored 22–78 in float form but only 10–90 after INT8 export. Keeping that policy head at INT16 increased the artifact to 138,616 bytes but recovered only 17–83. The averaged head was rejected as quantization-fragile.

Reproducing the strongest individual selective-head seed yielded 23–77 in float form. Its full INT8 export regressed to 12–88. Keeping only the policy weights at INT16 produced browser artifact `de07bf78e2b5c7c054b6b49c390b20b70b09bc7082586cf47a9b132357e2fc25`, measuring 138,616 bytes raw and 128,095 bytes with deterministic gzip.

The exact mixed-precision browser artifact scored 20–80 on the original arena and 14–86 at fresh opening offset 120,000. On the matched fresh set, this improves the previous browser artifact by four wins and one fewer move cap. All policy logits and values remained finite.

A fixed human-opening probe plays Black at intersections 20, 24, 56, and 60. Moka responded at 50, 48, 47, and 22, with zero first-line moves. This directly guards against the observed failure where White filled the top edge. The browser client now rejects any non-finite policy or value instead of allowing `NaN` into the interface.

The mixed-precision candidate remains in the research arena. Production remains on the verified stable model until broader human-opening probes and another disjoint arena confirm the gain.

## 2026-07-27 — On-policy browser correction

### Preserved policy head

Twenty thousand fresh positions from deterministic Moka-versus-KataGo trajectories at opening offset 150,000 were labeled by KataGo b6c96. A frozen-trunk policy-head correction used fourfold replay of this corpus and a 0.25 penalty against the starting policy.

The float candidate and its starting checkpoint scored 18 and 17 wins across 200 games from offsets 160,000 and 170,000. Move caps fell from 27 to 18. The exact mixed browser exports scored 22 and 21 wins, respectively, with 27 versus 29 caps. The one-win edge is marginal, but middle-game policy KL fell from 0.5747 to 0.5320 and endgame teacher-move agreement rose from 72.0% to 75.3%.

The accepted research artifact uses mixed INT8 trunk/value and INT16 policy tensors. Its weights are 138,616 bytes with SHA-256 `4a58dcfdb2febc826fcfc138c68ca2571716c6bfd4ed3a9690307efaee29a50c`. It passes the finite-output and fixed human-opening probes.

### Rejected corrections

A middle-game-weighted version lost one additional game on opening offset 180,000. Three sparse b18 middle-game corrections also failed their screen; the best full candidate scored 9–91 on offset 200,000 versus 11–89 for the preserved head. A wider eight-channel policy head, seed averaging, sibling seeds, and policy-head interpolation supplied no reproducible gain. These candidates were rejected and were not copied into the live arena.

## 2026-07-27 — Low-budget browser search

### Strength screen

Sixteen-visit PUCT search was compared with raw policy play on five disjoint opening ranges:

| Opening offset | Games | Raw wins | 16-visit wins |
| -------------: | ----: | -------: | ------------: |
|        220,000 |    20 |        1 |             4 |
|        230,000 |    20 |        2 |             4 |
|        240,000 |    20 |        1 |             2 |
|        250,000 |    20 |        0 |             3 |
|        260,000 |    40 |        5 |             3 |
|      **Total** |   120 |    **9** |        **16** |

Search improved aggregate wins but regressed on the final slice. It is therefore exposed as an optional arena mode, not presented as a stronger base checkpoint. Raw Moka remains the default.

An exploration sweep on offset 250,000 scored 3, 4, 3, and 2 wins at PUCT constants 0.75, 1.0, 1.5, and 2.0. On offset 260,000, 1.0 scored four wins and 1.5 scored three. The browser uses 1.0.

### Opponent-node pruning

Restricting only simulated opponent nodes to Moka's four highest-prior replies preserved every result across 80 games from offsets 270,000 and 280,000: both unrestricted and top-four search scored 10–70 with eight move caps. Python runtime fell from 40.0 to 33.8 seconds, a 15.5% reduction. The browser implementation uses the same top-four rule.

The arena now records the visit budget in run history. The exact browser artifact completed a four-game, 16-visit smoke run without illegal moves or non-finite outputs. Search remains deliberately opt-in because the JavaScript runtime cost is large relative to raw play.

## 2026-07-27 — Search-policy compression attempt

A fresh search-distillation corpus used 128 games from opening offset 300,000. Sixteen-visit Moka trajectories and KataGo replies produced 9,332 position-policy-value rows in a 984,426-byte archive.

Three frozen-trunk policy-head seeds were trained with the existing 20,000-position on-policy corpus as preservation replay. Their first 20-game screen scored 2, 2, and 3 wins versus two for the incumbent. Only seed 3 advanced. On a fresh 100-game comparison at offset 320,000, seed 3 scored 14–86 while the incumbent scored 16–84, with thirteen move caps each.

Search-policy compression did not transfer the runtime gain into the raw network and was rejected. The research arena continues to serve the preserved `4a58dcfd` artifact.

## 2026-07-27 — Website promotion

The exact preserved browser artifact was promoted to the main site under fingerprinted `4a58dcfd` URLs. Its deterministic gzip is 128,100 bytes and decompresses to weights with SHA-256 `4a58dcfdb2febc826fcfc138c68ca2571716c6bfd4ed3a9690307efaee29a50c`.

The deferred site Worker was rebuilt with mixed INT8/INT16 decoding. The landing page retains its visibility, delay, and idle-callback gates, so neither the model nor its runtime enters the initial critical path. A production Next.js build, finite-output smoke test, human-opening probe, landing-page self-play check, and interactive Moka-page check all passed.

## 2026-07-27 — Large browser-policy continuation

### Fresh DAgger corpus

The preserved browser checkpoint generated 100,000 deterministic Moka-versus-KataGo positions from opening offset 330,000. KataGo b6c96 labeled every reached state. Moka matched the teacher move on 54.2% of positions; the mean Goldilocks sample weight was 1.295. The compressed corpus occupies 16,681,298 bytes.

Three frozen-trunk policy-head continuations and four low-rate full-network continuations used the new corpus with the earlier 20,000-position on-policy set as replay. On a fresh 20-game screen at offset 440,000, the incumbent scored three wins. The head-only candidates scored three, two, and two; the full-network candidates scored two, two, two, and three. All were rejected.

### Symmetry and low-budget search

Eight-way symmetry averaging and every individual board orientation were neutral over 40 games from offset 450,000. On 100 games from offset 460,000, raw policy and two visits each scored eleven wins, while four and eight visits scored six and eight. Sixteen visits scored three wins over the first 40 games. None advanced.

### Strong-teacher replay

Three frozen-trunk heads used 50,000 native b18 labels from greedy Moka trajectories plus 20,000 b6 on-policy positions. The only first-stage candidate scored five wins versus four over 40 games, then regressed to seven versus eight on a disjoint 100-game confirmation.

A broader replay balance combined the same b18 trajectories with the new 100,000-position b6 corpus. Its only first-stage candidate scored six wins versus five, then regressed to twelve versus sixteen over 100 new games. Policy-head interpolation was neutral. Both strong-teacher branches were rejected.

### Hard-target correction

The original compact Moka recipe used a 50% hard teacher-move target, while the first large-corpus continuations used only the soft distribution. Restoring the 50% hard target improved all three frozen-trunk seeds on an initial 40-game screen. Their equal-weight policy-head average then scored 13–87 on offset 540,000 versus 7–93 for the incumbent.

The exact mixed INT8/INT16 browser artifact had SHA-256 `e8fc6d525e79015a9a7eb56d1fbef7cee9723a61abed79e5ff8f4ba632f96c4a`, remained 138,616 bytes, passed finite-output and human-opening probes, and scored:

| Browser offset |  Candidate |  Incumbent |
| -------------: | ---------: | ---------: |
|        550,000 |      12–88 |      10–90 |
|        560,000 |      11–89 |      11–89 |
|      **Total** | **23–177** | **21–179** |

The candidate added four move caps across the same 200 games. A quantization-aware continuation improved three seeds on its first 40-game proxy but tied or regressed on a 100-game confirmation.

Hard-target mixtures of 25%, 50%, 75%, and 100% were then compared through seed-averaged heads. On offset 590,000, the incumbent scored eight of 40 while the mixtures scored five, two, two, and two. The apparent hard-50 gain was therefore opening-slice dependent and did not generalize.

### Conclusion

The 100,000-position corpus is useful, and hard targets can shift game outcomes substantially, but no candidate cleared the multi-slice exact-browser gate. Production and the research arena remain on browser artifact `4a58dcfd`. Further work should change the sequential objective or representation rather than repeat frozen-head imitation on the same teacher labels.

## 2026-07-27 — Group-relative outcome training

### Outcome corpora

The outcome generator gained a greedy-opponent mode so KataGo always selected its highest-policy legal reply while Moka sampled alternatives within each shared opening group. The first 1,024-game corpus contained 80,582 positions and 40,224 Moka decisions, but only 13 wins. Its SHA-256 was `8c0b99c9dbb487ebd05a32c90f35ce21d5e8f124f77e2a15d020a3ce3ac663f0`.

Fixed Moka sampling temperatures of 0.25, 0.4, and 0.6 produced 12, 16, and 3 wins over 128-game screens. Temperature 0.4 advanced to a second 1,024-game corpus with 60,762 positions, 30,215 Moka decisions, and 85 wins split 44 as Black and 41 as White. Every eight-game opening group varied in outcome. The corpus SHA-256 was `480680eb187faa5519b7ce4bf942b3edab6e4a15ac5793d44de7d5c6df91c0fa`.

### Rejected objectives

Pure group-relative policy optimization, distilled variants, success-routed variants, and multiple seeds all produced initial slice gains that disappeared on independent openings. The strict success-routed candidate scored 5–35 versus 4–36 on its first screen, then every seed scored 1–39 versus 2–38 on the next slice. None reproduced, so no outcome-trained weights were exported or promoted.

The negative result suggests that 85 wins are still too sparse for stable group-relative optimization at this capacity. Future outcome work needs substantially more successful trajectories or a dense teacher-derived advantage target rather than stronger weighting of the same sparse returns.

## 2026-07-27 — Deployment-matched deeper browser search

### Budget and batching screen

Low-budget PUCT was retested with the exact mixed INT8/INT16 browser artifact. A packed Worker protocol now sends each multi-position evaluation wave in one message and validates every returned policy and value.

On a fresh exact-browser 20-game slice at offset 690,000, raw play scored 2–18. Eight, 16, and 64 visits scored 1–19, 6–14, and 6–14. The 16-visit run completed in 52 seconds versus 192 seconds for 64 visits and produced one rather than four move caps. On offset 700,000, raw, 16, and 32 visits scored 7–33, 8–32, and 9–31; 32 visits produced nine move caps. Sixteen visits remained the practical budget.

The original search evaluated eight reserved leaves per root wave. Reducing this internal batch to one makes the same 16 model evaluations build a deeper tree. Opponent pruning was also removed: at 16 visits, full legal branching and widths four, eight, and sixteen produced identical decisions on the measured 100-game slice.

| Python opening offset | Games | Eight-leaf waves | One-leaf waves |
| --------------------: | ----: | ---------------: | -------------: |
|               720,000 |   100 |               17 |             28 |
|               730,000 |   100 |                7 |             19 |
|             **Total** |   200 |           **24** |         **47** |

None of the 47 Moka wins was awarded at the move cap. The one-leaf runs produced 24 and 23 caps respectively, but every capped game was a Moka loss. The strength gain is therefore not an unfinished-board scoring artifact.

The exact browser implementation reproduced the deeper tree at offset 740,000: 6–14 versus 2–18 for raw play. Local interactive responses after model initialization took 102 ms and 86 ms. Search is dynamically imported only when a person starts a game; deferred model loading and raw sampled self-play remain unchanged.

### Termination diagnosis

On 40 games from offset 750,000, the 16-visit control scored 9–31 with eleven caps. A 0.05 late-area blend scored eight wins with ten caps, a 0.15 blend scored seven with eight caps, and a one-ply rollout scored seven with eleven caps while increasing runtime from about 40 to 69 seconds. Each reduced or preserved both strength and caps, so all were rejected.

The capped games contained 1,276 unique side-to-move positions and zero repeated positions. Across the full run, Moka passed 265 times and KataGo passed 96 times. Raising the earliest allowed pass from move 20 to 24 produced identical play; raising it to 40 lost one win without changing caps, and raising it to 60 added a cap. The failure is therefore neither a ko loop nor simple refusal to pass. Moka passes, KataGo finds another profitable move, and the losing game continues. No cosmetic termination rule was promoted.

### Visit-count confirmation

With one-leaf search waves fixed, visit budgets 16, 20, 24, and 32 scored 9, 11, 10, and 6 wins over the same 40 games. Their cap counts were 11, 10, 14, and 5. Twenty visits advanced to a preselected independent comparison at offset 760,000.

On the independent 100-game block, 16 visits scored 16–84 with 22 caps. Twenty visits scored 15–85 with 25 caps and took 119.6 seconds versus 98.0 seconds. The apparent 20-visit gain did not reproduce, so the browser remains at 16.

### Cross-turn tree reuse

The Python arena keeps the selected subtree after Moka's move and aligns it to the opponent's reply when that reply was already expanded. The first browser deployment rebuilt an empty tree every turn. The site now retains the same bounded subtree across turns and resets it whenever the move history no longer matches, including a new game or an undo into an unretained branch.

This adds no model bytes or evaluations. A clean production build completed two searched replies and a new-game reset without stale-state or legality errors. The live second response took 120 ms. The deployed implementation now matches the state-reuse behavior used by the Python strength measurements.

### Root-only symmetry ensemble

Moka's eight dihedral orientations were averaged only at each real move's root. Descendant evaluations remained canonical, the model stayed fixed, and no KataGo output was available to the search. Batched MLX evaluation kept runtime effectively unchanged.

| Opening offset | Games | Canonical root | Symmetry root |
| -------------: | ----: | -------------: | ------------: |
|        770,000 |    40 |              5 |            10 |
|        780,000 |   100 |             17 |            26 |
|      **Total** |   140 |         **22** |        **36** |

On the 100-game confirmation, caps fell from 35 to 18. None of the symmetry candidate's wins came from a capped game. The improvement reproduced across colors: the candidate scored five Black and five White wins on the screen, then eleven Black and fifteen White wins on confirmation.

The exact mixed-precision browser implementation matched the float search on all four fixed opening probes: moves 20, 24, 56, and 60 produced replies 50, 48, 32, and 30 in both runtimes. Production-mode replies took 166–179 ms after startup. Lighthouse remained 96, and the symmetry/search chunks were absent from its initial request set.

Root-only symmetry was promoted in commit `05707ac`. Live production reproduced C7→F4 and G3→D6; the cold first reply took 335 ms and the warm reply took 167 ms.

### Symmetry-search visit count

The root-symmetry candidate changed the useful depth range. On offset 770,000, 20 visits scored 12–28 with six caps versus 10–30 with eleven caps at 16 visits. A frozen independent comparison on 100 games from offset 790,000 scored:

| Root-symmetry budget | Wins | Caps | Runtime |
| -------------------: | ---: | ---: | ------: |
|            16 visits |   22 |   31 | 138.7 s |
|            20 visits |   27 |   23 | 160.3 s |

All wins completed normally. The 20-visit candidate improved both colors, from 8 Black and 14 White wins to 11 Black and 16 White wins.

The exact mixed-precision browser still matched all four fixed probes at 20 visits. Production-mode replies took 195–200 ms after initialization. Lighthouse behavior was unchanged because the search remains interaction-gated. The 20-visit budget was promoted in commit `4a638bb`; the live cold probe replied C7→F4 in 429 ms without runtime errors.

### Descendant symmetry scheduling

Cycling a single descendant orientation by absolute move depth was tested at the same 20-visit budget. This added no model evaluations or bytes: the root still averaged all eight orientations, while each descendant used one orientation selected from its move count.

On 40 games from opening offset 800,000, canonical descendants scored 13–27 with nine caps. Depth-cycled descendants scored 12–28 with eight caps. Neither configuration received a win at the move cap. The depth schedule was rejected because it lost one completed game and merely reduced caps by one.

Round-robin scheduling then assigned each newly evaluated descendant the next orientation, distributing views across sibling branches while preserving one evaluation per leaf. It scored 10–30 with nine caps on the same block. This also regressed from the 13-win control and was rejected.

Paired descendant inference averaged canonical features with one complementary orientation, doubling descendant inference without changing the model. On 20 games from opening offset 810,000, the canonical control scored six wins with four caps. Canonical plus 180° rotation scored four wins with six caps, canonical plus reflection scored five with four caps, and canonical plus 90° rotation scored four with three caps. All three lost completed games and were rejected.

### Root-symmetry search at 28 visits

Root symmetry changed the useful visit range, so budgets above 20 were screened separately from the earlier canonical-root sweep. On 20 games from opening offset 820,000, budgets 20, 24, 28, and 32 scored seven, seven, nine, and eight wins. Their cap counts were seven, four, three, and three. Twenty-eight visits was frozen before a fresh confirmation.

On 100 games from opening offset 830,000, 20 visits scored 22–78 with 13 Black wins, nine White wins, and 21 caps. Twenty-eight visits scored 28–72 with 14 Black wins, 14 White wins, and 15 caps. The control received no cap-awarded wins; the candidate received one, leaving a conservative completed-game comparison of 27 versus 22. Runtime increased from 164.3 to 209.1 seconds.

The exact mixed-precision browser path matched Python on the four fixed human probes. Human moves 20, 24, 56, and 60 produced Moka replies 50, 48, 47, and 22 in both runtimes. Production-mode browser replies took 265 ms cold and 252–253 ms warm, with no console errors. The model, Worker, initial request set, and deferred loading gates were unchanged.

The 28-visit budget was promoted in commit `3f8c4f3`. Live production reproduced C7→F4 and G7→D4; the first reply took 720 ms after cold dynamic imports, and the warm reply took 254 ms without console errors.

### Adaptive search ceiling

Starting at 20 visits and extending close roots to 36 visits at top-two visit-margin thresholds 0.1, 0.2, and 0.3 tied the fixed 28-visit control at five wins on 20 games from opening offset 840,000. A 48-visit ceiling at threshold 0.1 scored six wins with zero caps versus five wins and two caps for fixed 28; thresholds 0.2 and 0.3 scored five and six.

The 20→48, 0.1-margin candidate was frozen for a 100-game comparison at opening offset 850,000. Fixed 28 scored 30–70 with 19 Black wins, 11 White wins, 18 caps, and no cap-awarded wins. Adaptive search scored 27–73 with 18 Black wins, nine White wins, 11 caps, and no cap-awarded wins. Runtime fell from 160.6 to 139.2 seconds, but the candidate lost three completed games and was rejected.

### Sequential halving

The sequential-halving session had an unexercised constructor error and did not accept the root-symmetry evaluator. The research path was repaired so it refreshes averaged root priors and charges that evaluation against the fixed budget.

On 20 games from opening offset 860,000, plain 28-visit PUCT scored six wins with two caps. Sequential halving with four, eight, and 12 root candidates scored three, three, and two wins, with one, two, and four caps. All configurations regressed materially and were rejected.

### Root value tie-breaking

Final move selection was allowed to prefer the better searched value among children tied for the most visits or trailing by one or two visits. On 20 games from opening offset 870,000, visit slacks zero, one, and two reproduced every control decision exactly: all scored five wins with four caps. The rule was behaviorally inert on the measured roots and was not promoted.

### Root tactical priors

A root-only rule prior rewarded immediate captures and penalized non-capturing self-atari using only the visible board and legal move results. Capture bonuses 0.25 and 1.0, self-atari penalties 0.5 and 2.0, and both paired settings reproduced every control decision on 20 games from opening offset 880,000. All variants scored one win with one cap. The adjustment was inert even at aggressive weights and was rejected.

### Root-symmetry value weight

The policy–value balance was rescreened because earlier value-weight tests used a different search topology. On 20 games from opening offset 890,000, weights 0.5, 0.75, 1.0, 1.25, 1.5, and 2.0 scored four, two, six, six, five, and one win. Cap counts were two, four, two, three, seven, and 11. The deployed weight 1.0 retained the joint win-and-cap lead; every alternative was rejected.

### Cycle conclusion

The only reproducible strength gain in this cycle was root-symmetry search at 28 visits, promoted in commit `3f8c4f3`. Depth-cycled symmetry, round-robin symmetry, paired descendant inference, adaptive ceilings, sequential halving, root value tie-breaking, tactical priors, and alternate value weights either tied, regressed, or failed fresh confirmation.

The next model-side experiment should train a dense action-ranking or child-value target from search comparisons on Moka-reached states. Repeating policy-head imitation or sparse outcome weighting is unlikely to help: those objectives already failed across independent openings, while the accepted gain came from resolving local action ranking more accurately at inference time.

## 2026-07-28 — Deployment-matched search distillation

### Dense 28-visit corpus

The collector was aligned to the deployed player: one-leaf search waves, 28 visits, full legal branching, subtree reuse, and eight-symmetry averaging at every real root. Moka search supplied its own offline trajectory and visit target; KataGo b6c96 supplied opponent moves and value labels during data generation only.

Each Moka target mixed 75% search visits with 25% legal averaged root policy and received 4× sample weight. This retained dense support rather than collapsing targets to one-hot choices. The 128-game corpus from opening offset 900,000 contained 9,659 positions and 4,821 weighted Moka roots. Whole-game splits contained 7,589 training, 1,043 validation, and 1,027 test positions. Search targets averaged 44.2 nonzero moves and 0.387 top-move probability. The corpus SHA-256 was `b79fa044c60bc04f64f34dd61029e72faba8d8af252608ac6601658e9c240508`.

### Soft policy-head continuation

Three frozen-trunk policy-head seeds used the earlier 20,000-position browser on-policy corpus as preservation replay and a 0.25 logit-preservation penalty. All improved dense-target test loss from the incumbent's 2.7897 to 2.7749–2.7769, but reduced top-move agreement from 62.51% to 61.6–62.3%.

On a 20-game search screen at opening offset 920,000, the incumbent and seeds 41, 42, and 43 scored four, four, five, and four wins. Seed 42 alone advanced. On 100 fresh games from offset 930,000, the incumbent scored 34–66 with 20 Black wins, 14 White wins, 18 caps, and no cap-awarded wins. Seed 42 scored 29–71 with 17 Black wins, 12 White wins, 14 caps, and no cap-awarded wins. The soft candidate was rejected.

### Hard search-move mixture

A predeclared 25% hard-search-move mixture trained three more frozen-trunk seeds on the same corpus. Validation agreement rose to 63.5–64.0%, but untouched test agreement fell to 61.5–61.6%. On a 20-game search screen at opening offset 940,000, the incumbent and seeds 44, 45, and 46 scored five, four, five, and five wins. Every candidate added a cap or lost a completed game, so none advanced.

### Moka-turn-only search distillation

The next corpus removed opponent decision points entirely and retained only Moka roots. It contained 9,702 decisions from 256 games beginning at opening offset 1,000,000, split by whole game into 7,715 training, 983 validation, and 1,004 test positions. Targets again mixed 75% of the 28-visit distribution with 25% of the averaged root policy, but used 2× rather than 4× sample weight to balance the earlier on-policy replay. Policies averaged 44.5 nonzero moves and 0.396 top-move probability. The corpus SHA-256 was `2b47622971a3567adeec97bf1da10fda20f707c7a5c57d1c1e2d5ebc81cb505b`.

Three frozen-trunk heads reduced untouched test loss from 2.5958 to 2.5840–2.5848. Seed 47 also raised test top-move agreement from 78.09% to 78.39%; seeds 48 and 49 scored 77.89% and 78.29%.

On a predeclared 20-game screen at opening offset 1,100,000, the incumbent and seeds 47, 48, and 49 scored six, eight, six, and nine wins. Cap counts were five, five, four, and five. Seed 49 was frozen as the only confirmation candidate.

On 100 fresh games from opening offset 1,110,000, both the incumbent and seed 49 scored 34–66. The incumbent split 13 Black and 21 White wins with 18 caps and no cap-awarded wins. Seed 49 split 14 Black and 20 White wins with 31 caps and one cap-awarded win. Conservative completed wins therefore regressed from 34 to 33. The focused distillation candidate was rejected.

### Late-state balancing

Only 338 of the 9,702 Moka-turn samples occurred after the 50th recorded Moka decision. A predeclared phase correction increased those rows to 10× their original relative weight while leaving the architecture, search targets, replay corpus, and preservation penalty unchanged. After normalization, early and middle rows had weight 0.761 and late rows had weight 7.613. The reweighted corpus SHA-256 was `c934c80059192e166f2f5cb30fdac8b2b5076dcd34e81bbffb2f48aed6206c05`.

Three frozen-trunk heads reduced late-state test loss from 2.2655 to 2.2600–2.2611 while preserving the incumbent's top move on all 39 late test rows. Seed 52 had the best full-test agreement at 78.59%.

On a fresh 20-game screen at opening offset 1,120,000, the incumbent and seeds 50, 51, and 52 scored five, five, six, and six wins. Their cap counts were one, four, three, and two. Seed 52 was frozen because it had the best joint screen and held-out result.

On 100 new games from opening offset 1,130,000, the incumbent scored 26–74 with 12 Black wins, 14 White wins, 12 caps, and no cap-awarded wins. Seed 52 scored 23–77 with 11 Black wins, 12 White wins, 11 caps, and no cap-awarded wins. The candidate lost three completed games and was rejected.

Search distillation improved dense-target loss consistently, but neither Moka-turn filtering nor late-state balancing produced a reproducible game-strength gain. No distilled checkpoint was exported or promoted.

## 2026-07-28 — Low-budget PUCT exploration

### Root-symmetry screen

PUCT exploration had previously been tuned only under the old 256-visit search. It was rescreened with the deployed evaluator, one-leaf expansion, root symmetry ensemble, and 28 visits.

On 20 games from opening offset 1,140,000, exploration coefficients 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, and 3.0 scored four, five, three, three, five, seven, three, and seven wins. Their cap counts were 12, nine, ten, three, two, two, one, and two. Coefficient 2.0 was frozen because it matched the best win-and-cap result with the smallest change from production's 1.5.

### Independent confirmation

Two disjoint 100-game blocks compared the frozen 2.0 candidate with production:

| Opening offset | Exploration 1.5 |   Caps | Exploration 2.0 |   Caps |
| -------------: | --------------: | -----: | --------------: | -----: |
|      1,150,000 |              25 |     20 |              26 |     16 |
|      1,160,000 |              28 |     15 |              28 |     12 |
|      **Total** |          **53** | **35** |          **54** | **28** |

Every win completed normally. Exploration 2.0 added one completed win and removed seven caps over 200 fresh games without changing the model, visit count, inference count, or browser payload.

## 2026-07-28 — Parent-value first-play urgency

### Harness correction

The ordinary arena session parsed root-selection and tactical-prior options but did not pass them into `MokaSearchSession`. Their earlier bit-for-bit "inert" results were therefore invalid measurements rather than evidence that the methods had no effect. The session construction now uses named arguments and passes every root option. The sequential-halving constructor received the same correction.

### First-play urgency

The deployed tree assigned every unvisited child a value of zero. A parent-value first-play urgency candidate instead initializes an unvisited move from the current node's mean value minus a fixed reduction. This uses only Moka's own evaluation and changes neither the model nor the visit budget.

On 20 games from opening offset 1,170,000, the absolute-zero control and reductions 0, 0.1, 0.25, and 0.5 scored four, six, seven, seven, and five wins. Their cap counts were two, four, two, one, and three. Reduction 0.25 was frozen because it matched the best win count with the fewest caps.

### Independent confirmation

Two disjoint 100-game blocks compared the frozen 0.25 candidate with the exploration-2.0 production search:

| Opening offset | Absolute zero |   Caps | FPU 0.25 |   Caps |
| -------------: | ------------: | -----: | -------: | -----: |
|      1,180,000 |            25 |     15 |       27 |      6 |
|      1,190,000 |            27 |     10 |       35 |      6 |
|      **Total** |        **52** | **25** |   **62** | **12** |

Every win completed normally. Parent-value FPU added ten completed wins and removed 13 caps over 200 fresh games. It requires no additional model evaluations, weights, payload bytes, or KataGo information at runtime.

### Corrected root value selection

With the ordinary session wiring repaired, value-aware final move selection was rerun under exploration 2.0, FPU 0.25, root symmetry, and 28 visits. On 20 games from opening offset 1,200,000, ordinary maximum-visit selection scored eight wins with one cap. Choosing the best searched value among exact visit ties or children trailing by one or two visits scored seven wins with three, five, and two caps respectively.

Unlike the earlier invalid inert measurement, the corrected options changed play. Every setting lost one completed game, so value-aware final selection was rejected.

### Corrected tactical root priors

The repaired ordinary session also made the root capture and self-atari adjustments effective. On 20 games from opening offset 1,210,000, production scored nine wins with zero caps. Capture bonuses 0.25 and 0.5 each scored eight wins with zero caps. Self-atari penalties 0.5 and 1.0 scored eight and nine wins, also with zero caps.

No tactical prior improved the joint result. The rules-based root adjustments were rejected.

### Prior-mass first-play urgency

A standard FPU variant scaled the reduction by the square root of the prior mass already visited at the node. On 20 games from opening offset 1,220,000, fixed FPU 0.25 scored seven wins with zero caps. Prior-mass reductions 0.25, 0.5, and 0.75 scored six, six, and eight wins, also with zero caps. Reduction 0.75 was frozen as the only screen winner.

On 100 fresh games from opening offset 1,230,000, fixed FPU scored 29–71 with 14 Black wins, 15 White wins, seven caps, and no cap-awarded wins. Prior-mass FPU scored 25–75 with 11 Black wins, 14 White wins, ten caps, and no cap-awarded wins. It lost four completed games and was rejected.

### Root-policy temperature

The symmetry-averaged root policy was sharpened or flattened before allocating the same 28 visits. On 20 games from opening offset 1,240,000, temperatures 0.5, 0.75, 1.0, 1.25, and 1.5 scored seven, six, five, five, and five wins. Their cap counts were zero, two, zero, one, and two. Temperature 0.5 was frozen as the only clear joint screen winner.

On 100 fresh games from opening offset 1,250,000, production scored 36–64 with 17 Black wins, 19 White wins, seven caps, and no cap-awarded wins. Temperature 0.5 scored 26–74 with 13 Black wins, 13 White wins, four caps, and no cap-awarded wins. Sharpening lost ten completed games and was rejected.

### Post-FPU exploration retuning

Because FPU changed the policy–value allocation, exploration was retuned under fixed FPU 0.25. On 20 games from opening offset 1,260,000, coefficients 1.5, 1.75, 2.0, 2.25, and 2.5 scored six, seven, six, seven, and five wins. Their cap counts were one, two, one, one, and zero. Coefficient 2.25 was frozen as the best joint screen result.

On the first 100-game confirmation block at opening offset 1,270,000, production 2.0 scored 34–66 with five caps, while 2.25 scored 35–65 with seven caps. A second predeclared block at offset 1,280,000 scored 36–64 with seven caps for production and 35–65 with eight caps for the candidate.

Across 200 games, both settings won 70 completed games. Candidate caps increased from 12 to 15, so exploration 2.25 was rejected and production remained at 2.0.

### Opponent branch width under FPU

Opponent reply pruning was rescreened because FPU changed low-budget allocation. On 20 games from opening offset 1,290,000, full branching and widths two, four, eight, and 16 scored nine, eight, nine, nine, and eight wins. Every setting produced one cap. Width eight also produced 41 repeated positions inside its capped game, while the full-branching cap had none.

No pruned width improved completed wins, and one introduced repetition-heavy behavior. Full opponent branching was retained.

### Visit count under FPU

The visit budget was rescreened because FPU changed low-budget allocation. On 20 games from opening offset 1,300,000, budgets 20, 24, 28, 32, 36, and 40 scored seven, six, six, three, five, and ten wins. Their cap counts were three, one, four, four, three, and one. Forty visits was frozen as the clear strength-and-cap screen winner.

On the first 100-game confirmation block at opening offset 1,310,000, 28 visits scored 30–70 with three caps in 183.8 seconds. Forty visits scored 32–68 with eight caps in 226.1 seconds.

On a second predeclared block at offset 1,320,000, 28 visits scored 38–62 with seven caps in 165.1 seconds. Forty visits scored 42–58 with five caps in 268.4 seconds.

Across 200 games, 40 visits added six completed wins, 74 versus 68, but increased caps from ten to 13 and runtime from 348.9 to 494.5 seconds. It failed the predeclared no-cap-regression gate and was not promoted as a fixed budget.

### Adaptive 28-to-40 visits

Adaptive search started at 28 visits and extended only roots whose top-two visit margin remained below a threshold. On 20 games from opening offset 1,330,000, production scored seven wins with three caps, while fixed 40 visits scored nine with one cap. Adaptive thresholds 0.1, 0.2, and 0.3 scored five, eight, and eight wins, each with one cap. Threshold 0.2 was frozen because it matched the best adaptive result with lower runtime.

On the first 100-game confirmation block at opening offset 1,340,000, production scored 30–70 with nine caps in 134.4 seconds. Adaptive search scored 33–67 with four caps in 149.3 seconds.

On a second predeclared block at offset 1,350,000, production scored 30–70 with eight caps in 128.1 seconds. Adaptive search regressed to 25–75 with six caps in 151.8 seconds.

Across 200 games, adaptive search lost two completed games, 58 versus 60, while reducing caps from 17 to ten and increasing runtime from 262.5 to 301.1 seconds. It was rejected.

### Root-only first-play urgency

Fixed FPU 0.25 was restricted to the real-move root while descendant nodes returned to the earlier absolute-zero treatment. This kept the model, root symmetry, 28-visit budget, exploration coefficient, and number of evaluations unchanged.

On 20 fresh games from opening offset 1,360,000, the all-node FPU control scored six wins with three caps. Root-only FPU scored five wins with zero caps. Neither configuration received a cap-awarded win.

The candidate reduced unfinished games but lost a completed game. It failed the strength gate and was rejected. The option remains available in the research arena for reproducibility, but the browser retains all-node FPU.

### Root policy branch cap

Ordinary PUCT was restricted at each real-move root to the highest-prior moves from Moka's symmetry-averaged policy. Descendant branching, FPU 0.25, exploration 2.0, and the 28-visit evaluation budget were unchanged.

On 20 fresh games from opening offset 1,370,000, the unrestricted control scored six wins with three caps. Root caps of eight, 12, 16, 24, and 32 reproduced the control exactly, including colors, passes, caps, and capped-position counts.

More aggressive caps of two, four, and six scored four, four, and three wins. Their cap counts were zero, four, and three. None received a cap-awarded win.

The measured search already concentrated its visits within the top eight root priors, making wider caps behaviorally inert. Narrower caps removed useful alternatives and lost completed games. Explicit root branch pruning was rejected.

## 2026-07-28 — Search-specific list-wise continuation

### Protocol

The earlier generic list-wise experiment ranked KataGo's preferred moves on broad teacher positions. The earlier search-distillation experiment fit marginal cross-entropy on Moka's own 28-visit targets. This experiment combined the missing pieces: a Plackett–Luce top-eight ranking loss on the 9,702 Moka-turn search corpus from opening offset 1,000,000.

The nested trunk and value head remained frozen. Three policy-head seeds used two epochs at learning rate 0.00001, list-wise weight 0.1, the existing 20,000-position browser on-policy corpus as preservation replay, and a 0.25 logit-preservation penalty.

Seeds 53, 54, and 55 reached 78.2%, 78.2%, and 77.9% top-move agreement on the untouched test games. The starting checkpoint was 78.1%, so the static changes were within noise.

### Fresh arena screen

On 20 fresh games from opening offset 1,380,000, the incumbent scored eight wins with one cap. Seeds 53, 54, and 55 scored seven, six, and eight wins with three, five, and two caps. No configuration received a cap-awarded win.

The only win-rate tie added a cap; the other seeds lost completed games. Search-specific list-wise continuation was rejected without confirmation.

## 2026-07-28 — Parameter-free leaky activation

### Protocol

ReLU was replaced with leaky ReLU at slope 0.1 throughout the nested trunk and heads. The candidate retained exactly 104,129 parameters and identical tensor shapes, so it would not have added model weights or payload bytes.

Loading the incumbent weights under the new activation changed outputs substantially. Three two-epoch calibration seeds therefore trained the full network at learning rate 0.00001 on the 100,000-position browser DAgger corpus, with the 20,000-position on-policy preservation replay, 50% hard policy targets, and a 0.25 penalty against the incumbent logits.

The short runs reached only 44.1%–44.4% top-move agreement on untouched test games, versus 55.2% for the incumbent. A single predeclared 12-epoch recovery run improved steadily to 54.3% agreement. Its value MAE improved from the incumbent's 0.2892 to 0.2627, but policy agreement remained below the required recovery baseline.

The activation candidate failed the offline gate and was never evaluated on arena openings. Its implementation was removed rather than retaining an unpromoted runtime branch.

## 2026-07-28 — Deployment topology correction

### Harness mismatch

The browser expands and evaluates one leaf after every PUCT selection. The Python research constant had drifted back to batches of eight reserved leaves even though the earlier experiment log identified one-leaf search as the stronger topology. Fixed human probes still matched, but those four decisions were insufficient evidence that broad arena outcomes matched.

The Python default was restored to one leaf per wave before generating more training data.

### Matched comparison

On 20 fresh games from opening offset 1,390,000, deployment-exact one-leaf search scored seven wins with one cap. The stale eight-leaf topology scored five wins with four caps on the identical openings.

The result reproduced on 100 fresh games from offset 1,400,000:

| Topology       | Wins | Black | White | Caps | Cap wins |
| -------------- | ---: | ----: | ----: | ---: | -------: |
| One-leaf waves |   31 |    13 |    18 |    5 |        0 |
| Eight-leaf     |   20 |     8 |    12 |    9 |        0 |

A second disjoint one-leaf block at offset 1,410,000 scored 30 wins, split 16 Black and 14 White, with three caps and no cap-awarded wins. Across the two deployment-topology blocks, Moka won 61 of 200 completed games with eight caps.

This corrects the research harness rather than changing production: the browser already used the stronger one-leaf implementation.

## 2026-07-28 — Root action-value policy targets

### Corpus

The corrected one-leaf search exposed root-perspective Q-values as `-child.mean_value` for every visited sibling. Unvisited moves received zero target weight. A temperature-0.25 softmax converted the measured Q-values into a policy, mixed 75% Q policy with 25% visit policy, and retained the unchanged teacher value target.

The 128-game Moka-turn-only corpus from opening offset 1,420,000 used 28 visits, root symmetry, exploration 2.0, FPU 0.25, and full branching. It contained 4,535 roots: 3,602 training, 430 validation, and 503 test rows by whole game. Each root evaluated 5.24 siblings on average. All targets were finite and normalized. The archive SHA-256 was `da1562c0023168ea4481090f3bebd63c494499abbab3c758213368888614df92`.

### Frozen-head continuation

Three two-epoch policy-head seeds used learning rate 0.00001, the existing 20,000-position browser on-policy corpus as preservation replay, and a 0.25 logit-preservation penalty.

The incumbent scored 60.0% validation and 65.6% test top-move agreement against the Q-derived targets. Seeds 60, 61, and 62 reached 58.6%, 58.6%, and 59.8% validation agreement and 65.0%, 65.0%, and 65.4% test agreement.

Every candidate regressed on both held-out gates. None entered the arena. Direct root-Q targets are now reproducible, but this corpus and frozen-head recipe did not improve action ranking.

## 2026-07-28 — Exact-topology FPU, exploration, and visits

### First-play urgency

FPU was retuned after restoring one-leaf search. On 20 games from opening offset 1,430,000, absolute-zero treatment and reductions 0, 0.1, 0.25, and 0.5 scored three, five, six, seven, and five wins. Their cap counts were zero, two, two, one, and two.

The deployed reduction 0.25 remained the clear joint winner and was retained.

### Exploration

On 20 games from opening offset 1,440,000, exploration coefficients 1.0, 1.5, 2.0, 2.5, and 3.0 scored ten, ten, nine, seven, and seven wins. Their cap counts were two, three, four, three, and three. Coefficient 1.0 was frozen as the smallest joint leader.

On 100 untouched games from offset 1,450,000, deployed 2.0 scored 31 completed wins with eight caps. Candidate 1.0 scored 29 completed wins with six caps. The candidate lost two completed games and was rejected.

### Visit count screen

On 20 games from opening offset 1,460,000, budgets 20, 24, 28, 32, and 40 scored two, five, four, six, and nine wins. Their cap counts were one, zero, zero, zero, and zero. Forty visits was frozen as the clear leader.

### Independent confirmation

Two disjoint 100-game blocks compared deployed 28 visits with the 40-visit candidate:

| Opening offset | 28 visits |   Caps | 40 visits |   Caps | 40-visit cap wins |
| -------------: | --------: | -----: | --------: | -----: | ----------------: |
|      1,470,000 |        30 |     10 |        31 |      4 |                 0 |
|      1,480,000 |        28 |      8 |        36 |      7 |                 1 |
|      **Total** |    **58** | **18** |    **67** | **11** |             **1** |

After conservatively excluding the cap-awarded result, 40 visits won 66 completed games versus 58 for 28 visits. Caps fell from 18 to 11. Runtime rose from 267.7 to 399.0 seconds across the 200 games.

Forty visits is accepted for the browser. It changes neither model weights nor download size and remains behind the existing interaction-only search gate.

## 2026-07-28 — Search budgets above forty visits

### Budget screen

The corrected one-leaf topology, root symmetry, exploration 2.0, FPU 0.25, full branching, and unchanged browser checkpoint were held fixed. On 20 fresh games from opening offset 1,490,000, budgets 40, 48, 56, 64, and 80 scored six, seven, ten, eleven, and nine wins. Their cap counts were one, two, one, zero, and one.

Sixty-four visits was frozen as the screen winner. It split its eleven wins as four Black and seven White, and every win completed normally.

### Sixty-four-visit confirmation

Two disjoint 100-game blocks compared the frozen 64-visit candidate with the deployed 40-visit search:

| Opening offset | 40 visits | 40 caps | 64 visits | 64 caps |
| -------------: | --------: | ------: | --------: | ------: |
|      1,500,000 |        35 |       7 |        41 |       4 |
|      1,510,000 |        40 |       5 |        39 |       9 |
|      **Total** |    **75** |  **12** |    **80** |  **13** |

Every win completed normally. Sixty-four visits added five completed wins but increased caps by one, so it failed the predeclared no-cap-regression gate and was rejected.

### Late-budget schedule

A research-only schedule allowed a higher early budget to return to 40 visits after a move-count cutoff. Regression tests cover the disabled and active scheduling paths.

On 20 new games from offset 1,520,000, fixed 40 visits scored ten wins with one cap. Fixed 64 visits and 64-to-40 schedules beginning at moves 40, 60, and 80 each scored nine wins with one cap. The phase schedule lost a completed game and was rejected.

### Fifty-six-visit fallback

The predeclared screen runner-up was evaluated on the same two 100-game confirmation blocks:

| Opening offset | 40 visits | 40 caps | 56 visits | 56 caps |
| -------------: | --------: | ------: | --------: | ------: |
|      1,500,000 |        35 |       7 |        39 |       8 |
|      1,510,000 |        40 |       5 |        37 |       4 |
|      **Total** |    **75** |  **12** |    **76** |  **12** |

Every win completed normally. Fifty-six visits added one completed win without increasing caps, model bytes, or browser payload. It is accepted for browser validation.

## 2026-07-28 — Sibling-Q normalization

### Method

Moka's visited sibling values were optionally min-max normalized before PUCT selection, then blended with their raw values. The method used only Moka's own value estimates and did not change the model, evaluation count, legal moves, or final visit-count selection.

The implementation is disabled by default and supports all-node or root-only normalization. Regression tests verify that normalization can amplify a small value difference, that root-only mode disables the transform below the root, and that the unchanged default retains raw values.

### Screen

On 20 fresh games from opening offset 1,530,000, all settings used 56 visits, one-leaf expansion, root symmetry, exploration 2.0, FPU 0.25, and full branching:

| Normalization | Blend | Wins | Caps |
| ------------- | ----: | ---: | ---: |
| Disabled      |  0.00 |    8 |    0 |
| All nodes     |  0.25 |    8 |    2 |
| All nodes     |  0.50 |   13 |    2 |
| All nodes     |  1.00 |    4 |    3 |
| Root only     |  0.25 |    9 |    1 |
| Root only     |  0.50 |    6 |    2 |
| Root only     |  1.00 |    5 |    1 |

The 0.50 all-node blend was frozen as the only large screen gain. It split six Black and seven White wins, with no cap-awarded wins.

### Fresh confirmation

On 100 untouched games from opening offset 1,540,000, the disabled control scored 45 wins with four caps. The frozen 0.50 all-node candidate scored 36 wins with ten caps. Every win completed normally.

Sibling-Q normalization lost nine completed games and added six caps. It was rejected after the first confirmation block and was never added to the browser.

## 2026-07-28 — Geometric root-symmetry consensus

### Method

The deployed root evaluator arithmetically averaged the eight aligned symmetry policies. The candidate also computed their normalized geometric mean, then blended 25% geometric consensus with 75% arithmetic probability. This suppresses moves supported by only a few orientations while retaining the calibrated arithmetic distribution.

The model, value average, legal moves, 56-visit budget, one-leaf topology, exploration 2.0, FPU 0.25, full branching, and final maximum-visit selection were unchanged. The candidate uses only Moka's eight existing root evaluations and adds no model query, teacher information, rule heuristic, or payload tensor.

### Screen

On 20 fresh games from opening offset 1,550,000:

| Geometric blend | Wins | Caps |
| --------------: | ---: | ---: |
|            0.00 |    2 |    2 |
|            0.25 |    3 |    0 |
|            0.50 |    3 |    1 |
|            0.75 |    3 |    1 |
|            1.00 |    3 |    1 |

No setting received a cap-awarded win. The 0.25 blend was frozen because it matched the best completed-win result with the fewest caps and made the smallest change to the incumbent distribution.

### Independent confirmation

Two disjoint 100-game blocks compared the frozen blend with arithmetic averaging:

| Opening offset | Arithmetic wins | Arithmetic caps | Geometric wins | Geometric caps |
| -------------: | --------------: | --------------: | -------------: | -------------: |
|      1,560,000 |              24 |               6 |             27 |              4 |
|      1,570,000 |              32 |               7 |             31 |              6 |
|      **Total** |          **56** |          **13** |         **58** |         **10** |

Every win completed normally. The arithmetic runs took 269.9 and 264.7 seconds; the geometric runs took 263.7 and 264.0 seconds. The candidate added two completed wins, removed three caps, and introduced no measurable runtime cost across the 200-game confirmation.

The 0.25 geometric consensus blend is accepted for the browser.

## 2026-07-28 — Trimmed symmetry outliers

The accepted 0.25 geometric blend was held fixed. A research-only robust aggregate sorted the eight aligned probabilities independently for every move, discarded the highest and lowest orientation, averaged the remaining six, and blended that trimmed policy with the accepted aggregate. It changed no model evaluation, value, legal move, search budget, or payload tensor.

On 20 fresh games from opening offset 1,580,000:

| Trimmed blend | Wins | Caps |
| ------------: | ---: | ---: |
|          0.00 |    7 |    0 |
|          0.25 |    6 |    0 |
|          0.50 |    6 |    0 |
|          0.75 |    6 |    0 |
|          1.00 |    7 |    0 |

Every win completed normally. Partial trimming lost one completed game, while a fully trimmed policy only tied the accepted control. No setting advanced to confirmation. The browser remains on the 0.25 geometric consensus blend without trimming.

## 2026-07-28 — Trimmed symmetry values

The accepted geometric policy remained fixed while the root evaluator optionally replaced part of the arithmetic mean of its eight symmetry values with a mean that discarded the highest and lowest values. This changes only Moka's existing root value aggregate, which can affect parent-value FPU; it adds no inference or runtime information.

On 20 fresh games from opening offset 1,590,000, trimmed-value weights 0, 0.25, 0.50, 0.75, and 1.00 all reproduced the same three wins, one cap, Black/White split, pass counts, and capped-position counts. Every win completed normally.

The transform was behaviorally inert across the complete range and did not advance to confirmation. The browser retains arithmetic symmetry-value averaging.

## 2026-07-28 — Top-eight symmetry rank consensus

### Method

Each aligned symmetry policy ranked its eight highest-probability moves. Linear Borda weights from eight through one were summed across orientations and normalized into a rank policy. The candidate blended this distribution with the accepted 0.25-geometric aggregate. The method uses only Moka's existing eight root evaluations and does not change its value, legal moves, tree budget, or runtime teacher access.

### Screen

On 20 fresh games from opening offset 1,600,000:

| Rank blend | Wins | Caps |
| ---------: | ---: | ---: |
|       0.00 |    5 |    1 |
|       0.10 |    6 |    1 |
|       0.25 |    5 |    1 |
|       0.50 |    7 |    1 |
|       0.75 |    7 |    1 |
|       1.00 |    6 |    1 |

Every win completed normally. Weight 0.50 was frozen as the smallest joint screen leader.

### Fresh confirmation

On 100 untouched games from opening offset 1,610,000, the accepted geometric control scored 27 wins with eight caps. The frozen rank candidate scored 32 wins with ten caps. Every win completed normally; the control took 270.4 seconds and the candidate took 273.4 seconds.

Rank consensus added five completed wins but also two unfinished games. It failed the predeclared no-cap-regression gate after the first confirmation block and was rejected without retuning or a second block. The browser remains unchanged.

## 2026-07-28 — Phase-limited rank consensus

### Hypothesis

Full-game rank consensus added wins and caps. A deterministic schedule therefore used the frozen 0.50 top-eight rank blend only before a move-count cutoff, then returned to the accepted geometric policy. The hypothesis was that earlier rank agreement contained the useful signal while its coarse late-game distribution caused loops.

### Screen

On 20 fresh games from opening offset 1,620,000:

| Rank schedule | Wins | Caps |
| ------------- | ---: | ---: |
| Disabled      |    5 |    2 |
| Full game     |    6 |    1 |
| End at 24     |    6 |    2 |
| End at 40     |    8 |    3 |
| End at 56     |    6 |    1 |
| End at 72     |    6 |    1 |

Every win completed normally. The move-40 schedule added the most wins but regressed caps and was rejected. Move 56 was frozen as the earliest schedule that added a completed win while removing a cap.

### Fresh confirmation

On 100 untouched games from opening offset 1,630,000, the geometric control scored 34 wins with seven caps. The frozen move-56 schedule scored 33 wins with six caps. Every win completed normally; the control took 278.2 seconds and the candidate took 282.6 seconds.

Ending rank consensus before the late game removed one cap but also lost one completed game. It failed the strength gate after the first confirmation block and was rejected without testing another cutoff on the confirmation openings. Production remains unchanged.

## 2026-07-28 — Confidence-gated rank consensus

### Hypothesis

The top-eight rank blend was activated only when at least a configured number of the eight aligned symmetry policies independently selected the same top move. This tests whether rank consensus is useful only at roots with genuine orientation agreement. The frozen rank weight remained 0.50 and the rank width remained eight.

### Screen

On 20 fresh games from opening offset 1,640,000:

| Required top-move votes | Wins | Caps |
| ----------------------: | ---: | ---: |
|                Disabled |    4 |    3 |
|                       0 |    5 |    3 |
|                       2 |    5 |    3 |
|                       3 |    5 |    3 |
|                       4 |    6 |    3 |
|                       5 |    5 |    3 |
|                       6 |    5 |    3 |

Every win completed normally. Four votes was frozen as the clear screen leader.

### Fresh confirmation

On 100 untouched games from opening offset 1,650,000, the accepted geometric control scored 35 wins with seven caps. The four-vote rank candidate scored 38 wins with eleven caps. Every win completed normally; the control took 263.5 seconds and the candidate took 278.9 seconds.

Confidence gating added three completed wins but four unfinished games. It failed the predeclared no-cap-regression gate after the first confirmation block and was rejected without testing another threshold on the confirmation openings. The browser remains unchanged.

## 2026-07-28 — Late area-value blend

The current area score was blended into Moka's value estimate only as the game progressed, beginning at move 40 and reaching its configured maximum at move 80. This used only the public board state and ordinary area-scoring rules. It did not query KataGo, inspect future moves, change legal moves, or add model evaluations.

On 20 fresh games from opening offset 1,660,000, maximum area weights 0, 0.05, 0.10, 0.20, 0.30, and 0.50 all scored eight wins with zero caps. Every setting preserved the five-Black, three-White win split. Weights at or above 0.20 changed pass behavior but not a single outcome.

The phase-ramped area signal was behaviorally neutral at the measured strength gate and did not advance to confirmation. Production retains the learned value without an area-score blend.

## 2026-07-28 — Top-choice symmetry voting

The broad top-eight Borda blend was replaced with a sparse distribution containing only the move each aligned symmetry policy ranked first. Blending this vote distribution into the accepted geometric aggregate tests whether independent top-choice agreement preserves the useful symmetry signal without reshaping dozens of lower-priority moves.

On 20 fresh games from opening offset 1,670,000:

| Top-choice vote blend | Wins | Caps |
| --------------------: | ---: | ---: |
|                  0.00 |    7 |    2 |
|                  0.10 |    6 |    2 |
|                  0.25 |    5 |    2 |
|                  0.50 |    5 |    1 |
|                  0.75 |    5 |    1 |
|                  1.00 |    5 |    1 |

Every win completed normally. The smallest vote blend lost one completed game, while every larger blend lost two. Top-choice voting was rejected without confirmation and was never added to the browser.

## 2026-07-28 — Rank consensus with late area value

### Hypothesis

Full-game rank consensus had repeatedly added wins and caps. The phase-ramped area value was neutral by itself but changed late pass behavior. Combining them tested whether the rules-derived late signal could preserve rank consensus's tactical gain while removing its unfinished games.

### Screen

On 20 fresh games from opening offset 1,680,000, production and the 0.50 rank blend each scored seven wins with two caps. Adding late area weights 0.10, 0.20, and 0.50 to the rank blend also scored seven wins with two caps. Weight 0.30 scored eight wins with two caps and was frozen as the only completed-win improvement.

### Fresh confirmation

On 100 untouched games from opening offset 1,690,000, production scored 35 wins with four caps. The frozen rank-plus-area candidate scored 36 wins with eight caps. Every win completed normally; production took 256.5 seconds and the candidate took 279.7 seconds.

The combined candidate added one completed win but doubled unfinished games. It failed the no-cap-regression gate after the first confirmation block and was rejected. Production remains on geometric symmetry consensus without rank or area blending.

## 2026-07-28 — PUCT value weight under geometric consensus

### Screen

The policy–value balance was retuned because the previous sweep used a 28-visit player before geometric symmetry consensus. The exact current checkpoint, 56-visit budget, one-leaf topology, root symmetry, 0.25 geometric policy blend, exploration 2.0, FPU 0.25, full branching, and maximum-visit move selection were held fixed.

On 20 fresh games from opening offset 1,700,000:

| Value weight | Wins | Caps |
| -----------: | ---: | ---: |
|         0.50 |    8 |    3 |
|         0.75 |    8 |    2 |
|         1.00 |    8 |    2 |
|         1.25 |   11 |    0 |
|         1.50 |    7 |    1 |
|         2.00 |    8 |    1 |

Weight 1.25 was frozen as the unique joint screen winner. No intermediate value was tested after observing the screen.

### Independent confirmation

Two disjoint 100-game blocks compared the frozen candidate with production:

| Opening offset | Weight 1.00 | 1.00 caps | Weight 1.25 | 1.25 caps |
| -------------: | ----------: | --------: | ----------: | --------: |
|      1,710,000 |          32 |        10 |          34 |         7 |
|      1,720,000 |          39 |         6 |          45 |         9 |
|      **Total** |      **71** |    **16** |      **79** |    **16** |

Every win completed normally. Weight 1.25 added eight completed wins without increasing caps, model evaluations, model bytes, or browser payload. It is accepted for the browser.

## 2026-07-28 — Post-value-weight first-play urgency

First-play urgency was retuned because increasing the PUCT value weight changes the penalty's effective influence on unvisited moves. The accepted checkpoint, 56-visit budget, one-leaf topology, root symmetry, 0.25 geometric policy blend, exploration 2.0, value weight 1.25, full branching, and maximum-visit move selection were held fixed.

On 20 fresh games from opening offset 1,730,000:

| FPU reduction | Wins | Caps |
| ------------: | ---: | ---: |
| Absolute zero |    5 |    1 |
|          0.00 |    7 |    0 |
|          0.10 |    7 |    0 |
|          0.25 |    7 |    0 |
|          0.40 |    5 |    1 |
|          0.50 |    7 |    1 |

Every win completed normally. Reductions 0, 0.10, and the accepted 0.25 tied exactly on wins and caps. No setting improved the joint result, so no candidate advanced to confirmation and production retains 0.25.

## 2026-07-28 — Post-value-weight exploration

Exploration was retuned on a new opening block after the FPU screen retained 0.25. The accepted value weight 1.25 and every other deployed search setting were held fixed.

On 20 fresh games from opening offset 1,740,000:

| Exploration | Wins | Caps |
| ----------: | ---: | ---: |
|        1.25 |    3 |    1 |
|        1.50 |    6 |    1 |
|        1.75 |    5 |    2 |
|        2.00 |    8 |    2 |
|        2.25 |    5 |    1 |
|        2.50 |    5 |    1 |

Every win completed normally. The deployed 2.0 coefficient remained the unique win leader. No alternative advanced to confirmation.

## 2026-07-28 — Post-value-weight root-policy temperature

The symmetry-aggregated root prior was sharpened or flattened before allocating the accepted 56 visits. The checkpoint, geometric consensus, value weight 1.25, exploration 2.0, FPU 0.25, full branching, and maximum-visit selection were held fixed.

On 20 fresh games from opening offset 1,750,000:

| Temperature | Wins | Caps |
| ----------: | ---: | ---: |
|        0.50 |    8 |    0 |
|        0.75 |   10 |    1 |
|        1.00 |    7 |    3 |
|        1.25 |    8 |    1 |
|        1.50 |    6 |    0 |

Temperature 0.75 was frozen as the screen winner without testing an intermediate value.

On 100 untouched games from opening offset 1,760,000, production scored 38 wins with two caps. The frozen candidate reported 47 wins with five caps, including one cap-awarded win, so it completed 46 wins normally.

Sharpening added eight normally completed wins but more than doubled unfinished games. It failed the predeclared no-cap-regression gate after the first confirmation block and was rejected. Production retains temperature 1.0.

## 2026-07-28 — Post-value-weight geometric consensus

The geometric root-policy blend was retuned because its accepted 0.25 value predated the higher PUCT value weight. The accepted checkpoint, 56 visits, one-leaf topology, root symmetry, exploration 2.0, value weight 1.25, FPU 0.25, full branching, and maximum-visit selection were held fixed.

On 20 fresh games from opening offset 1,770,000:

| Geometric blend | Wins | Caps |
| --------------: | ---: | ---: |
|            0.00 |   14 |    1 |
|            0.10 |   14 |    1 |
|            0.25 |   11 |    1 |
|            0.40 |   11 |    1 |
|            0.50 |   10 |    1 |
|            0.75 |   10 |    2 |

Weights 0 and 0.10 tied on the joint outcome. Weight 0.10 was frozen because it was the smaller change from production.

On 100 untouched games from opening offset 1,780,000, production 0.25 scored 33 completed wins with six caps. Candidate 0.10 scored 35 completed wins with seven caps. Neither side received a cap-awarded win.

The candidate added two completed wins but also added an unfinished game. It failed the no-cap-regression gate after the first confirmation block and was rejected. Production retains the 0.25 geometric blend.

## 2026-07-28 — Phase-limited root-policy sharpening

Full-game temperature 0.75 added completed wins and caps. A deterministic research schedule therefore sharpened the root prior only before a move-count cutoff, then returned to temperature 1.0. The option is disabled by default and regression-tested at the cutoff boundary.

On 20 fresh games from opening offset 1,790,000:

| Temperature schedule | Wins | Caps |
| -------------------: | ---: | ---: |
|            Fixed 1.0 |    8 |    1 |
|           Fixed 0.75 |    6 |    0 |
|      0.75 to move 40 |    5 |    1 |
|      0.75 to move 56 |    6 |    0 |
|      0.75 to move 72 |    6 |    0 |
|      0.75 to move 88 |    6 |    0 |

Every win completed normally. Every sharpened configuration lost at least two completed games. The cap reduction therefore did not represent a strength improvement, and no schedule advanced to confirmation. Production retains fixed temperature 1.0.

## 2026-07-28 — Cap diagnosis and explicit resignation

### Capped-game traces

The arena gained an opt-in capped-game trace containing the deterministic opening identifier, colors, final area score, repetition count, complete move history, and final board. Replaying the six production caps at opening offset 1,780,000 showed one consistent failure:

- every cap was a Moka loss as White;
- every final area score was +74 for Black;
- no position repeated;
- Moka repeatedly selected pass while KataGo continued filling Black territory.

The failure was not ko cycling. The existing two-pass game ending could not represent resignation, so a hopeless player could pass dozens of times without ending the game.

### Rejected opponent-pass rule

An initial rule accepted an opponent pass when current-board area was sufficiently unfavorable. On the previously measured temperature-0.75 block at offset 1,760,000, margin 20 reduced the candidate from 47 reported wins to 43 while leaving all five caps. It could concede recoverable positions and did not address the actual self-pass pattern. The rule was removed.

### Explicit self-pass resignation

The replacement records an ordinary loss only when all of these conditions hold:

- Moka itself selected pass;
- at least 20 moves have been played;
- current-board area is at least a configured margin against Moka.

It uses only the visible board and Moka's own move. It does not query KataGo, inspect future play, alter legal moves, or award a win.

On the known offset-1,760,000 diagnostic block, margins 40 and 60 both preserved all 47 reported outcomes while reducing caps from five to one. Margin 40 recorded five resignations; margin 60 recorded four. The remaining cap was a repetition-heavy game Moka led by ten points, so resignation correctly did not fire. Margin 60 was frozen as the more conservative threshold.

## 2026-07-28 — Root-policy temperature with resignation

### Fresh screen

On 20 fresh games from opening offset 1,820,000:

| Configuration                | Wins | Caps | Resignations |
| ---------------------------- | ---: | ---: | -----------: |
| Production, temperature 1.00 |    7 |    2 |            0 |
| Temperature 1.00, margin 60  |    7 |    0 |            2 |
| Temperature 0.95, margin 60  |    7 |    0 |            0 |
| Temperature 0.90, margin 60  |    7 |    0 |            1 |
| Temperature 0.85, margin 60  |    9 |    0 |            1 |
| Temperature 0.75, margin 60  |    8 |    0 |            2 |

Temperature 0.85 was frozen as the only two-win screen improvement. Resignation alone preserved all wins and removed both caps.

### Independent confirmation

Two untouched 100-game blocks compared the frozen temperature candidate with production:

| Opening offset | Production wins | Production caps | Candidate wins | Candidate caps | Candidate resignations |
| -------------: | --------------: | --------------: | -------------: | -------------: | ---------------------: |
|      1,830,000 |              37 |               2 |             39 |              1 |                      5 |
|      1,840,000 |              46 |              13 |             44 |              0 |                      8 |
|      **Total** |          **83** |          **15** |         **83** |          **1** |                 **13** |

Every reported win completed normally. Temperature 0.85 did not improve aggregate completed wins and was rejected as a strength change. The cap reduction established that explicit resignation generalized, but it did not justify attributing strength to the temperature candidate.

## 2026-07-28 — Sixty-four visits with resignation

### Budget screen

The accepted temperature 1.0, value weight 1.25, exploration 2.0, FPU 0.25, geometric consensus 0.25, one-leaf topology, full branching, and margin-60 resignation were held fixed. On 20 fresh games from opening offset 1,850,000:

|                   Visits | Wins | Caps | Resignations |
| -----------------------: | ---: | ---: | -----------: |
| 56, resignation disabled |    8 |    0 |            0 |
|                       56 |    8 |    0 |            0 |
|                       64 |    9 |    0 |            0 |
|                       72 |    8 |    0 |            0 |
|                       80 |    7 |    0 |            0 |
|                       96 |    4 |    0 |            2 |

Sixty-four visits was frozen as the unique win leader. Larger budgets were weaker rather than merely slower.

### Independent confirmation

Two untouched 100-game blocks compared the frozen candidate with the 56-visit control:

| Opening offset | 56 visits | 56 caps | 64 visits | 64 caps | 64 resignations |
| -------------: | --------: | ------: | --------: | ------: | --------------: |
|      1,860,000 |        33 |      10 |        34 |       0 |               8 |
|      1,870,000 |        36 |       6 |        40 |       0 |              10 |
|      **Total** |    **69** |  **16** |    **74** |   **0** |          **18** |

Every reported win completed normally. Sixty-four visits added five completed wins and explicit resignation removed all 16 unfinished games. Aggregate runtime rose from 545.5 to 605.7 seconds, about 11%. The model, teacher, legal moves, and payload were unchanged.

The 64-visit, margin-60 configuration is accepted for Moka research. Per user direction, it is not being added to or deployed on the Million website.

The Python arena defaults now reproduce the accepted research player: 64 visits, root symmetry enabled, exploration 2.0, FPU 0.25, value weight 1.25, geometric consensus 0.25, and resignation margin 60. Explicit negative flags remain available for controls. The browser package, model files, and Million website remain unchanged.

## 2026-07-28 — Full-network distillation from the accepted search player

### Hypothesis

Earlier 28-visit search distillation improved held-out policy loss but regressed in the arena when only the policy head adapted. The accepted 64-visit player supplies a stronger target. A low-learning-rate update to the complete 104,129-parameter network may transfer some of that search improvement into the checkpoint while a policy-preservation penalty and the existing browser on-policy replay set limit forgetting.

KataGo remains an offline teacher for target values and the opponent in evaluation. It is never queried while Moka selects an arena move.

### Fresh trajectory corpus

The accepted checkpoint generated 128 teacher-opponent games from opening offset 1,900,000. Collection used the accepted 64-visit search, root symmetry, exploration 2.0, FPU 0.25, value weight 1.25, geometric policy blend 0.25, and margin-60 resignation. Only Moka decision positions were retained.

The resulting corpus contained 4,246 positions. Each policy target blended 75% of the 64-visit distribution with 25% of the legal symmetry-averaged root policy. Teacher values supplied the value targets. Complete games were assigned to train, validation, and test buckets before training.

Corpus SHA-256:

```text
9b85b38a9154f86c40b58441d24a0558cc9c8aa892893dcaaaceb141a4b227c5
```

The collector now applies the same explicit self-pass resignation rule as the arena. The selected pass is retained as a training decision, then a hopeless trajectory ends rather than accumulating artificial pass/fill positions.

### Frozen training recipe

Three seeds used the same predeclared recipe:

- one epoch;
- batch size 128;
- learning rate 0.000005;
- full nested network adaptation;
- policy-preservation weight 0.25 against the incumbent;
- 20,000 existing browser on-policy positions as training-only replay;
- validation-loss checkpoint selection.

No seed-specific hyperparameter was changed after observing its result.

### Held-out gate

| Checkpoint | Search test loss | Search top move | Search value MAE | Replay test loss | Replay top move | Replay value MAE |
| ---------- | ---------------: | --------------: | ---------------: | ---------------: | --------------: | ---------------: |
| Incumbent  |           1.8717 |           66.7% |           0.3557 |           2.6044 |           54.5% |           0.2494 |
| Seed 71    |           1.8507 |           68.3% |           0.3024 |           2.5121 |           55.0% |           0.2281 |
| Seed 72    |           1.8459 |           67.8% |           0.2874 |           2.5143 |           55.0% |           0.2340 |
| Seed 73    |           1.8492 |           68.0% |           0.2990 |           2.5108 |           54.7% |           0.2288 |

All three candidates passed the offline gate.

### Arena screen

The incumbent and candidates played the same 20 fresh games from opening offset 1,910,000. The arena used the accepted 64-visit player and margin-60 resignation.

| Checkpoint | Wins | Caps | Resignations |
| ---------- | ---: | ---: | -----------: |
| Incumbent  |    6 |    0 |            3 |
| Seed 71    |    9 |    0 |            1 |
| Seed 72    |    6 |    0 |            1 |
| Seed 73    |    8 |    0 |            1 |

Seed 71 was frozen as the only three-win screen improvement. Seeds 72 and 73 were rejected without confirmation.

### Independent confirmation

Two untouched 100-game blocks compared the frozen seed-71 candidate with the incumbent on identical openings and colors:

| Opening offset | Incumbent wins | Incumbent caps | Candidate wins | Candidate caps | Candidate resignations |
| -------------: | -------------: | -------------: | -------------: | -------------: | ---------------------: |
|      1,920,000 |             42 |              0 |             43 |              0 |                      4 |
|      1,930,000 |             35 |              1 |             41 |              0 |                      7 |
|      **Total** |         **77** |          **1** |         **84** |          **0** |                 **11** |

The candidate improved both disjoint blocks, added seven completed wins in aggregate, and removed the incumbent's single unfinished game. It is accepted as the new Moka checkpoint.

### INT8 deployment-format gate

The exact exported INT8 tensors were dequantized into the Python evaluator so the deployable artifact could be compared with the float checkpoint and previous INT8 artifact.

On the untouched search-corpus test split, INT8 quantization preserved 68.3% top-move agreement exactly. Loss changed from 1.8507 to 1.8531, and value MAE changed from 0.3024 to 0.2933.

A 20-game sanity block at offset 1,940,000 scored nine wins for float and eight for INT8, both with zero caps. Because that one-game difference was inconclusive, two new 100-game blocks directly compared the previous and candidate INT8 artifacts:

| Opening offset | Previous INT8 wins | Previous caps | Candidate INT8 wins | Candidate caps |
| -------------: | -----------------: | ------------: | ------------------: | -------------: |
|      1,950,000 |                 40 |             0 |                  43 |              0 |
|      1,960,000 |                 46 |             0 |                  48 |              0 |
|      **Total** |             **86** |         **0** |              **91** |          **0** |

The exact candidate artifact improved both quantized blocks, adding five wins without a cap regression. It passes the deployment-format gate.

The network remains 104,129 parameters. Its INT8 browser artifact remains 111,920 bytes. The accepted float checkpoint has SHA-256 `8640f8687c95d17bd938ed8b732270a54faa9fc0f838760209f2932761015004`; the exported weights have SHA-256 `80e9188841a23e9f83bbb14cc8faa92f23cf2c61e86ec12e32289aafc34329d4`.

Only the Moka repository artifact is updated. The Million website is intentionally unchanged.

## 2026-07-28 — Repeated full-network search distillation

### Hypothesis

A second on-policy distillation round from the newly accepted checkpoint might compound the first-round gain by exposing the model to positions created by its changed policy.

### Corpus and frozen recipe

The accepted checkpoint generated 128 new teacher-opponent games from opening offset 2,000,000 with the same 64-visit collection settings. The corpus contained 4,491 Moka decision positions and had SHA-256 `0739683eef886401a25b7f0cc9a8029b43680939bf192f54df3cc5d4e97713eb`.

Seeds 81, 82, and 83 used the accepted one-epoch, 0.000005-learning-rate, full-network recipe. The accepted checkpoint supplied the preservation reference. Both the first-round search corpus and the original 20,000-position browser on-policy corpus were training-only replay.

All candidates reduced held-out loss on the new and first-round search corpora without materially regressing the replay set. On the new corpus, incumbent test loss was 2.0614 and candidate losses were 2.0117, 2.0148, and 2.0074.

### Arena screen

On 20 fresh games from opening offset 2,010,000:

| Checkpoint | Wins | Caps | Resignations |
| ---------- | ---: | ---: | -----------: |
| Incumbent  |   11 |    0 |            2 |
| Seed 81    |   10 |    0 |            2 |
| Seed 82    |    8 |    0 |            1 |
| Seed 83    |   10 |    0 |            1 |

No candidate matched the incumbent's wins. The entire round is rejected without confirmation. Lower distillation loss did not compound arena strength, so further seeds of the same recipe are not justified.

## 2026-07-28 — Conservative round-two interpolation

### Hypothesis

Round-two seed 83 improved policy loss and value MAE but lost one game in its screen. Linear interpolation might retain the accepted checkpoint's decisions while importing a smaller part of the candidate's value improvement.

Fixed 25%, 50%, and 75% candidate-weight blends were created before a new screen. On the round-two test games, their value MAE improved monotonically from the incumbent's 0.4805 to 0.4712, 0.4625, and 0.4530.

### Screen and confirmation

On 20 fresh games from opening offset 2,020,000, the incumbent scored 12 wins. The 25%, 50%, and 75% blends scored 13, 12, and 10. Every player had zero caps. The 25% blend was frozen as the only screen improvement.

Two untouched 100-game blocks produced:

| Opening offset | Incumbent wins | Incumbent caps | Blend wins | Blend caps |
| -------------: | -------------: | -------------: | ---------: | ---------: |
|      2,030,000 |             38 |              0 |         40 |          0 |
|      2,040,000 |             41 |              0 |         34 |          0 |
|      **Total** |         **79** |          **0** |     **74** |      **0** |

The first-block gain reversed on the second block. The blend lost five games in aggregate and is rejected. Neither a second full distillation step nor partial interpolation compounds the accepted first-round checkpoint.

## 2026-07-28 — Hard-mixture round-two distillation

### Hypothesis

The repeated dense objective improved loss without strength. Mixing 25% of the 64-visit target's top move with 75% of its dense distribution might emphasize decisive search choices while preserving secondary-action information.

Seeds 84, 85, and 86 used the same round-two full-network recipe, corpora, preservation reference, and learning rate as the rejected dense continuation. Only the hard-target weight changed from zero to 0.25.

All three candidates passed the offline gate. Round-two test loss improved from 2.0614 to 2.0063–2.0102. Top-move agreement remained 70.6%–71.0% versus 70.8%, and replay top-move agreement increased from 55.0% to 55.4%–55.8%.

On 20 fresh games from opening offset 2,050,000:

| Checkpoint | Wins | Caps | Resignations |
| ---------- | ---: | ---: | -----------: |
| Incumbent  |    7 |    0 |            1 |
| Seed 84    |    6 |    0 |            0 |
| Seed 85    |    7 |    0 |            1 |
| Seed 86    |    6 |    0 |            1 |

No seed improved wins. The hard-mixture branch is rejected without confirmation. Dense loss, hard-target agreement, and replay agreement all remain insufficient proxies for search strength.

## 2026-07-28 — Isolated value-head correction

### Hypothesis

The accepted full-network update improved both policy and value, while repeated full-network updates improved offline metrics but regressed in play. Fitting only the value head to fresh accepted-player roots would preserve policy logits exactly and test whether search benefits from better pointwise values without policy drift.

The two 128-game search corpora were combined by complete game into 8,737 roots. The combined archive had SHA-256 `a48ee16a9a0ab3767a8bc5b622feecf97ec9bcad5519899b0199ea56d44c9a28`.

Seeds 87, 88, and 89 trained only the existing value convolution, hidden layer, and output for ten epochs at learning rate 0.0001. Tensor comparison confirmed that every trunk and policy parameter remained exactly equal to the incumbent.

The incumbent's combined test value MAE was 0.4143. Candidate MAEs were 0.3134, 0.3096, and 0.3094.

On 20 fresh games from opening offset 2,060,000:

| Checkpoint | Wins | Caps | Resignations |
| ---------- | ---: | ---: | -----------: |
| Incumbent  |    8 |    0 |            0 |
| Seed 87    |    8 |    0 |            0 |
| Seed 88    |    6 |    0 |            0 |
| Seed 89    |    7 |    0 |            0 |

No value head improved wins, so none advanced. A roughly 25% reduction in pointwise value MAE did not improve search strength. Future value work must target action-relative ordering or search dynamics rather than another pointwise fit.

## 2026-07-28 — Accepted-search child-value ranking

### Hypothesis

Pointwise value correction reduced MAE without improving play. PUCT depends on relative action values, so training the value head to order siblings from the accepted 64-visit search might provide a better-aligned signal while preserving the policy and trunk exactly.

### Collector and corpus

The search collector gained an opt-in child-value path. For every visited root child it records:

- the exact child-state Moka features;
- the search child's mean value in the child player's perspective;
- the visit count as confidence weight;
- the complete-game ID and parent-root index.

The pair builder accepts these search-collector arrays directly. Regression tests verify that unvisited children are excluded, child perspective is preserved, and parent groups form the intended low-opponent-value preference.

The 128-game corpus from opening offset 2,080,000 contained 4,310 roots, 29,490 visited children, and 14,429 sibling comparisons with a minimum value gap of 0.1. Complete-game splits retained 11,429 training, 1,346 validation, and 1,654 test pairs. The archive SHA-256 was `cf436f72c196ef2761a484bc8c8a10ac64339c987e601714e6ee64d51fb50967`.

### Training and offline gate

Seeds 90, 91, and 92 trained only the six existing value-head tensors for ten epochs at learning rate 0.0001. Pointwise child-value loss and sibling-ranking loss had equal weight. Every policy and trunk tensor remained exactly equal to the incumbent.

The incumbent's test pair MSE was 0.0788. Candidate pair MSEs were 0.0698, 0.0703, and 0.0699. Pointwise test MAE changed from 0.1705 to 0.1791, 0.1790, and 0.1764, an intentional trade toward ordering.

### Arena result

On 20 fresh games from opening offset 2,090,000, the incumbent scored eight wins. Seeds 90, 91, and 92 scored nine, ten, and ten, all with zero caps. Seed 92 was frozen using its best validation ranking checkpoint and lower pointwise test MAE among the tied screen winners.

Two untouched 100-game blocks produced:

| Opening offset | Incumbent wins | Incumbent caps | Seed 92 wins | Seed 92 caps |
| -------------: | -------------: | -------------: | -----------: | -----------: |
|      2,100,000 |             46 |              0 |           42 |            0 |
|      2,110,000 |             44 |              0 |           46 |            0 |
|      **Total** |         **90** |          **0** |       **88** |        **0** |

The candidate improved the second block but regressed by four wins on the first, losing two games in aggregate. It is rejected. The child-target collector remains because the experiment is leakage-free and reproducible, but the accepted checkpoint and INT8 artifact are unchanged.

## 2026-07-28 — Phase-specific PUCT value weight

### Hypothesis

The pointwise and action-relative value heads both failed despite improving their offline targets. The existing value might instead be useful at a different strength relative to the policy prior after the middle game. An opt-in schedule retained value weight 1.25 through move 49 and changed it from move 50 onward without adding evaluations or model bytes.

Regression coverage verifies the cutoff and disabled behavior.

### Screen

On 20 fresh games from opening offset 2,120,000:

| Late value weight | Wins | Caps | Resignations |
| ----------------: | ---: | ---: | -----------: |
|     Constant 1.25 |    7 |    0 |            3 |
|              0.50 |    7 |    0 |            4 |
|              0.75 |    8 |    0 |            4 |
|              1.00 |    7 |    0 |            4 |
|              1.50 |    7 |    0 |            3 |
|              2.00 |    7 |    0 |            4 |

Late weight 0.75 was frozen as the only screen improvement.

### Independent confirmation

| Opening offset | Constant wins | Constant caps | Late 0.75 wins | Late 0.75 caps |
| -------------: | ------------: | ------------: | -------------: | -------------: |
|      2,130,000 |            43 |             0 |             42 |              0 |
|      2,140,000 |            42 |             1 |             40 |              0 |
|      **Total** |        **85** |         **1** |         **82** |          **0** |

The schedule removed one unfinished game but lost three completed wins. It is rejected and remains disabled by default. Production research play retains constant value weight 1.25.

## 2026-07-28 — Phase-specific visit allocation

### Hypothesis

Global budgets above 64 visits were weaker, but that result did not isolate the endgame. Keeping 64 visits through move 59 and changing only the late budget tested whether fewer noisy evaluations or greater tactical resolution improved completion.

On 20 fresh games from opening offset 2,150,000:

| Late visits | Wins | Black | White | Caps | Runtime |
| ----------: | ---: | ----: | ----: | ---: | ------: |
| Constant 64 |    8 |     4 |     4 |    0 |   64.1s |
|          32 |    8 |     4 |     4 |    0 |   60.1s |
|          48 |    8 |     4 |     4 |    0 |   62.0s |
|          80 |    8 |     4 |     4 |    0 |   66.0s |
|          96 |    8 |     4 |     4 |    0 |   68.4s |
|         128 |    8 |     4 |     4 |    0 |   71.6s |

Every schedule tied exactly on wins, colors, caps, and resignations. Thirty-two late visits reduced runtime by about 6%, but speed alone does not satisfy the strength gate. No schedule advanced, and constant 64 visits remain the accepted research default.

## 2026-07-28 — Stronger native b18 teacher on accepted-Moka trajectories

### Hypothesis

Repeated b6 and self-search targets stopped compounding. A stronger native 9×9 b18 teacher might provide useful policy corrections that b6 cannot, while low-rate continuation and preservation replay protect the accepted player.

KataGo b18 was used only offline to label fixed trajectories. Arena move selection remained Moka's 104,129-parameter checkpoint with the accepted 64-visit search.

### Corpus and training

Sixty-four deterministic accepted-Moka-versus-b6 games from opening offset 2,170,000 were analyzed at 32 native b18 visits per position. The corpus contained 4,224 positions. Policies averaged 7.6 supported moves and 0.484 top-move probability. Its SHA-256 was `18d6d2d3e3ae1789f87a9ba82c3ef3e731965c1a8e9fb7922279d49a0feaad35`.

Bounded policy-surprise weights had mean 1.0, 95th percentile 1.66, and maximum 3.38. The weighted archive SHA-256 was `e7ae3ec983478ab1e161483ae87c2e0561bf988d6d9792fa43367eea72cfe9c1`.

Seeds 101, 102, and 103 used one full-network epoch at learning rate 0.000002, incumbent-logit preservation weight 0.25, fourfold training-only b18 replay, the first accepted-search corpus, and the 20,000-position browser on-policy replay set.

Seed 101 improved b18 test loss from 2.5665 to 2.5317 and value MAE from 0.3184 to 0.3043. It reduced accepted-search agreement from 68.3% to 66.7%.

### Full-candidate gate

On 20 fresh games from opening offset 2,180,000, the incumbent scored eight wins. Seeds 101, 102, and 103 scored eleven, eight, and ten, all with zero caps. Seed 101 was frozen.

| Opening offset | Incumbent wins | Incumbent caps | Seed 101 wins | Seed 101 caps |
| -------------: | -------------: | -------------: | ------------: | ------------: |
|      2,190,000 |             38 |              2 |            40 |             2 |
|      2,200,000 |             43 |              0 |            37 |             0 |
|      **Total** |         **81** |          **2** |        **77** |         **2** |

The full stronger-teacher update was rejected.

### Conservative float interpolation

Fixed 25%, 50%, and 75% seed-101 blends were screened on fresh offset 2,210,000. The incumbent and blends scored four, four, ten, and seven wins. The 75% blend added a cap; the 50% blend was frozen as the unique completed-win leader.

| Opening offset | Incumbent wins | Incumbent caps | 50% blend wins | 50% blend caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,220,000 |             46 |              0 |             48 |              0 |
|      2,230,000 |             44 |              0 |             49 |              0 |
|      **Total** |         **90** |          **0** |         **97** |          **0** |

The float blend improved both blocks by seven wins in aggregate. Its architecture and 111,920-byte INT8 size were unchanged.

### Exact INT8 rejection

Ordinary INT8 export preserved offline b18 loss but did not preserve arena strength:

| Opening offset | Current INT8 | Candidate INT8 | Current caps | Candidate caps |
| -------------: | -----------: | -------------: | -----------: | -------------: |
|      2,240,000 |           49 |             43 |            0 |              0 |
|      2,250,000 |           43 |             33 |            0 |              0 |
|      **Total** |       **92** |         **76** |        **0** |          **0** |

The ordinary candidate was rejected as quantization-fragile.

A deployment-aware salvage anchored 25%, 50%, and 75% stronger-teacher blends to the exact dequantized current INT8 artifact before exporting. On a 20-game exact-INT8 screen at offset 2,260,000, the current artifact scored nine wins; 25% and 50% scored ten without caps, while 75% scored ten with one cap-awarded win. The conservative 25% blend was frozen.

| Opening offset | Current INT8 | Anchored 25% | Current caps | Candidate caps |
| -------------: | -----------: | -----------: | -----------: | -------------: |
|      2,270,000 |           36 |           38 |            1 |              1 |
|      2,280,000 |           42 |           39 |            0 |              0 |
|      **Total** |       **78** |       **77** |        **1** |          **1** |

The deployment-aware blend improved one block and regressed the other, losing one game in aggregate. It is rejected.

The stronger teacher contains useful float signal, but neither ordinary nor deployment-aware INT8 export cleared the exact-artifact gate. The accepted Moka checkpoint and artifact remain unchanged. No website files were touched.

## 2026-07-28 — Exact-INT8 quantization-aware continuation

### Hypothesis

The stronger b18 teacher improved a float interpolation from 90 to 97 wins, but ordinary INT8 export reduced it to 76 wins against the exact incumbent's 92. Training through the same per-output-channel INT8 rounding used by export might preserve useful teacher signal in the deployable artifact.

An opt-in quantization-aware path now keeps float shadow weights for optimization while every training, validation, and test forward pass uses straight-through symmetric INT8 weights. Biases remain float32, matching export. Regression tests verify the quantization values and identity gradient. Direct comparison of an exported candidate with its dequantized evaluation checkpoint found zero parameter error.

Ordinary training is unchanged.

### Stronger-teacher QAT

Seeds 104, 105, and 106 started from the exact dequantized accepted artifact. They used one epoch at learning rate 0.000002, policy-preservation weight 0.25, fourfold stronger-teacher replay, the accepted-search corpus, and the 20,000-position browser replay set.

Their b18 test losses were 2.5600, 2.5590, and 2.5603, compared with 2.5750 for the accepted INT8 artifact. Value MAE improved from 0.3197 to 0.3121, 0.3114, and 0.3125.

On 20 fresh exact-INT8 games from opening offset 2,290,000:

| Artifact  | Wins | Caps | Resignations |
| --------- | ---: | ---: | -----------: |
| Incumbent |    9 |    0 |            1 |
| Seed 104  |    7 |    0 |            1 |
| Seed 105  |    8 |    0 |            0 |
| Seed 106  |    7 |    0 |            2 |

The teacher-heavy branch was rejected.

### Lower teacher mixture

Seeds 107, 108, and 109 removed the three extra copies of the b18 corpus while retaining accepted-search and browser replay. Their b18 test losses were 2.5658, 2.5662, and 2.5661.

On 20 fresh exact-INT8 games from opening offset 2,300,000:

| Artifact  | Wins | Caps | Black | White |
| --------- | ---: | ---: | ----: | ----: |
| Incumbent |    9 |    0 |     6 |     3 |
| Seed 107  |    9 |    0 |     5 |     4 |
| Seed 108  |    8 |    0 |     5 |     3 |
| Seed 109  |    7 |    0 |     5 |     2 |

The best lower-mixture seed tied rather than improved, so none advanced.

### Search-target QAT

Seeds 110, 111, and 112 replaced the b18 primary corpus with the fresh second-round 64-visit search corpus. The first-round search corpus and browser corpus supplied preservation replay. One epoch used learning rate 0.000005 and policy-preservation weight 0.25.

Their second-round test losses were 2.0352, 2.0335, and 2.0338, with top-move agreements of 71.2%, 71.0%, and 70.6%.

On 20 fresh exact-INT8 games from opening offset 2,310,000:

| Artifact  | Wins | Caps | Resignations |
| --------- | ---: | ---: | -----------: |
| Incumbent |    8 |    0 |            1 |
| Seed 110  |    4 |    0 |            0 |
| Seed 111  |    7 |    0 |            0 |
| Seed 112  |    7 |    0 |            0 |

All candidates remained 111,920 bytes. Quantization-aware optimization fixed the float-to-INT8 mismatch as intended, but lower teacher loss and higher search-target agreement still failed to predict game strength. Every candidate is rejected. The accepted checkpoint and artifact remain unchanged, and no website files were touched.

## 2026-07-28 — Transposition-aware search values

### Hypothesis

The evaluator already caches feature-identical positions, but separate tree nodes do not share the value evidence accumulated after reaching the same state through different move orders. Reusing that evidence might spend the accepted 64-visit budget more effectively without adding model evaluations, model bytes, or teacher access during play.

Across eight fresh games, 449 of 15,859 descendant evaluation requests hit an existing state cache entry, a 2.83% reuse rate. This was sufficient to test a bounded shared-value path.

The experimental implementation kept visits and priors local to each parent edge. Only the mean-value evidence was shared by the exact Moka feature-state key: board, next color, ko point, and two recent moves. This prevented another parent's visits from contaminating root move counts. A regression test verified that identical positions shared value evidence while retaining independent edge visits and priors.

### Screen

The exact accepted INT8 artifact, 64 visits, root symmetry, geometric policy blend, exploration, FPU, value weight, full branching, and resignation behavior were fixed. On 20 fresh games from opening offset 2,330,000:

| Shared-value weight | Wins | Caps | Black | White |
| ------------------: | ---: | ---: | ----: | ----: |
|                0.00 |    7 |    1 |     2 |     5 |
|                0.25 |    7 |    1 |     2 |     5 |
|                0.50 |    7 |    1 |     2 |     5 |
|                1.00 |    8 |    1 |     2 |     6 |

Full sharing was frozen as the only screen improvement.

### Independent confirmation

Two untouched 100-game blocks compared the frozen candidate with the control on identical openings and colors:

| Opening offset | Control wins | Control caps | Shared-value wins | Shared-value caps |
| -------------: | -----------: | -----------: | ----------------: | ----------------: |
|      2,340,000 |           41 |            0 |                41 |                 0 |
|      2,350,000 |           46 |            0 |                47 |                 0 |
|      **Total** |       **87** |        **0** |            **88** |             **0** |

Control runtime was 574.5 seconds in aggregate. Shared-value runtime was 624.1 seconds, an 8.6% increase.

The candidate tied one block and added only one game in the other. That effect is too small to establish improved strength and does not justify the runtime cost. Transposition sharing is rejected, and its runtime path was removed. The accepted player, checkpoint, and INT8 artifact are unchanged. No website files were touched.

## 2026-07-28 — Search-disagreement preference optimization

### Hypothesis

Dense and list-wise search distillation move many logits even when the accepted model already agrees with search. A reference-anchored pairwise objective based on [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) can instead update only positions where 64-visit search selects a different top move. The preferred-versus-rejected logit margin is measured relative to the exact incumbent, limiting unnecessary policy drift.

The experimental loss ignored agreement rows. On disagreement rows, the search winner was preferred and the exact incumbent's top move was rejected. A regression test verified that an unchanged policy produced the expected log-two loss and that increasing only the incumbent-relative preferred margin reduced it.

### Exact-INT8 experiment

Seeds 113, 114, and 115 started from the exact dequantized accepted artifact. They used one quantization-aware epoch at learning rate 0.000002, preference beta 1.0, preference weight 1.0, policy-preservation weight 0.25, the second accepted-search corpus, the first accepted-search corpus as replay, and the 20,000-position browser replay set.

Their second-round search test losses were 2.0565, 2.0538, and 2.0541. Top-move agreements were 70.4%, 70.4%, and 70.1%. Every export remained exactly 111,920 bytes.

On 20 fresh exact-INT8 games from opening offset 2,360,000:

| Artifact  | Wins | Caps | Black | White |
| --------- | ---: | ---: | ----: | ----: |
| Incumbent |    8 |    0 |     3 |     5 |
| Seed 113  |    7 |    0 |     4 |     3 |
| Seed 114  |    8 |    0 |     4 |     4 |
| Seed 115  |    7 |    0 |     3 |     4 |

No seed improved wins, so none advanced. The preference-training branch was removed after rejection. The accepted player, checkpoint, and INT8 artifact are unchanged. No website files were touched.

## 2026-07-28 — Stronger-teacher root-only evaluator

### Hypothesis

The stronger b18 candidate contained useful float policy signal but failed when used for every descendant policy and value. Supplying its policy only at real move roots while retaining the accepted artifact for the complete descendant tree could isolate that signal. Both evaluators remained local Moka models; KataGo was never queried for Moka's moves during play.

The research arena temporarily accepted a separate root checkpoint. Root symmetry, geometric aggregation, the 64-visit budget, and every other accepted search setting were unchanged.

On 20 fresh exact-INT8 games from opening offset 2,370,000:

| Root evaluator        | Wins | Caps | Black | White |
| --------------------- | ---: | ---: | ----: | ----: |
| Accepted artifact     |   11 |    0 |     5 |     6 |
| Ordinary b18 blend    |    9 |    0 |     4 |     5 |
| INT8-anchored 25% b18 |   11 |    0 |     5 |     6 |

The ordinary stronger-teacher root lost two games. The conservative root reproduced the control rather than improving it. Neither advanced, and the alternate-root runtime path was removed. The accepted player, checkpoint, and INT8 artifact are unchanged. No website files were touched.

## 2026-07-28 — Selective high-visit b18 reanalysis

### Hypothesis

The earlier stronger-teacher corpus spent 32 native b18 visits on every reached position. KataGo's training loop instead concentrates full search on a subset of positions. Spending approximately the same teacher budget on fewer, harder positions might produce more useful targets without increasing offline teacher compute.

Sixty-four deterministic accepted-Moka-versus-b6 games from opening offset 2,380,000 supplied the trajectories. One quarter of each game's turns were selected: half uniformly and half by b6 policy KL plus 0.25 times value disagreement. Native b18 then analyzed only those turns at 128 visits.

The corpus contained 1,063 positions and had SHA-256 `8ff4f6bb08ada9168d541bbacfcb182fb3e37bcf241de3a847c0c0ea322e0bb8`. Its nominal teacher budget was 136,064 visits, compared with 135,168 visits for the earlier 4,224-position, 32-visit corpus. Policies averaged 11.88 supported moves, 0.460 top-move probability, and 1.632 nats of entropy. The earlier labels averaged 7.61 supported moves, 0.484 top-move probability, and 1.435 nats.

Seeds 116, 117, and 118 started from the exact dequantized accepted INT8 artifact. They used one quantization-aware epoch at learning rate 0.000002, policy-preservation weight 0.25, fourfold selective-corpus replay, and both accepted 64-visit search corpora as preservation replay.

On the untouched selective-corpus test split:

| Artifact  |   Loss | Top move | Value MAE |
| --------- | -----: | -------: | --------: |
| Incumbent | 3.1464 |    25.0% |    0.4131 |
| Seed 116  | 3.1764 |    25.0% |    0.4439 |
| Seed 117  | 3.1726 |    25.0% |    0.4414 |
| Seed 118  | 3.1726 |    25.0% |    0.4413 |

Every exact INT8 candidate worsened both its primary loss and value error. The branch failed before arena screening and was rejected. Concentrating the same teacher budget produced richer labels but not a trainable improvement under the current compact architecture and deployment-aware objective.

## 2026-07-28 — KataGo-style root lower confidence bound

### Hypothesis

[KataGo's play selection](https://github.com/lightvector/KataGo/blob/master/cpp/search/searchresults.cpp) can favor a sufficiently visited move whose estimated utility has the best lower confidence bound. Moka ordinarily plays the most-visited root child. Tracking the second moment of backed-up values and applying KataGo's bounded virtual-visit bonus might make better use of the same 64 evaluations.

The experimental path tracked each node's value-square sum. Root moves with at least 15% of the leading visit count received a variance-prior lower confidence bound. The best bound received KataGo's bounded selection-weight bonus. The model, exact INT8 artifact, 64-visit budget, root symmetry, exploration, FPU, value weight, branching, and resignation rule were fixed.

On 20 fresh games from opening offset 2,390,000:

| LCB standard deviations | Wins | Black | White | Caps |
| ----------------------: | ---: | ----: | ----: | ---: |
|                     0.0 |    7 |     4 |     3 |    0 |
|                     0.5 |    7 |     5 |     2 |    0 |
|                     1.0 |    7 |     4 |     3 |    0 |
|                     2.0 |    9 |     7 |     2 |    0 |
|                     4.0 |   11 |     6 |     5 |    0 |

Four standard deviations was frozen as the unique completed-win leader.

| Opening offset | Control wins | LCB wins | Control caps | LCB caps |
| -------------: | -----------: | -------: | -----------: | -------: |
|      2,400,000 |           42 |       39 |            0 |        0 |
|      2,410,000 |           41 |       43 |            0 |        0 |
|      **Total** |       **83** |   **82** |        **0** |    **0** |

The candidate regressed by three games on the first block and improved by two on the second, losing one game in aggregate. Control runtime was 584.6 seconds; LCB runtime was 607.1 seconds, about 3.8% slower. The screen gain did not reproduce, so the complete LCB runtime path and its second-moment storage were removed.

Both experiments leave the accepted checkpoint, exact 111,920-byte INT8 artifact, and accepted search unchanged. No Million website files were touched.

## 2026-07-28 — Per-channel INT8 scale optimization

### Hypothesis

The accepted float checkpoint was one game stronger than its ordinary INT8 export in their original 20-game sanity block. Choosing a lower clipping scale per output channel might reduce quantization error and recover strength without changing tensor shapes, parameter count, or the 111,920-byte artifact size.

Fixed clipping fractions of 0.99, 0.97, 0.95, and 0.90 were compared with a per-channel MSE search over 61 fractions from 0.70 through 1.00. The MSE search clipped 27% of output channels, selected a mean fraction of 0.9986 and a minimum of 0.99, and reduced relative weight RMSE to 0.01942. On the two accepted-search test splits:

| INT8 scales | Float top move | Logit RMSE | Value RMSE | First loss | Second loss |
| ----------- | -------------: | ---------: | ---------: | ---------: | ----------: |
| Ordinary    |          93.8% |    0.14243 |    0.03728 |     1.8519 |      2.0789 |
| MSE grid    |          94.1% |    0.14326 |    0.01586 |     1.8566 |      2.0843 |
| Clip 0.99   |          93.8% |    0.14594 |    0.03109 |     1.8611 |      2.0936 |
| Clip 0.97   |          92.7% |    0.25015 |    0.05157 |     1.8528 |      2.0730 |

More aggressive fixed clipping materially worsened float-policy fidelity and was excluded before arena screening.

On 20 fresh exact-dequantized games from opening offset 2,420,000:

| INT8 scales | Wins | Black | White | Caps |
| ----------- | ---: | ----: | ----: | ---: |
| Ordinary    |    9 |     5 |     4 |    0 |
| MSE grid    |    8 |     5 |     3 |    0 |
| Clip 0.97   |    9 |     5 |     4 |    0 |
| Clip 0.99   |    7 |     3 |     4 |    0 |

The candidate with the best reconstruction metrics lost one game. The best fixed candidate only tied the ordinary export. No scale rule advanced, so the exporter and accepted artifact remain unchanged.

## 2026-07-28 — Reused-root symmetry value refresh

### Hypothesis

When a retained subtree becomes the real root, Moka refreshes its priors with the eight-way root evaluator but ignores the simultaneously computed eight-way value. Unvisited root children therefore use FPU based on the older single-orientation subtree mean. Blending the already available symmetry value into the reused root might improve allocation without another evaluation.

The experimental path blended 0%, 25%, 50%, 75%, or 100% of the refreshed symmetry value into the reused root mean. It changed neither the model, artifact, inference count, visit budget, nor any descendant search setting.

On 20 fresh games from opening offset 2,430,000, every blend scored exactly five wins: four as Black and one as White, with zero caps. Runtime ranged from 56.2 to 57.4 seconds. Pass counts changed slightly, proving the path was active, but no setting changed a game result.

The complete value-refresh path was removed after the screen. Both experiments leave the accepted checkpoint, exact INT8 artifact, and accepted search unchanged. No Million website files were touched.

## 2026-07-28 — Root and descendant PUCT exploration split

### Hypothesis

Moka's real-move root uses an eight-way averaged policy, while every descendant uses a single-view policy. Both levels nevertheless share exploration coefficient 2.0. Tuning them separately might allocate the fixed 64 visits more effectively without changing the model, inference count, payload, or legal moves.

An opt-in root coefficient retained the existing coefficient at descendants. On 20 fresh games from opening offset 2,440,000:

| Root exploration | Wins | Black | White | Caps |
| ---------------: | ---: | ----: | ----: | ---: |
|             0.75 |   12 |     6 |     6 |    0 |
|             1.25 |   10 |     6 |     4 |    1 |
|             1.50 |   10 |     4 |     6 |    1 |
|             2.00 |   14 |     6 |     8 |    0 |
|             2.50 |   11 |     6 |     5 |    0 |

The accepted root coefficient 2.0 remained the clear joint leader.

The complementary screen fixed root exploration at 2.0 and varied only descendants. On 20 new games from opening offset 2,450,000:

| Descendant exploration | Wins | Black | White | Caps |
| ---------------------: | ---: | ----: | ----: | ---: |
|                   0.75 |   12 |     7 |     5 |    0 |
|                   1.25 |    9 |     5 |     4 |    0 |
|                   1.50 |   10 |     5 |     5 |    0 |
|                   2.00 |    7 |     4 |     3 |    0 |
|                   2.50 |   10 |     5 |     5 |    0 |

Descendant coefficient 0.75 was frozen as the unique five-win screen improvement.

| Opening offset | Control wins | Candidate wins | Control caps | Candidate caps |
| -------------: | -----------: | -------------: | -----------: | -------------: |
|      2,460,000 |           48 |             42 |            0 |              0 |
|      2,470,000 |           41 |             38 |            0 |              0 |
|      **Total** |       **89** |         **80** |        **0** |          **0** |

The screen gain reversed on both independent blocks. The frozen candidate lost nine games in aggregate. Its 559.2-second runtime was lower than the control's 579.0 seconds because its games ended sooner, not because it changed the fixed inference budget.

Separate root and descendant exploration is rejected, and the entire runtime path was removed. The accepted coefficient remains 2.0 at every node. The checkpoint, exact INT8 artifact, 64-visit budget, and Million website remain unchanged.

## 2026-07-28 — Accepted-search b18 distillation

### Method correction

The native b18 search generator labeled positions from games attributed to Moka, but Moka's rollout moves came directly from its raw policy. The deployed player instead uses 64-visit tree search with an eight-symmetry root evaluation. The resulting training distribution did not represent the states reached by the accepted player.

The generator now supports the accepted search player during rollouts and can restrict b18 analysis to Moka's own decision turns. The rollout sessions preserve subtrees independently for every concurrent game and apply the accepted pass-resignation rule. Raw-policy rollout remains the default so existing experiments stay reproducible.

The corrected collection used:

- 64 deterministic games at opening offset 2,481,000
- 64 Moka simulations per move
- alternating Moka colors
- b18 analysis only on Moka decision turns
- four b18 visits per analyzed position
- 2,116 parent positions and 19,663 analyzed child states
- 866,925 compressed bytes
- SHA-256 `0657c38d2b74b35320d35e5d2e4cca48f847ecece7b9881a74bb33e73ed5653f`

The average b18 policy support was 9.264 moves, mean top probability was 0.47637, mean entropy was 1.5128, and mean absolute value was 0.9078. The held-out split contained 216 positions. The accepted exact INT8 model scored 3.1451 loss, 35.6% top-move agreement, and 0.6016 value MAE on it.

### Quantization-aware adaptation

Three one-epoch candidates started from the accepted exact-dequantized checkpoint. Each used learning rate 0.000002, batch size 128, INT8 quantization-aware training, policy preservation weight 0.25, the corrected b18 corpus twice, and both accepted 64-visit search corpora once.

| Seed | Validation loss | Validation move | Validation value MAE | Test loss | Test move | Test value MAE |
| ---: | --------------: | --------------: | -------------------: | --------: | --------: | -------------: |
|  119 |          2.7023 |           41.1% |               0.3910 |    3.1030 |     35.6% |         0.5504 |
|  120 |          2.6995 |           41.1% |               0.3886 |    3.1004 |     36.6% |         0.5477 |
|  121 |          2.6975 |           41.1% |               0.3862 |    3.0988 |     36.6% |         0.5453 |

Every offline metric favored seed 121. All candidates exported to exactly 111,920 bytes. Their binary SHA-256 values were:

- seed 119: `672691c38991bb02f9132d4f9f0bbe147167f02aa9a923054f5fff40be485ebc`
- seed 120: `b9e625af3e1a143ccf42698f228c589a55a034a59f68303e9f266c2038f739ab`
- seed 121: `22b51188fc0c3eb59dc0fc9d1df929069c312dcbcf40176b578708ebb2cfe526`

On 20 exact-dequantized games at opening offset 2,490,000, the incumbent and seed 119 each won six games. Seeds 120 and 121 each won seven. Seed 121 advanced because it tied the arena lead and had the best offline metrics.

| Opening offset | Incumbent wins | Seed 121 wins | Incumbent caps | Seed 121 caps |
| -------------: | -------------: | ------------: | -------------: | ------------: |
|      2,500,000 |             35 |            29 |              1 |             0 |
|      2,510,000 |             42 |            43 |              0 |             0 |
|      **Total** |         **77** |        **72** |          **1** |         **0** |

The offline improvement did not improve completed wins. Seed 121 lost five games in aggregate and was rejected.

### Conservative checkpoint blending

Because the learned direction improved every held-out metric but moved behavior too far, exact INT8 candidates blended 25%, 50%, or 75% of seed 121 into the incumbent. On 20 new games at opening offset 2,520,000:

| Candidate | Wins | Black | White | Caps |
| --------- | ---: | ----: | ----: | ---: |
| Incumbent |    6 |     2 |     4 |    0 |
| 25% blend |    8 |     3 |     5 |    0 |
| 50% blend |    7 |     4 |     3 |    0 |
| 75% blend |    6 |     2 |     4 |    0 |

The unique screen winner was frozen before confirmation.

| Opening offset | Incumbent wins | 25% blend wins | Incumbent caps | 25% blend caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,530,000 |             38 |             39 |              0 |              0 |
|      2,540,000 |             44 |             37 |              0 |              0 |
|      **Total** |         **82** |         **76** |          **0** |          **0** |

The blend improved by one game on the first block and regressed by seven on the second. It lost six games in aggregate and was rejected. The accepted checkpoint, exact 111,920-byte INT8 artifact, and runtime search remain unchanged. No Million website files were touched.

## 2026-07-28 — Policy-isolated b18 correction

### Hypothesis

The accepted-search b18 continuation improved held-out policy and value metrics but lost five games in confirmation. Updating the trunk and value head may have damaged the existing search evaluator. Restricting training to the policy head can import move-choice signal while preserving every trunk and value tensor exactly.

Three one-epoch candidates started from the accepted exact-dequantized artifact. All used the corrected b18 corpus, both accepted 64-visit search corpora as replay, batch size 128, and policy-preservation weight 0.25:

| Candidate                    | Trainable tensors | Learning rate | Test loss | Test move | Test value MAE |
| ---------------------------- | ----------------: | ------------: | --------: | --------: | -------------: |
| Linear only                  |                 2 |      0.000010 |    3.1450 |     36.1% |         0.6016 |
| Complete policy head         |                 4 |      0.000005 |    3.1484 |     36.6% |         0.6016 |
| Policy head, stronger anchor |                 4 |      0.000010 |    3.1531 |     37.0% |         0.6016 |

Tensor comparison verified that the linear candidate changed only `policy_linear.weight` and `policy_linear.bias`. Every trunk and value parameter remained bit-identical. Each exact INT8 export remained 111,920 bytes.

On 20 fresh games from opening offset 2,550,000, the incumbent scored seven wins. The linear and low-rate complete-head candidates scored eight, while the stronger-anchor candidate scored six. All had zero caps. The linear candidate advanced because it changed fewer tensors and had the better held-out loss among the screen leaders.

| Opening offset | Incumbent wins | Linear wins | Incumbent caps | Linear caps |
| -------------: | -------------: | ----------: | -------------: | ----------: |
|      2,560,000 |             37 |          38 |              0 |           0 |
|      2,570,000 |             40 |          39 |              0 |           0 |
|      **Total** |         **77** |      **77** |          **0** |       **0** |

The candidate moved one win between blocks but did not improve the aggregate. It was rejected.

### Raw-policy disagreement weighting

The accepted raw policy disagreed with the b18 top move on 1,220 of 2,116 roots. Those rows received eight times the weight of agreement rows, normalized to mean one. Linear candidates at learning rates 0.000005 and 0.000010 and a complete-head candidate at 0.000005 were screened on opening offset 2,580,000.

The incumbent and all three candidates scored exactly nine wins with zero caps. Both linear candidates reproduced every aggregate counter exactly. Raw-policy disagreement was therefore too broad and mostly inert.

## 2026-07-28 — Actual-search rollout regret

### Hypothesis

Raw-policy disagreement does not identify mistakes made by the accepted 64-visit player. Recording Moka's actual searched rollout move at every b18-labeled root allows teacher action values to estimate the consequence of the move Moka really played.

The search generator now stores `rollout_moves` aligned with the existing root features, policies, and child values. A new rollout-regret weighting mode compares the b18 value of its top-visit move with the b18 value of Moka's searched move. Missing searched-move values and non-regrets retain the minimum weight.

The identical 64 games from opening offset 2,481,000 were regenerated. Native analysis responses arrived in a different order, but all 2,116 `(game ID, feature state)` keys exactly matched the earlier corpus. The new archive is 871,068 bytes with SHA-256 `50e8ffc9ebc1cc29025230addadee391c34ad8627c082b319941fe1c9fd1fbec`.

B18 evaluated Moka's selected move on 90.1% of roots. Moka selected the b18 top move on 45.7%, but most disagreements were value-equivalent:

| Minimum b18 regret | Fraction of all roots |
| -----------------: | --------------------: |
|               0.05 |                 7.37% |
|               0.10 |                 5.81% |
|               0.20 |                 4.06% |
|               0.40 |                 2.08% |
|               0.80 |                 0.71% |

Among covered moves, mean regret was 0.0306 and median regret was zero. The weighted archive had SHA-256 `fa0efe5383888daa2122746125ab388874b4f7d83e33bb5aa2a7911525788d7c`. Its mean weight was 0.344, with 86 rows at weight one or greater and 37 at weight two or greater.

Three linear-only candidates retained the accepted trunk and value tensors:

| Candidate | Epochs | Learning rate | Test loss | Test move | Test value MAE |
| --------- | -----: | ------------: | --------: | --------: | -------------: |
| Low rate  |      1 |      0.000010 |    3.1660 |     38.0% |         0.6015 |
| High rate |      1 |      0.000020 |    3.1676 |     39.4% |         0.6015 |
| Two epoch |      2 |      0.000010 |    3.1660 |     38.0% |         0.6015 |

The incumbent scored 3.1611 loss, 38.0% top-move agreement, and 0.6015 value MAE on the same test games. The low-rate and two-epoch exports quantized to identical bytes.

On 20 fresh exact-INT8 games from opening offset 2,590,000, the incumbent and both unique candidates scored eight wins, with three Black wins, five White wins, and zero caps. The candidates changed pass counts but not game outcomes. They were rejected without confirmation.

This experiment shows that top-move disagreement greatly overstates actual search mistakes: only about one in 25 reached roots has b18 regret of at least 0.2. The rollout-move and regret instrumentation remains because it enables a larger targeted corpus without changing ordinary collection or training defaults. The accepted checkpoint, exact 111,920-byte INT8 artifact, runtime search, and Million website remain unchanged.

## 2026-07-28 — Selective exact-artifact regret correction

### Proxy gate

A larger regret corpus is useful only if expensive b18 analysis can be concentrated without hiding mistakes the proxy cannot recognize. The existing 2,116-root corpus was replayed with b6 action regret for Moka's actual searched move. Selecting the highest-regret 12.5% within each game recovered:

| Minimum b18 regret | Positive roots | Proxy recall | Proxy precision |
| -----------------: | -------------: | -----------: | --------------: |
|               0.05 |            156 |        28.8% |           16.9% |
|               0.10 |            123 |        32.5% |           15.0% |
|               0.20 |             86 |        31.4% |           10.1% |
|               0.40 |             44 |        29.5% |            4.9% |

The continuous b6/b18 regret correlation was 0.251. At the material 0.2 threshold, the proxy recovered about 2.5 times as many mistakes as uniform 12.5% selection. The scaled collector therefore fixed its b18 budget at 25% of Moka turns: half proxy-ranked and half uniformly sampled.

### Exact deployment corpus

The earlier corrected corpus used the accepted float checkpoint for rollout moves. Production uses the exact INT8 artifact. The scaled run removed that mismatch:

- exact-dequantized accepted INT8 checkpoint
- 256 deterministic Moka-versus-b6 games
- alternating colors
- 64 Moka search visits per move
- isolated opening offset 3,000,000
- Moka decision turns only
- 12.5% highest b6 action regret and 12.5% uniform turns per game
- 64 native b18 visits per selected root

The corpus contains 2,181 roots from all 256 games and is 844,414 compressed bytes. Its SHA-256 is `0a46d240aa9240f4652f0cad53cb7056715343295def0c1d10f7c9d38482359f`.

B18 evaluated Moka's selected move on 82.3% of roots. Moka selected the b18 top move on 36.8%. The selected set contained 153 roots at regret 0.2 or greater, 7.02% of all labels versus 4.06% in the unselected corpus. Mean covered regret doubled from 0.0306 to 0.0607.

|      Split | Roots | Regret ≥ 0.2 |
| ---------: | ----: | -----------: |
|      Train | 1,742 |          116 |
| Validation |   209 |           17 |
|       Test |   230 |           20 |

The continuous regret archive had SHA-256 `e398611d6760bc0487a01be461b6fb7a1ba4714ed36c79d8212d965b903eaaeb`.

### Policy-only quantization-aware training

Post-training export had repeatedly rounded small policy-only updates into ineffective INT8 changes. The QAT path now supports a frozen trunk: it initializes the complete quantized evaluator from the deployment model, updates only trainable quantized tensors during differentiation, and leaves every frozen tensor exact. Fully trainable QAT behavior is unchanged.

Three one-epoch linear-policy QAT candidates used continuous regret weights, learning rates 0.000010 or 0.000020, optional 25% hard-target mixing, policy preservation weight 0.25, and both accepted search corpora as replay. Exact test loss was 2.9009–2.9013 versus 2.9004 for the incumbent. All retained 33.5% top-move agreement.

On 20 fresh games from opening offset 2,610,000, the incumbent and all candidates scored six wins with zero caps. Continuous regret QAT was rejected.

### Critical regret targets

A critical mode keeps only roots where b18 regret is at least 0.2, uses the b18 top move as a hard target, weights ordinary material errors one, and weights winner-flipping errors four. The archive contains 153 material roots, including 51 winner flips. Training has 116 material roots and 37 winner flips. Its SHA-256 is `8045cab154d82ab63922539909a6c354561cfdacf38136dc88ce92994473fcac`.

Repeating a sparse archive eight times exposed an all-zero sample-weight batch. Sample-weight normalization now clamps its denominator, making such a batch contribute zero weighted task loss rather than NaN. A regression test verifies finite zero output.

Three exact policy-linear QAT candidates replayed the critical corpus four or eight times:

| Candidate    | Critical replay | Learning rate | Exact test loss | Test move |
| ------------ | --------------: | ------------: | --------------: | --------: |
| Conservative |              4× |      0.000020 |          2.3468 |     33.5% |
| Higher rate  |              4× |      0.000050 |          2.3484 |     33.5% |
| More replay  |              8× |      0.000020 |          2.3476 |     33.5% |
| Incumbent    |               — |             — |          2.3567 |     33.5% |

Every exported candidate remained exactly 111,920 bytes. Exact tensor comparison found changes only in `policy_linear.weight` and `policy_linear.bias`; the trunk and value estimator stayed bit-identical.

On 20 fresh games from opening offset 2,620,000:

| Artifact     | Wins | Black | White | Caps |
| ------------ | ---: | ----: | ----: | ---: |
| Incumbent    |    7 |     5 |     2 |    0 |
| 4×, 0.000020 |    8 |     6 |     2 |    0 |
| 4×, 0.000050 |   11 |     5 |     6 |    0 |
| 8×, 0.000020 |   10 |     6 |     4 |    0 |

The unique 11-win candidate was frozen.

### Confirmation and adjudication

The first two untouched blocks left the candidate ahead by only three games and split directionally, so one fixed third block was declared before promotion. Acceptance required the candidate to remain ahead across all 300 games without a cap regression.

| Opening offset | Incumbent wins | Candidate wins | Incumbent caps | Candidate caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,630,000 |             37 |             36 |              0 |              0 |
|      2,640,000 |             40 |             44 |              0 |              0 |
|      2,650,000 |             43 |             34 |              0 |              0 |
|      **Total** |        **120** |        **114** |          **0** |          **0** |

The screen and second-block gains did not generalize. The adjudication reversed the close lead, and the frozen candidate lost six games over 300. It is rejected.

Selective b6 regret successfully enriched consequential b18 mistakes, policy-only QAT preserved exact deployment constraints, and critical targets improved exact held-out loss. The frozen screen leader did not prove a strength gain, so production remained unchanged pending separate confirmation of the already-frozen conservative runner-up.

## 2026-07-28 — Conservative critical-QAT confirmation

The critical-QAT screen also produced a more conservative runner-up: eightfold critical replay at learning rate 0.000020 scored 10–7 against the incumbent. It was frozen in the original screen and retained without retraining, interpolation, or re-export.

Its exact INT8 binary has SHA-256 `5d38b3d3f88582212065e6d2aee7b5d638c13e3c4ccaf9e1cab1cd341f757714`. It is 111,920 bytes and differs from the previous artifact only in `policy_linear.weight` and `policy_linear.bias`.

Two fresh 100-game blocks tied in aggregate. The candidate removed both of the incumbent's losing move caps:

| Opening offset | Incumbent wins | Candidate wins | Incumbent caps | Candidate caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,660,000 |             34 |             35 |              1 |              0 |
|      2,670,000 |             42 |             41 |              1 |              0 |
|      **Total** |         **76** |         **76** |          **2** |          **0** |

The predeclared single adjudication block required a strict aggregate win advantage rather than cap reduction alone:

| Opening offset | Incumbent wins | Candidate wins | Incumbent caps | Candidate caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,680,000 |             37 |             41 |              0 |              0 |
|  **300 total** |        **113** |        **117** |          **2** |          **0** |

The candidate finishes four completed wins ahead across 300 paired games and removes both unfinished losses. It passes the exact-artifact promotion gate.

The accepted float shadow checkpoint now has SHA-256 `1ee7771f0cc68e41b22e6a05a2384338ec4d3a3763714244a6103d171abecc0a`. Its exact-dequantized checkpoint has SHA-256 `040fae5ff09762db03913e70fae1f5260d4746f5c390c2e36382b3d46468ec1e`. The accepted browser artifact has SHA-256 `5d38b3d3f88582212065e6d2aee7b5d638c13e3c4ccaf9e1cab1cd341f757714`.

The architecture remains 104,129 parameters, the browser artifact remains 111,920 bytes, and the accepted runtime search remains 64 visits. Only the Moka repository artifact is updated. The Million website remains untouched.

## 2026-07-28 — Second-round critical-QAT distillation

### Question

Can the promoted sparse critical-QAT method compound by collecting mistakes from the new accepted artifact's changed trajectories?

### New on-policy corpus

The exact accepted artifact generated a separate corpus using the unchanged successful collection recipe:

- 256 Moka-versus-b6 games from opening offset 3,100,000
- alternating colors and deterministic 64-visit Moka search
- 12.5% highest b6 action-regret turns and 12.5% uniform turns
- Moka decision turns only
- 64 native b18 visits per selected root

The corpus contains 2,154 roots and is 822,012 compressed bytes. Its SHA-256 is `728bd8b9ecb68751aa6bf4060360109a68fe26b8024fa1160c638408fa085175`.

B18 evaluated Moka's actual move on 84.3% of roots. Moka selected the b18 top move on 38.3%. The selected set remained enriched:

| Minimum b18 regret | Roots | Fraction |
| -----------------: | ----: | -------: |
|               0.05 |   272 |   12.63% |
|               0.10 |   226 |   10.49% |
|               0.20 |   166 |    7.71% |
|               0.40 |    91 |    4.22% |
|               0.80 |    31 |    1.44% |

The critical archive has SHA-256 `218b4128b338dcc0ac911f3cf486775182eb9f5ad33d9e6bc012840065d4406f`. It contains 134 material training roots, including 47 winner flips; validation contains 20 material roots and test contains 12.

### Pure round-two replay

Seeds 161, 162, and 163 used the accepted eightfold critical-QAT recipe unchanged: one epoch, learning rate 0.000020, policy-linear tensors only, exact INT8-aware forward passes, policy preservation weight 0.25, and both accepted search corpora as replay.

Every exact export remained 111,920 bytes. Critical test loss changed from 2.2689 for the incumbent to 2.2679–2.2681. Seed 162 preserved the incumbent's 39.6% top-move agreement; the other seeds scored 39.2%.

On 20 fresh games from opening offset 2,690,000, the incumbent scored ten wins. All three seeds scored nine, with six Black wins, three White wins, and zero caps. Pure second-round replay was rejected without confirmation.

### Balanced old/new replay

A bounded follow-up kept the same eightfold critical budget but split it evenly: four copies of the first accepted critical corpus and four copies of the new corpus. Seeds 167, 168, and 169 retained the same optimizer, learning rate, preservation, exact-QAT, and search replay settings.

On 20 fresh games from opening offset 2,700,000, the incumbent and all three candidates scored exactly seven wins, with four Black wins, three White wins, and zero caps. Aggregate pass and resignation counts changed, proving that the exports were active, but no game result improved.

Immediate repeated critical distillation does not compound the promoted gain, even when old mistakes are replayed to limit forgetting. All round-two candidates are rejected. The accepted checkpoint, exact 111,920-byte artifact with SHA-256 `5d38b3d3f88582212065e6d2aee7b5d638c13e3c4ccaf9e1cab1cd341f757714`, runtime search, and Million website remain unchanged.

## 2026-07-28 — Post-promotion geometric consensus

### Question

Does the accepted geometric root-policy blend remain optimal after the promoted policy-linear correction changed Moka's priors?

The exact accepted artifact, KataGo b6c96 opponent, 64-visit search, exploration 2.0, value weight 1.25, FPU 0.25, full branching, maximum-visit selection, and all other search settings were fixed. Only the blend between the arithmetic and normalized geometric means of Moka's existing eight root-symmetry policies changed.

### Screen

On 20 fresh games from opening offset 2,710,000:

| Geometric blend | Wins | Black | White | Caps |
| --------------: | ---: | ----: | ----: | ---: |
|           0.000 |    9 |     5 |     4 |    0 |
|           0.125 |   11 |     7 |     4 |    0 |
|           0.250 |   10 |     6 |     4 |    0 |
|           0.375 |    8 |     5 |     3 |    0 |
|           0.500 |    9 |     5 |     4 |    0 |

Weight 0.125 was frozen as the unique screen leader.

### Independent confirmation

Two disjoint 100-game blocks compared the frozen candidate with production:

| Opening offset | Production wins | Candidate wins | Production caps | Candidate caps |
| -------------: | --------------: | -------------: | --------------: | -------------: |
|      2,720,000 |              41 |             44 |               0 |              0 |
|      2,730,000 |              37 |             37 |               0 |              0 |
|      **Total** |          **78** |         **81** |           **0** |          **0** |

Production split 38 Black and 40 White wins. The candidate split 41 Black and 40 White wins. Every game completed normally. Candidate runtimes were 282.9 and 279.3 seconds; production runtimes were 283.6 and 280.6 seconds.

The candidate adds three completed wins without a cap or runtime regression. Geometric root-policy weight 0.125 is accepted as Moka's search default. Model bytes, parameter count, teacher access during play, and the Million website remain unchanged.

## 2026-07-28 — Post-consensus value and exploration retune

### Value weight

The accepted 0.125 geometric consensus changed Moka's root priors, so the policy–value balance was screened again. The exact accepted artifact, KataGo b6c96 opponent, 64 visits, exploration 2.0, FPU 0.25, full branching, and maximum-visit selection were fixed.

On 20 fresh games from opening offset 2,740,000:

| Value weight | Wins | Black | White | Caps |
| -----------: | ---: | ----: | ----: | ---: |
|        1.000 |   10 |     5 |     5 |    0 |
|        1.125 |   10 |     5 |     5 |    0 |
|        1.250 |    7 |     3 |     4 |    0 |
|        1.375 |   10 |     4 |     6 |    0 |
|        1.500 |    6 |     4 |     2 |    0 |

Three alternatives tied. Selecting among them after observing the screen would add selection bias, so none advanced. Production retains value weight 1.25.

### Exploration screen

With value weight 1.25 fixed, a preplanned exploration screen used 20 fresh games from opening offset 2,750,000:

| Exploration | Wins | Black | White | Caps |
| ----------: | ---: | ----: | ----: | ---: |
|        1.50 |    9 |     5 |     4 |    0 |
|        1.75 |    9 |     3 |     6 |    0 |
|        2.00 |   10 |     4 |     6 |    0 |
|        2.25 |   10 |     5 |     5 |    0 |
|        2.50 |   13 |     6 |     7 |    0 |

Exploration 2.5 was frozen as the unique screen leader.

### Exploration confirmation

Two disjoint 100-game blocks compared the frozen candidate with production:

| Opening offset | Production wins | Candidate wins | Production caps | Candidate caps |
| -------------: | --------------: | -------------: | --------------: | -------------: |
|      2,760,000 |              38 |             36 |               1 |              0 |
|      2,770,000 |              43 |             37 |               0 |              0 |
|      **Total** |          **81** |         **73** |           **1** |          **0** |

Production's cap was awarded to Moka by the final area score. Excluding that award leaves 80 normally completed production wins, still seven more than the candidate. The 2.5 candidate regressed on both blocks and is rejected.

Moka retains exploration 2.0 and value weight 1.25. The accepted 0.125 geometric blend, model bytes, and Million website remain unchanged.

## 2026-07-28 — Post-consensus FPU and temperature retune

### First-play urgency

The exact accepted artifact, 64 visits, geometric consensus 0.125, exploration 2.0, value weight 1.25, full branching, and maximum-visit selection were fixed. On 20 fresh games from opening offset 2,780,000:

| FPU reduction | Wins | Black | White | Caps |
| ------------: | ---: | ----: | ----: | ---: |
|          0.00 |    7 |     4 |     3 |    0 |
|          0.10 |    9 |     5 |     4 |    0 |
|          0.20 |    8 |     4 |     4 |    0 |
|          0.25 |    7 |     2 |     5 |    0 |
|          0.30 |    9 |     4 |     5 |    0 |
|          0.40 |    6 |     1 |     5 |    0 |

Reductions 0.10 and 0.30 tied. Neither advanced because selecting between tied siblings after observing the screen would add selection bias. Production retains FPU reduction 0.25.

### Root-policy temperature screen

A narrow calibration screen held every accepted setting fixed and varied only the symmetry-aggregated root-policy temperature. On 20 fresh games from opening offset 2,790,000:

| Temperature | Wins | Black | White | Caps |
| ----------: | ---: | ----: | ----: | ---: |
|        0.85 |    4 |     3 |     1 |    0 |
|        0.90 |    4 |     2 |     2 |    0 |
|        0.95 |    7 |     4 |     3 |    0 |
|        1.00 |    7 |     4 |     3 |    0 |
|        1.05 |    9 |     5 |     4 |    0 |
|        1.10 |    7 |     4 |     3 |    0 |

Temperature 1.05 was frozen as the unique screen leader.

### Temperature confirmation

Two disjoint 100-game blocks compared the frozen candidate with production:

| Opening offset | Production wins | Candidate wins | Production caps | Candidate caps |
| -------------: | --------------: | -------------: | --------------: | -------------: |
|      2,800,000 |              33 |             27 |               0 |              0 |
|      2,810,000 |              39 |             37 |               0 |              1 |
|      **Total** |          **72** |         **64** |           **0** |          **1** |

The candidate regressed on both blocks, lost eight completed games, and introduced one losing move cap. Temperature 1.05 is rejected.

Moka retains root-policy temperature 1.0 and FPU reduction 0.25. The accepted search, model artifact, and Million website remain unchanged.

## 2026-07-28 — Equal-budget deep selective reanalysis

### Hypothesis

KataGo's training methods allocate expensive search to surprising positions rather than using the same shallow budget everywhere. The accepted selective-regret collector already chooses 25% of Moka turns, but its b18 labels used 64 visits. This experiment held total teacher visits approximately constant while trading four times fewer roots for four times deeper labels.

The exact accepted artifact generated 64 Moka-versus-b6 games from opening offset 3,200,000. Moka used its accepted 64-visit search. Only Moka turns were eligible, half of the selected turns were uniform, half were highest b6 action regret, and b18 analyzed every selected root at 256 visits.

The corpus contains 536 roots in 64 games and occupies 287,478 compressed bytes. Its SHA-256 is `fa448fcf90384557042511a72b734c5ee47259a56388147f33e95771c740a8c8`.

Compared with the 64-visit round-two corpus, deeper labels increased searched-action coverage from 84.3% to 89.4% and material-regret density:

| Minimum b18 regret | Roots | Fraction |
| -----------------: | ----: | -------: |
|               0.05 |    82 |   15.30% |
|               0.10 |    70 |   13.06% |
|               0.20 |    55 |   10.26% |
|               0.40 |    31 |    5.78% |
|               0.80 |    12 |    2.24% |

Moka selected b18's top move on 37.9% of roots. The critical archive has SHA-256 `61e142d0cba194df096d7216bdbd7a1988b19edf08d235d6c770037680e22a3f`. It contains 54 material roots across 32 games: 46 train, six validation, and two test roots.

### Conservative policy-linear QAT

Seeds 177, 178, and 179 used the accepted policy-only recipe: one epoch, learning rate 0.000020, eightfold critical replay, both accepted search corpora once, frozen trunk and policy convolution, INT8-aware forward passes, and policy preservation weight 0.25.

All exact exports remained 111,920 bytes. On 20 fresh games from opening offset 2,820,000, the incumbent and every candidate scored seven wins with five Black and two White wins and zero caps. No conservative candidate advanced.

### Stronger policy-linear QAT

A predeclared bounded follow-up increased only the learning rate to 0.000050. Seeds 181, 182, and 183 retained the same data, replay, preservation, frozen tensors, epoch count, and exact INT8 path.

The exact exports remained 111,920 bytes. Their SHA-256 values were:

| Seed | SHA-256                                                            |
| ---: | :----------------------------------------------------------------- |
|  181 | `9b2349b2fa9de83b8075a6aa6749ec09c4c6becf2aa279ea6801788590385614` |
|  182 | `db75be8dc188463beb78fd2b699aa86d8e63cadc1bb225838a985877b5e52e6f` |
|  183 | `4353b046175ccd7cae8136aa687e9d18d5b2a80e17cabd29f42c61c6cd942d86` |

On 20 fresh games from opening offset 2,830,000:

| Player    | Wins | Black | White | Caps |
| :-------- | ---: | ----: | ----: | ---: |
| Incumbent |   10 |     5 |     5 |    0 |
| Seed 181  |    9 |     4 |     5 |    0 |
| Seed 182  |   11 |     6 |     5 |    0 |
| Seed 183  |    6 |     4 |     2 |    0 |

Seed 182 was frozen as the unique leader.

### Standalone confirmation

Two fresh 100-game blocks compared frozen seed 182 with the incumbent:

| Opening offset | Incumbent wins | Candidate wins | Incumbent caps | Candidate caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,840,000 |             41 |             43 |              0 |              1 |
|      2,850,000 |             44 |             48 |              0 |              0 |
|      **Total** |         **85** |         **91** |          **0** |          **1** |

The candidate improved both blocks and added six completed wins, but introduced one losing cap. The trace showed Moka repeatedly selecting pass while behind 55 points, just below the accepted 60-point resignation threshold. Seed 182 failed the predeclared no-cap-regression gate and was not promoted.

### Separate resignation composite

Prior evidence showed that resignation margins 40 and 60 preserved outcomes while the lower threshold ended one additional hopeless game. A new composite candidate paired frozen seed 182 with the minimal 55-point margin.

On 20 fresh games from opening offset 2,860,000, the incumbent scored seven wins. Seed 182 scored eight at both margins 60 and 55, with identical colors, passes, resignations, and zero caps. Margin 55 therefore did not manufacture the screen gain.

The frozen composite then played two new 100-game blocks:

| Opening offset | Incumbent wins | Composite wins | Incumbent caps | Composite caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      2,870,000 |             44 |             42 |              0 |              1 |
|      2,880,000 |             46 |             45 |              0 |              0 |
|      **Total** |         **90** |         **87** |          **0** |          **1** |

The composite regressed on both fresh blocks and produced a different losing repetition cap. Seed 182, margin 55, and the deep-critical training branch are rejected.

Deeper selective reanalysis increased target coverage and material-regret density at equal teacher-visit cost, but sparse policy-linear QAT did not convert that offline signal into reproducible playing strength. The accepted artifact remains SHA-256 `5d38b3d3f88582212065e6d2aee7b5d638c13e3c4ccaf9e1cab1cd341f757714`, 111,920 bytes. The accepted search and Million website remain unchanged.

## 2026-07-28 — Fresh selective ownership and score auxiliaries

### Pipeline correction

The native selective-reanalysis collector retained b18 policy, value, and searched child values but discarded b18 ownership and score outputs. It now has an opt-in auxiliary-target path that requests root ownership from KataGo analysis and stores the ownership map and root score lead alongside the exact selected root.

KataGo analysis reports ownership in top-left row-major board order. The analysis engine is configured with `reportAnalysisWinratesAs=SIDETOMOVE`, matching Moka's current-player feature perspective.

The training-only score head uses `tanh`, but existing score preprocessing allowed normalized targets beyond its \([-1,1]\) range. Score targets are now clipped after dividing by the 40-point normalization scale. Tests cover auxiliary query routing, ownership order, score extraction, and bounded score normalization. Auxiliary collection remains disabled by default.

### Fresh corpus

The exact accepted artifact generated 128 Moka-versus-b6 games from opening offset 3,300,000. Moka used accepted 64-visit search. Only Moka turns were eligible, 25% were selected with equal uniform and b6-regret components, and native b18 analyzed each selected root at 128 visits with ownership enabled.

The archive contains 1,077 roots across 128 games and occupies 618,549 compressed bytes. Its SHA-256 is `33a1255efaa8f0e8f2dd0a5700b603d75d190c5ac324be49a13dd8bbda60212e`.

All policy, value, ownership, and score targets are finite. The whole-game split contains 878 training, 96 validation, and 103 test roots. Mean ownership is +0.562 on current-player stones and -0.718 on opponent stones, confirming perspective alignment. Score leads range from -87.8 to +87.4 points; 6.0% are clipped to the representable training range.

Loading the accepted checkpoint into the training-only auxiliary network preserves all 108 shared parameter tensors exactly before training.

### Matched ablation

Three matched seeds compared an ordinary continuation with an ownership-plus-score continuation. Every run used:

- one epoch at learning rate 0.000002;
- fourfold replay of the fresh corpus;
- both accepted 64-visit search corpora once;
- batch size 128;
- policy preservation weight 0.25;
- the exact accepted checkpoint as initialization.

The auxiliary arm added only training-time ownership and score heads, ownership weight 0.02, and the fixed score weight 0.1. Auxiliary tensors were stripped before export.

Control and auxiliary validation loss, test loss, and move agreement were nearly identical within each seed. All six exact INT8 exports were distinct and remained 111,920 bytes.

### Exact-artifact screen

On 20 fresh games from opening offset 2,890,000:

| Player        | Wins | Black | White | Caps |
| :------------ | ---: | ----: | ----: | ---: |
| Incumbent     |   10 |     6 |     4 |    0 |
| Control 201   |    8 |     5 |     3 |    0 |
| Auxiliary 201 |    9 |     6 |     3 |    0 |
| Control 202   |    8 |     5 |     3 |    0 |
| Auxiliary 202 |    8 |     5 |     3 |    0 |
| Control 203   |    7 |     5 |     2 |    0 |
| Auxiliary 203 |    7 |     5 |     2 |    0 |

Ownership and score supervision improved one matched seed by one game and tied the other two, but every continuation remained below the incumbent. No candidate advanced to confirmation, and no auxiliary-weight tuning was performed.

Fresh, correctly aligned spatial targets and bounded score targets do not improve the accepted full-network continuation recipe. The collection and normalization fixes remain useful research infrastructure, but every candidate is rejected. The accepted artifact, search, and Million website remain unchanged.

## 2026-07-28 — Fresh searched child-Q auxiliary

The fresh 128-visit selective corpus also contains native b18 values and edge visits for every searched child. All 1,077 roots have finite child-Q targets, with a mean 9.9 and median eight searched children per root. The archive contains 137,852 total child visits.

Three child-Q candidates reused the exact matched-control protocol from the ownership experiment:

- seeds 201, 202, and 203;
- one epoch at learning rate 0.000002;
- fourfold fresh-corpus replay;
- both accepted 64-visit search corpora once;
- batch size 128;
- policy preservation weight 0.25;
- the exact accepted checkpoint as initialization.

The only difference from each control was a training-only searched child-Q head with visit-weighted loss. The Q head was stripped before export. All three exact INT8 exports were distinct and remained 111,920 bytes.

On 20 fresh games from opening offset 2,900,000:

| Player      | Wins | Black | White | Caps |
| :---------- | ---: | ----: | ----: | ---: |
| Incumbent   |   10 |     5 |     5 |    1 |
| Control 201 |    5 |     4 |     1 |    0 |
| Q 201       |    5 |     4 |     1 |    0 |
| Control 202 |    8 |     4 |     4 |    0 |
| Q 202       |    8 |     4 |     4 |    0 |
| Control 203 |    6 |     4 |     2 |    0 |
| Q 203       |    6 |     4 |     2 |    0 |

The incumbent's cap was a losing game and did not contribute a win. Every Q candidate exactly tied its matched control on wins and colors, and every continuation remained below the incumbent. No candidate advanced.

Current-player searched child-Q supervision at this continuation scale changes exact model bytes but does not improve play. All Q candidates are rejected. The accepted artifact, search, and Million website remain unchanged.

## 2026-07-28 — Conservative deep-critical policy soup

Deep-critical seed 182 previously improved both standalone confirmation blocks, 91 wins to 85, but failed the no-cap gate and regressed in a later composite confirmation. It differs from the accepted exact artifact only in `policy_linear.weight` and `policy_linear.bias`.

To test whether a conservative model soup could retain useful policy corrections while damping unstable decisions, exact-dequantized incumbent and candidate tensors were linearly interpolated at candidate weights 0.25, 0.50, and 0.75. Every blend was requantized, materialized as an exact checkpoint, and exported at 111,920 bytes.

On 20 fresh games from opening offset 2,910,000:

| Candidate | Wins | Black | White | Caps |
| :-------- | ---: | ----: | ----: | ---: |
| Incumbent |   14 |     6 |     8 |    0 |
| 25% blend |   13 |     6 |     7 |    0 |
| 50% blend |   10 |     5 |     5 |    0 |
| 75% blend |   10 |     5 |     5 |    0 |

Every blend was weaker than the incumbent, and performance declined as the rejected candidate's contribution increased. No blend advanced to confirmation.

Conservative interpolation does not recover a stable gain from deep-critical seed 182. All blends are rejected. The accepted artifact, search, and Million website remain unchanged.

## 2026-07-28 — Teacher-correction side branches

### Method

KataGo training uses side positions to broaden coverage around informative states. Moka's selective reanalysis previously corrected isolated roots but did not teach continuations after the corrected move.

The native generator now has an opt-in teacher-branch path. When b18's searched top move differs from Moka's actual accepted-search move, it:

1. plays the b18 move from the exact selected root;
2. analyzes the resulting child position with the same native visit budget;
3. stores its policy, value, searched child values, and optional auxiliary targets;
4. assigns the branch to the parent game ID so whole-game splits remain intact.

Branch rows record their parent row index. The path is disabled by default and cannot be combined with counterfactual reanalysis. Tests cover visit normalization, parent/child value perspective, and the existing query behavior.

### Fresh corpus

The exact accepted artifact generated 64 Moka-versus-b6 games from opening offset 3,400,000. Moka used accepted 64-visit search. Only Moka turns were eligible, 25% were selected with equal uniform and b6-regret components, and b18 analyzed roots and branches at 128 visits.

The archive contains 572 selected roots and 385 teacher-correction branches across 64 games. It occupies 442,867 compressed bytes and has SHA-256 `e10a590c72df69c4c777c6926e65c1d1a9792bd3e63b833e93bf4c4b57c43539`.

All features, policies, values, and searched child targets are finite. Policies are normalized. Every branch follows its parent and shares its game ID. Moka matches the b18 top move on 35.7% of selected roots and 42.3% of branch positions.

A root-only archive was derived from the exact same 572 root rows. It occupies 63,545 compressed bytes and has SHA-256 `8efc189b10a1ccbeb4d2865b5a4073cb5c25dc1aa3fef13173252a2977388737`.

### Matched policy-linear QAT

Three matched seed pairs used:

- one exact INT8-aware epoch;
- frozen trunk and policy convolution;
- policy preservation weight 0.25;
- both accepted 64-visit search corpora once;
- sevenfold root-only replay versus fourfold branch replay, approximately 4,000 target rows per arm.

At learning rate 0.000002, all six distinct exact exports reproduced the incumbent exactly on 20 fresh games from opening offset 2,920,000: seven wins, four Black and three White, zero caps, 46 passes, and one resignation. No candidate advanced.

A predeclared final intensity increased only the learning rate to 0.000010. Every exact export remained 111,920 bytes.

On 20 new games from opening offset 2,930,000:

| Player     | Wins | Black | White | Caps |
| :--------- | ---: | ----: | ----: | ---: |
| Incumbent  |    8 |     5 |     3 |    0 |
| Root 231   |    7 |     5 |     2 |    0 |
| Branch 231 |    8 |     5 |     3 |    0 |
| Root 232   |    7 |     5 |     2 |    0 |
| Branch 232 |    7 |     5 |     2 |    0 |
| Root 233   |    7 |     5 |     2 |    0 |
| Branch 233 |    7 |     5 |     2 |    0 |

Branch supervision recovered one game over one matched root-only control and tied the incumbent, but no candidate beat production. No candidate advanced to confirmation, and no further intensity was tested.

Teacher-correction side branches provide clean sequential coverage but do not improve this policy-linear continuation recipe. All candidates are rejected. The opt-in generator remains available for future architectures; the accepted artifact, search, and Million website remain unchanged.

## 2026-07-28 — Moka-only blunder prediction and selective intervention

### Loss profile

The exact accepted artifact was evaluated in all eight aligned symmetries on the existing 2,181-root selective b18 corpus from opening offset 3,000,000. A b18-critical mistake remained defined as at least 0.2 searched value regret for Moka's actual 64-visit move.

The corpus contained 7.02% critical roots. Jensen–Shannon disagreement among Moka's legal normalized symmetry policies was only a weak predictor: the highest-disagreement 20% contained 29.4% of critical roots at a 10.3% critical rate.

A predeclared screen spent extra search only at high-disagreement roots:

| Configuration      | Wins | Black | White | Caps | Extensions |
| :----------------- | ---: | ----: | ----: | ---: | ---------: |
| Accepted 64 visits |   11 |     7 |     4 |    0 |          0 |
| Fixed 72 visits    |    9 |     4 |     5 |    0 |          0 |
| 64→80, top 20%     |   10 |     6 |     4 |    0 |        120 |
| 64→96, top 20%     |   11 |     6 |     5 |    0 |        123 |
| 64→96, top 10%     |   10 |     5 |     5 |    0 |         75 |

No disagreement schedule beat the control.

### Frozen multivariate risk score

A class-balanced logistic score combined only signals available from Moka's existing root evaluation: symmetry policy disagreement, value spread, top-move votes, accepted-policy confidence and entropy, view dispersion, board occupancy and stone balance, mean value, value magnitude, canonical-value deviation, and top-move agreement indicators.

The score was fitted once on the 2,181-root offset-3,000,000 corpus. The untouched 1,077-root, 128-visit auxiliary corpus from opening offset 3,300,000 supplied the independent proxy test.

| Corpus    | Critical rate | ROC AUC | Top-20% critical rate | Top-20% recall |
| :-------- | ------------: | ------: | --------------------: | -------------: |
| Fit       |         7.02% |   0.774 |                16.97% |          48.4% |
| Untouched |         6.04% |   0.717 |                13.49% |          44.6% |

The untouched top 10% had a 15.22% critical rate and recovered 21.5% of critical roots. The score therefore generalized as a blunder predictor even though no runtime intervention had yet been shown to help.

On the same fresh 20-game screen used for the fixed controls:

| Risk schedule  | Wins | Black | White | Caps | Extensions |
| :------------- | ---: | ----: | ----: | ---: | ---------: |
| 64→80, top 20% |   10 |     6 |     4 |    0 |        107 |
| 64→96, top 20% |   12 |     7 |     5 |    0 |        106 |
| 64→96, top 10% |   12 |     7 |     5 |    0 |         55 |

The top-10% schedule was frozen because it tied the screen lead with half as many extensions.

On 100 new paired games from opening offset 2,960,000, accepted Moka and the frozen schedule each won 40 games with zero caps. The candidate extended 258 roots and took 280.2 seconds versus 267.4 seconds for the control. More visits at predicted mistakes changed the color split but did not improve aggregate strength, so the search extension was rejected.

### Selective policy routing

Deep-critical seed 182 previously improved both of its first standalone confirmation blocks but failed later confirmation. It differs from accepted Moka only in the linear policy tensors. A second intervention retained the accepted trunk, value, descendant evaluator, and 64-visit tree, but substituted seed 182's root policy only when the frozen risk score fired.

On 20 fresh games from opening offset 2,970,000:

| Root policy            | Wins | Black | White | Caps | Routed roots |
| :--------------------- | ---: | ----: | ----: | ---: | -----------: |
| Accepted               |    8 |     4 |     4 |    0 |            0 |
| Seed 182 at every root |    6 |     3 |     3 |    0 |          651 |
| Seed 182, risk top 20% |    9 |     4 |     5 |    0 |          143 |
| Seed 182, risk top 10% |    9 |     4 |     5 |    0 |           72 |

The top-10% route was frozen as the smaller screen leader. On 100 untouched paired games from opening offset 2,980,000, accepted Moka won 41 games and the routed candidate won 39. Both had zero caps. The candidate routed 261 roots and took 270.8 seconds versus 268.6 seconds for the control.

The independent risk score identifies materially bad roots, but neither extra visits nor routing to a globally unstable policy improved confirmed play. Both runtime paths were removed after rejection. This separates detection from correction: future work should use the score to improve training coverage or candidate generation, not to spend more of the same search or switch between the same two policies. The accepted model, 64-visit search, 111,920-byte artifact, and Million website remain unchanged.

## 2026-07-28 — Blunder-risk teacher selection

### Hypothesis

The frozen Moka-only blunder score may be more useful for allocating offline b18 labels than for changing runtime search. The score was integrated as an opt-in selective-reanalysis mode. It evaluates the existing model in all eight root symmetries, uses the previously frozen logistic coefficients, and never runs in the browser or during arena play.

The implementation first reproduced the untouched offset-3,300,000 benchmark exactly:

| Metric                  | Result |
| :---------------------- | -----: |
| Critical rate           |  6.04% |
| ROC AUC                 |  0.717 |
| Top-20% critical rate   | 13.49% |
| Top-20% critical recall |  44.6% |

### Fresh selected corpus

The exact accepted artifact generated 256 new deterministic games from opening offset 3,500,000. Moka used the accepted 64-visit search against b6c96. Only Moka turns were eligible. One quarter of each game's eligible turns were labeled by native b18 at 64 visits, split evenly between uniform selection and highest predicted blunder risk.

The resulting archive contains 2,145 roots and has SHA-256 `98cdb718b186866681a8cb1682acd2c743aad6f1d90d9e62c9b86174d5e417af`. B18 evaluated Moka's selected move on 90.0% of roots. Moka selected b18's top move on 42.2%.

| Selector corpus             | Roots | Regret ≥ 0.2 | Critical rate | Winner flips |
| :-------------------------- | ----: | -----------: | ------------: | -----------: |
| Previous b6-regret selector | 2,181 |          153 |         7.02% |           51 |
| Moka-risk selector          | 2,145 |          177 |         8.25% |           51 |

The Moka-risk selector increased material-root density by 17.5% relative and found 24 more material errors with slightly fewer labels. Its critical archive has SHA-256 `fbb750af0f4a419bf4215fb9cf259b1aa72ae87a20eac2cc3c3d16a0332d2fec`. It contains 146 material training roots, 13 validation roots, and 18 test roots.

### Exact policy-QAT screen

Seeds 241, 242, and 243 used the accepted policy-only recipe: one epoch, learning rate 0.000020, eightfold critical replay, both accepted search corpora once, frozen trunk and policy convolution, INT8-aware forward passes, and policy preservation weight 0.25.

Every exact export remained 111,920 bytes. Seed 243 had the best new-corpus test result, improving hard-target move agreement from 29.3% to 31.1% while matching the incumbent's loss. Seeds 241 and 242 preserved the older critical sets more closely.

On 20 fresh games from opening offset 3,010,000:

| Player    | Wins | Black | White | Caps |
| :-------- | ---: | ----: | ----: | ---: |
| Incumbent |   10 |     5 |     5 |    0 |
| Seed 241  |    7 |     5 |     2 |    0 |
| Seed 242  |    7 |     4 |     3 |    0 |
| Seed 243  |    7 |     3 |     4 |    0 |

All candidates regressed and were rejected without confirmation. The blunder-risk selector remains available for offline collection, but sparse policy-linear correction is still unstable. The accepted checkpoint, search, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Symmetry-consensus distillation

### Search audit

The accepted MCTS was audited before another training round. Legal expansion, current-player value perspective, terminal backup signs, root visit accounting, subtree alignment, and pass handling were consistent with the intended player. No correctness fix was justified.

The actionable mismatch was between root and descendant evaluation. Every real root uses Moka's eight aligned symmetries, but each descendant uses one canonical view. Across three disjoint accepted-player corpora, the canonical policy agreed with the accepted eight-view consensus on 77.3%, 77.5%, and 81.6% of positions. Mean consensus-to-canonical KL was 0.0926, 0.0923, and 0.1046. The same symmetry instability had already predicted b18-critical errors.

### Consensus targets

An offline generator now evaluates every input position in all eight symmetries, aligns the policies, applies the accepted 0.125 geometric blend, and averages the values. It replaces only the policy and value targets in an existing archive. It does not run during browser inference, search, or arena play.

The frozen accepted artifact generated consensus targets for 6,480 positions:

| Opening offset | Positions | Target archive SHA-256                                             |
| -------------: | --------: | :----------------------------------------------------------------- |
|      3,000,000 |     2,181 | `b6bf51161595c78f821e8f50ecaccad8a068c8228eb0af1259183e96fc203afc` |
|      3,100,000 |     2,154 | `e6ea90788dfd48c8a3cac7e7a56df519cecbe0941fd1c2fb7b49706d17171333` |
|      3,500,000 |     2,145 | `bc384529e096ee3459ef911825c9cae32c1b9d6d501b61f486cab70375365c2b` |

The committed generator reproduced the experimental policy and value targets exactly.

### Training

Two quantization-aware one-epoch families started from the exact accepted artifact and used all three corpora with policy-preservation weight 0.25:

- policy-linear-only seeds 251–253 at learning rate 0.000020;
- full-network seeds 254–256 at learning rate 0.000002.

These conservative rates changed symmetry KL by too little to justify an arena. A bounded stronger round retained the targets, epoch count, preservation, and exact INT8 path:

- policy-linear-only seeds 257–259 at learning rate 0.000100;
- full-network seeds 260–262 at learning rate 0.000010.

The stronger policy-only family increased symmetry disagreement and was rejected offline. The full-network family reduced consensus KL by 5.6%–6.8% and improved b18 policy loss on the fresh 3,300,000 and 3,500,000 test buckets. Seed 260 retained the smallest value-error regression among the family and advanced with both siblings to the fixed screen.

### Arena

On 20 fresh games from opening offset 3,020,000:

| Player    | Wins | Black | White | Caps |
| :-------- | ---: | ----: | ----: | ---: |
| Incumbent |    7 |     3 |     4 |    0 |
| Seed 260  |    8 |     5 |     3 |    0 |
| Seed 261  |    6 |     5 |     1 |    0 |
| Seed 262  |    7 |     5 |     2 |    0 |

Seed 260 was frozen as the unique screen leader. Two untouched 100-game blocks compared that exact candidate with the incumbent:

| Opening offset | Incumbent wins | Candidate wins | Incumbent caps | Candidate caps |
| -------------: | -------------: | -------------: | -------------: | -------------: |
|      3,030,000 |             35 |             50 |              0 |              0 |
|      3,040,000 |             45 |             42 |              0 |              0 |
|      **Total** |         **80** |         **92** |          **0** |          **0** |

The incumbent split 42 Black and 38 White wins. Seed 260 split 41 Black and 51 White wins. Candidate runtime was 518.9 seconds versus 533.8 seconds for the incumbent. The candidate added 12 completed wins without a cap or runtime regression and is promoted.

The accepted float-shadow checkpoint now has SHA-256 `58abe149b8dd9cc1cb5f869f8c6cd7788f3b0858294ea553e0531161cded0474`. Its exact-dequantized checkpoint has SHA-256 `a36f61fedd6e9ecc8bd426e8fdaa08a4f989e39c1ffd78d8b3cfab3d20cf5563`. The 111,920-byte browser artifact has SHA-256 `51a9f91f66d8c3725911afad7ad299bb6b50ea3a5ef2b16e2a62ec5379942a75`.

The previously frozen blunder-risk score remained valid after promotion. On the untouched offset-3,300,000 proxy corpus, ROC AUC increased from 0.717 to 0.724 while top-20% critical rate remained 13.49% and recall remained 44.6%.

The Million website remains unchanged.

## 2026-07-28 — Post-promotion calibration

### Search constants

The promoted symmetry-distilled artifact was used for every control and candidate. Each screen used 20 fresh games at 64 visits with identical openings and zero capped games.

| Parameter                    | Candidates                     | Accepted result                                                                                                                  |
| :--------------------------- | :----------------------------- | :------------------------------------------------------------------------------------------------------------------------------- |
| Root geometric policy weight | 0, 0.0625, 0.125, 0.1875, 0.25 | The frozen 0.125 remained accepted. Zero and 0.0625 each won eight games, while 0.125 won seven; there was no unique challenger. |
| PUCT exploration             | 1.5, 1.75, 2, 2.25, 2.5        | The frozen 2 remained accepted with eight wins. Only 2.5 tied it, so there was no unique challenger.                             |
| Value weight                 | 1, 1.125, 1.25, 1.375, 1.5     | Weight 1 led the screen with 11 wins versus eight for the frozen 1.25.                                                           |

The value-weight challenger advanced to 100 fresh games from opening offset 3,070,000. Weight 1 won 34 games, split 16 Black and 18 White, while the frozen 1.25 won 38, split 17 Black and 21 White. Both had zero caps. Weight 1 was rejected.

The accepted search remains 64 visits, root geometric policy weight 0.125, exploration 2, value weight 1.25, and first-play urgency reduction 0.25.

### Second-generation consensus targets

The promoted model generated a new set of eight-symmetry consensus targets:

| Opening offset | Compressed bytes | Target archive SHA-256                                             |
| -------------: | ---------------: | :----------------------------------------------------------------- |
|      3,000,000 |        1,125,645 | `c517b98e9938a57582e3ed041566c9b0b08907cab2e8a23a0ac528595c3419dd` |
|      3,100,000 |        1,099,182 | `4ee81747444ff51d4b7b11db707f3fcd51296c6865920bfc01b46390b3b0e8dd` |
|      3,500,000 |        1,040,385 | `97fe558381d8e8a13e6361bb43d4be7216fbc0981a6373e1df5d8674447fbbc0` |

Full-network exact-QAT seeds 263–265 reused the promoted recipe: one epoch, learning rate 0.000010, policy preservation weight 0.25, and all three target corpora. On the 3,500,000 consensus test bucket, the incumbent had loss 2.2795, move agreement 75.6%, and value error 0.1214. Every candidate regressed: their losses were 2.2926–2.2960, move agreement was 74.7%–75.1%, and value error was 0.1329–0.1401.

All three also regressed on the independent b18 test buckets at opening offsets 3,300,000 and 3,500,000. They were rejected offline without spending arena games. Repeated self-distillation from the already-distilled student amplified its approximation errors instead of adding new information.

## 2026-07-28 — Full-model interpolation

### Hypothesis

The promoted model gained 13 White wins and lost one Black win relative to its predecessor. A convex parameter blend could retain more of the predecessor's Black behavior while preserving the promoted model's symmetry improvements.

Exact INT8-aware blends used 25%, 50%, and 75% of the promoted float-shadow parameters. Their exact-dequantized SHA-256 digests were:

| Promoted-model fraction | Exact-dequantized SHA-256                                          |
| ----------------------: | :----------------------------------------------------------------- |
|                     25% | `45d0be6c9071be5677dda11b229ca7102466798536fdc52bd1fe93b44dfb767b` |
|                     50% | `4bb7e376e2420964e2b2ecc39023fb9df0d09fda872cd2c02356d37c497345e9` |
|                     75% | `e2263151fbf7dfba1ffe94cf394e30bcda0617d0967e8731f0952ccaf819734a` |

On the fresh consensus test bucket, all three blends remained between the two endpoint models. The 75% blend had the best blend loss at 2.2856 and the best move agreement at 77.3%, but no blend consistently improved the independent b18 buckets.

### Arena

On 20 fresh games from opening offset 3,090,000, the promoted model and the 75% blend each won 10 games, the 50% blend won nine, and the 25% blend uniquely led with 11. All had zero caps. The 25% blend was frozen for confirmation.

Two untouched paired 100-game blocks produced:

| Opening offset | Promoted wins | Blend wins | Promoted Black / White | Blend Black / White |      Caps |
| -------------: | ------------: | ---------: | :--------------------- | :------------------ | --------: |
|      3,100,000 |            42 |         44 | 22 / 20                | 22 / 22             |     0 / 0 |
|      3,110,000 |            36 |         35 | 19 / 17                | 20 / 15             |     0 / 0 |
|      **Total** |        **78** |     **79** | **41 / 37**            | **42 / 37**         | **0 / 0** |

The blend's one-game aggregate edge was smaller than its opposite block-to-block swings. It regressed in the second block and did not retain the promoted model's demonstrated White advantage. The result is indistinguishable from arena variance, so the blend is rejected.

The accepted model, search, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Mixed teacher and symmetry-consensus targets

### Hypothesis

Pure second-generation self-distillation added no new information and regressed offline. Mixing the promoted model's stable eight-view consensus with the original b18 policy and value labels could retain symmetry consistency while restoring a stronger external training signal.

The symmetry-target generator now accepts a bounded source-target weight. It mixes normalized policy distributions and scalar values before writing the training archive. The default remains pure consensus. A separate command materializes an exact INT8-aware checkpoint from a float shadow, replacing the one-off conversion used by earlier experiments. Tests cover target mixing, invalid weights, and exact checkpoint materialization.

Three matched target families used 12.5%, 25%, and 50% source-teacher weight over the same 6,480 positions. The 50% family's archive SHA-256 digests were:

| Opening offset | Target archive SHA-256                                             |
| -------------: | :----------------------------------------------------------------- |
|      3,000,000 | `db4ff7b4f343d768c33e41194f6e98f28a29315ebcaa6cd13f3fd4bf936aaa6f` |
|      3,100,000 | `09583e732be81855f7d8704eccdb61d214c2cd6703fe0dd57a5c76bbe3c5eaef` |
|      3,500,000 | `8751b56db543cbbda10cd1140bd2ca1ba993dab3904c6a74a7c58824d16d8437` |

Each candidate used seed 266, one full-network INT8-aware epoch, learning rate 0.000010, policy preservation weight 0.25, and the promoted checkpoint as its initialization and preservation reference. Using one seed made source weight the only training variable.

### Offline gate

On the untouched b18 offset-3,500,000 test bucket, the 50% candidate improved loss from 2.9491 to 2.9162 and value error from 0.6076 to 0.5851. On the independent b18 offset-3,300,000 bucket, it improved loss from 2.9976 to 2.9411 and value error from 0.6045 to 0.5716. Its consensus loss regressed from 2.2734 to 2.2931 and top-move agreement fell from 75.6% to 72.4%.

The lighter candidates made smaller or inconsistent teacher improvements. All three advanced to a fixed screen because the offline tradeoff did not identify whether external calibration or consensus agreement mattered more to tree search.

### Arena

On 20 fresh games from opening offset 3,120,000:

| Player               | Wins | Black | White | Caps |
| :------------------- | ---: | ----: | ----: | ---: |
| Accepted             |    5 |     3 |     2 |    0 |
| 12.5% teacher target |    8 |     2 |     6 |    0 |
| 25% teacher target   |   10 |     3 |     7 |    0 |
| 50% teacher target   |   11 |     4 |     7 |    0 |

The 50% candidate was frozen as the unique screen leader. Its exact-dequantized checkpoint has SHA-256 `85556c2015a90c431717879fe05dc295d0eccfc7578ac04ed86ed674da914907`.

On 100 untouched paired games from opening offset 3,130,000, the accepted model won 34 games, split 20 Black and 14 White, in 261.0 seconds. The candidate won 30, split 14 Black and 16 White, in 273.1 seconds. Both had zero caps.

The screen gain reversed under confirmation. Better pointwise agreement with b18 values was not enough to preserve the accepted search policy, and the candidate lost four games while running slower. It is rejected without a second confirmation block.

The target mixer and exact-checkpoint materializer remain as reproducible offline tools. The accepted model, search, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Fresh promoted-search distillation

### Hypothesis

Pointwise teacher/consensus mixing improved offline values but weakened play. The promoted model's actual 64-visit distribution has a closer causal relationship to its accepted moves. Distilling fresh searched trajectories into the single-view evaluator could transfer root search quality without adding browser inference work.

The exact accepted artifact played 128 deterministic games from opening offset 3,600,000 against b6c96. Moka used accepted 64-visit search and eight-view root evaluation. Only Moka decision positions were retained. Policy targets blended 75% of the visit distribution with 25% of the legal eight-view root policy; b6 supplied scalar value labels.

The resulting corpus contains 4,216 positions across 128 games: 3,330 training, 438 validation, and 448 test rows by whole-game split. Every target is finite and normalized. The 738,634-byte archive has SHA-256 `9d960febdc17397ded3d065d775934b6afc895a75cad8fb42cb5eed43a2ee182`.

### Exact-QAT candidates

Seeds 267–269 started from the promoted float shadow. Each used one full-network INT8-aware epoch, learning rate 0.000005, and policy preservation weight 0.25. No older trajectory replay was used.

| Player   | Search loss | Search top move | Search value error | b18 loss | Consensus top move |
| :------- | ----------: | --------------: | -----------------: | -------: | -----------------: |
| Accepted |      1.9819 |           70.1% |             0.3515 |   2.9491 |              75.6% |
| Seed 267 |      1.9625 |           67.6% |             0.3373 |   2.9625 |              72.0% |
| Seed 268 |      1.9625 |           67.9% |             0.3375 |   2.9638 |              72.0% |
| Seed 269 |      1.9614 |           67.6% |             0.3365 |   2.9620 |              72.0% |

Dense search loss and value error improved, but decisive top-move agreement and both independent policy metrics regressed.

On 20 fresh games from opening offset 3,610,000:

| Player   | Wins | Black | White | Caps |
| :------- | ---: | ----: | ----: | ---: |
| Accepted |    5 |     3 |     2 |    0 |
| Seed 267 |    6 |     4 |     2 |    0 |
| Seed 268 |    8 |     5 |     3 |    0 |
| Seed 269 |    8 |     4 |     4 |    0 |

Seeds 268 and 269 tied the screen lead. Seed 269 was frozen because it had the best primary-corpus test loss and value error. Its exact-dequantized checkpoint has SHA-256 `e2e51a93c5cc92362036f39201d64e7da5c33d86528eb45dbf41fdee58dc7176`.

On 100 untouched paired games from opening offset 3,620,000, the accepted model won 43 games, split 25 Black and 18 White, in 261.6 seconds. Seed 269 won 38, split 24 Black and 14 White, in 257.7 seconds. Both had zero caps.

The screen gain reversed by five games under confirmation. Dense visit-distribution matching softened decisive choices that the accepted tree needed. This branch is rejected without a second confirmation block. Future search distillation should protect ranked or high-visit actions explicitly rather than optimize dense cross-entropy alone.

The accepted model, search, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Ranked fresh-search continuation

### Offline gate

Dense fresh-search distillation lowered average loss but reduced held-out top-move agreement from 70.1% to approximately 67.7%. Four matched seed-270 candidates used the same fresh 4,216-position corpus, accepted initialization, one exact-QAT epoch, learning rate 0.000005, and policy preservation weight 0.25:

| Objective              | Test loss | Test top move | Test value error |
| :--------------------- | --------: | ------------: | ---------------: |
| Accepted               |    1.9819 |         70.1% |           0.3515 |
| 50% hard winner        |    1.9620 |         68.5% |           0.3371 |
| Top-eight truncation   |    1.9613 |         68.1% |           0.3374 |
| Top-four truncation    |    1.9617 |         68.5% |           0.3379 |
| Top-eight listwise 0.1 |    1.9614 |         68.1% |           0.3373 |

Every objective improved dense loss and value error but failed the predeclared requirement to preserve the accepted top-move agreement. None entered the arena. Rank protection changed the update direction but did not prevent decisive policy drift.

## 2026-07-28 — Descendant symmetry consensus

### Hypothesis

The real-move root uses all eight aligned board symmetries, while every tree descendant previously used one canonical orientation. The promoted checkpoint reduced this mismatch through offline distillation but did not eliminate it. Test-time symmetry consensus can remove the residual orientation error without changing weights, model bytes, visits, or teacher access.

On the 4,216-position fresh search corpus, each canonical-plus-one-view pair was compared with the accepted eight-view target:

| Complementary view | Consensus KL | Top-move agreement | Value error |
| :----------------- | -----------: | -----------------: | ----------: |
| Reflection         |     0.036515 |              84.6% |    0.059383 |
| 90°                |     0.034500 |              84.9% |    0.054325 |
| 90° + reflection   |     0.039397 |              83.9% |    0.059570 |
| 180°               |     0.033167 |              85.5% |    0.058033 |
| 180° + reflection  |     0.039939 |              84.5% |    0.065120 |
| 270°               |     0.034317 |              85.4% |    0.057719 |
| 270° + reflection  |     0.039985 |              84.6% |    0.066486 |

The 180°, 90°, and 270° pairs advanced as the best policy, value, and close-policy controls. On 20 fresh games from opening offset 3,630,000, canonical descendants won nine, while the pairs won 10, seven, and 11. All had zero caps. The 270° pair was frozen as the unique screen leader.

Two untouched 100-game paired blocks produced:

| Opening offset | Canonical wins | Pair wins | Canonical caps | Pair caps |
| -------------: | -------------: | --------: | -------------: | --------: |
|      3,640,000 |             40 |        47 |              0 |         0 |
|      3,650,000 |             40 |        35 |              0 |         0 |
|      **Total** |         **80** |    **82** |          **0** |     **0** |

The pair's first-block gain reversed in the second block. A two-game aggregate edge with opposite block outcomes is insufficient, so the pair is rejected.

### Full consensus

The exact eight-view descendant ensemble won nine of 20 screen games from opening offset 3,660,000 versus seven for canonical descendants. Both had zero caps and took 55.2 and 54.4 seconds.

Three untouched paired 100-game blocks produced:

| Opening offset | Canonical wins | Full-consensus wins | Canonical Black / White | Consensus Black / White | Canonical / consensus caps |
| -------------: | -------------: | ------------------: | :---------------------- | :---------------------- | -------------------------: |
|      3,670,000 |             39 |                  47 | 19 / 20                 | 26 / 21                 |                      0 / 0 |
|      3,680,000 |             42 |                  44 | 23 / 19                 | 20 / 24                 |                      0 / 1 |
|      3,700,000 |             37 |                  43 | 14 / 23                 | 26 / 17                 |                      1 / 1 |
|      **Total** |        **118** |             **134** | **56 / 62**             | **72 / 62**             |                  **1 / 2** |

Full consensus improved every block by eight, two, and six games, adding 16 wins in aggregate. Runtime was 852.1 seconds versus 791.6 for canonical descendants, a 7.6% increase enabled by batched symmetry inference.

The repository intentionally uses simple ko to match the b6c96 teacher configuration. The three-block audit contained one canonical and two consensus repetition caps. Canonical Moka received one area-adjudicated cap win; both consensus caps were Moka losses. The strength gain therefore did not come from cap adjudication or teacher information.

Full descendant symmetry consensus is accepted as the default research search player. `--no-symmetry-ensemble` provides the canonical control. The default propagates through arena play and future search-data collection. The 64-visit budget, 104,129 parameters, 111,920-byte browser artifact, and model digest remain unchanged. The Million website remains unchanged.

## 2026-07-28 — Full-symmetry search calibration

### Value weight

The accepted value weight was re-audited because all earlier calibration used canonical descendants. On 20 fresh full-symmetry games from opening offset 3,710,000:

| Value weight | Wins | Black | White | Caps |
| -----------: | ---: | ----: | ----: | ---: |
|         1.00 |    6 |     3 |     3 |    0 |
|         1.25 |   10 |     5 |     5 |    0 |
|         1.50 |    8 |     4 |     4 |    0 |

The accepted 1.25 remained the unique winner and was retained.

### Exploration

On 20 fresh games from opening offset 3,720,000, exploration 1.5 won 13 games, accepted 2.0 won nine, and 2.5 won five. All had zero caps. Exploration 1.5 advanced as the unique screen winner.

On 100 untouched paired games from opening offset 3,730,000, accepted exploration 2.0 won 54 games, split 28 Black and 26 White, in 299.2 seconds. Candidate 1.5 won 43, split 27 Black and 16 White, in 303.6 seconds. Both had zero caps. The screen gain reversed by 11 games, so exploration 1.5 is rejected and 2.0 remains accepted.

### First-play urgency

On 20 fresh games from opening offset 3,740,000:

| FPU reduction | Wins | Black | White | Caps |
| ------------: | ---: | ----: | ----: | ---: |
|          0.00 |    6 |     3 |     3 |    0 |
|          0.25 |   10 |     6 |     4 |    0 |
|          0.50 |   11 |     4 |     7 |    0 |

Reduction 0.5 advanced as the unique screen winner. Two untouched paired 100-game blocks produced:

| Opening offset | FPU 0.25 wins | FPU 0.50 wins | FPU 0.25 Black / White | FPU 0.50 Black / White |      Caps |
| -------------: | ------------: | ------------: | :--------------------- | :--------------------- | --------: |
|      3,750,000 |            44 |            52 | 20 / 24                | 26 / 26                |     0 / 0 |
|      3,760,000 |            39 |            44 | 23 / 16                | 24 / 20                |     0 / 0 |
|      **Total** |        **83** |        **96** | **43 / 40**            | **50 / 46**            | **0 / 0** |

FPU reduction 0.5 improved both blocks by eight and five games, adding 13 wins in aggregate without a cap regression. Runtime fell from 611.1 to 590.2 seconds, a 3.4% improvement.

FPU reduction 0.5 is accepted. The full-symmetry evaluator, exploration 2.0, value weight 1.25, 64-visit budget, model parameters, browser artifact, and model digest remain unchanged. The Million website remains unchanged.

## 2026-07-28 — Local full-symmetry coefficient refinement

### Screens

The accepted full-symmetry search was refined locally without combining unconfirmed changes. Each screen used 20 fresh games, the exact accepted INT8 artifact, 64 visits, exploration 2.0, value weight 1.25, FPU reduction 0.5, and full descendant symmetry except for the coefficient under test.

On opening offset 3,780,000:

| FPU reduction | Wins | Black | White | Caps |
| ------------: | ---: | ----: | ----: | ---: |
|         0.375 |    7 |     5 |     2 |    0 |
|         0.500 |    9 |     5 |     4 |    0 |
|         0.625 |   10 |     5 |     5 |    0 |

On opening offset 3,790,000:

| Exploration | Wins | Black | White | Caps |
| ----------: | ---: | ----: | ----: | ---: |
|        1.75 |   11 |     6 |     5 |    0 |
|        2.00 |    7 |     3 |     4 |    0 |
|        2.25 |    5 |     3 |     2 |    0 |

On opening offset 3,800,000:

| Value weight | Wins | Black | White | Caps |
| -----------: | ---: | ----: | ----: | ---: |
|        1.125 |   10 |     5 |     5 |    0 |
|        1.250 |    9 |     5 |     4 |    0 |
|        1.375 |    7 |     5 |     2 |    0 |

Exploration 1.75 had the largest screen gain and advanced first. FPU 0.625 and value weight 1.125 were the remaining unique screen leaders.

### Paired confirmations

Each challenger played the same 100 deterministic openings as its accepted control:

| Opening offset | Coefficient        | Control wins | Candidate wins | Control Black / White | Candidate Black / White | Control / candidate caps |
| -------------: | :----------------- | -----------: | -------------: | :-------------------- | :---------------------- | -----------------------: |
|      3,810,000 | Exploration 1.75   |           51 |             46 | 31 / 20               | 29 / 17                 |                    0 / 0 |
|      3,820,000 | FPU 0.625          |           46 |             43 | 25 / 21               | 22 / 21                 |                    0 / 0 |
|      3,830,000 | Value weight 1.125 |           46 |             44 | 22 / 24               | 21 / 23                 |                    0 / 0 |

All three screen gains reversed on their untouched confirmations. Exploration 1.75 lost five games, FPU 0.625 lost three, and value weight 1.125 lost two. No run reached the move cap, so cap adjudication cannot explain the result.

Exploration 2.0, FPU reduction 0.5, and value weight 1.25 remain accepted. The checkpoint, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Full-symmetry adaptive visit allocation

### Ceiling screen

The earlier adaptive-search experiment used the old canonical 256- and 512-visit players. It was re-audited under the accepted exact INT8 artifact, 64-visit base, full descendant symmetry, exploration 2.0, value weight 1.25, and FPU reduction 0.5. A root received additional simulations only when its top-two visit margin was below the configured threshold.

On 20 fresh games from opening offset 3,840,000, the threshold remained at 10%:

| Maximum visits | Wins | Black | White | Caps | Runtime |
| -------------: | ---: | ----: | ----: | ---: | ------: |
|       Fixed 64 |    8 |     4 |     4 |    0 |   60.5s |
|             80 |    9 |     4 |     5 |    0 |   60.6s |
|             96 |    9 |     5 |     4 |    0 |   63.6s |
|            128 |    8 |     4 |     4 |    0 |   72.0s |

Maximum 80 tied the win leader at the lowest runtime and advanced to threshold calibration.

### Margin screen

On 20 fresh games from opening offset 3,850,000:

| Maximum / margin | Wins | Black | White | Caps | Runtime |
| :--------------- | ---: | ----: | ----: | ---: | ------: |
| Fixed 64         |   12 |     7 |     5 |    0 |   55.2s |
| 80 / 5%          |   12 |     6 |     6 |    0 |   54.4s |
| 80 / 10%         |   12 |     7 |     5 |    0 |   55.6s |
| 80 / 15%         |   13 |     7 |     6 |    0 |   57.5s |
| 80 / 20%         |   13 |     7 |     6 |    0 |   57.5s |

Margins 15% and 20% tied. The smaller 15% intervention was frozen before confirmation.

### Paired confirmation

Two untouched 100-game blocks produced:

| Opening offset | Fixed wins | Adaptive wins | Fixed Black / White | Adaptive Black / White | Fixed / adaptive caps | Fixed / adaptive runtime |
| -------------: | ---------: | ------------: | :------------------ | :--------------------- | --------------------: | -----------------------: |
|      3,860,000 |         53 |            56 | 26 / 27             | 28 / 28                |                 0 / 0 |          284.7s / 302.3s |
|      3,870,000 |         46 |            44 | 26 / 20             | 23 / 21                |                 0 / 0 |          276.7s / 282.0s |
|      **Total** |     **99** |       **100** | **52 / 47**         | **51 / 49**            |             **0 / 0** |      **561.4s / 584.3s** |

The first-block three-game gain reversed by two games in the second block. A one-game aggregate edge with opposite block outcomes and 4.1% more runtime is insufficient evidence of stronger play. Adaptive 64-to-80 search at a 15% margin is rejected.

Fixed 64 visits remain accepted. The model, artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Descendant symmetry aggregation

### Robust value aggregation

Root-only trimmed symmetry values were previously behaviorally inert because a real root's aggregate value has limited influence after expansion. Full descendant symmetry makes the same aggregation actionable: every descendant value enters a search backup.

A research control blended the arithmetic mean of the eight aligned values with a mean that discarded the highest and lowest view. On 20 fresh games from opening offset 3,890,000:

| Trimmed-value weight | Wins | Black | White | Caps | Runtime |
| -------------------: | ---: | ----: | ----: | ---: | ------: |
|                 0.00 |   10 |     4 |     6 |    0 |   53.3s |
|                 0.25 |    9 |     4 |     5 |    0 |   55.9s |
|                 0.50 |    9 |     4 |     5 |    0 |   54.8s |
|                 0.75 |    9 |     4 |     5 |    0 |   54.7s |
|                 1.00 |    9 |     4 |     5 |    0 |   56.6s |

Arithmetic averaging beat every nonzero blend. No candidate advanced, and the rejected runtime option was removed.

### Geometric policy coefficient

The accepted full descendant ensemble initially inherited the root's 0.125 geometric policy blend. That coefficient was selected for real roots before full descendant symmetry existed. Root and descendant coefficients were separated so they could be calibrated independently without changing default behavior. Search-data generators now pass the root coefficient explicitly.

On 20 fresh games from opening offset 3,910,000:

| Descendant geometric weight | Wins | Black | White | Caps | Runtime |
| --------------------------: | ---: | ----: | ----: | ---: | ------: |
|                      0.0000 |   12 |     8 |     4 |    0 |   55.6s |
|                      0.0625 |   12 |     8 |     4 |    0 |   56.2s |
|                      0.1250 |   11 |     7 |     4 |    0 |   55.4s |
|                      0.2500 |   11 |     6 |     5 |    0 |   56.3s |
|                      0.5000 |    9 |     6 |     3 |    0 |   56.0s |

Arithmetic and 0.0625 tied the lead. A fresh 20-game tie-break at opening offset 3,920,000 produced:

| Descendant geometric weight | Wins | Black | White | Caps | Runtime |
| --------------------------: | ---: | ----: | ----: | ---: | ------: |
|                      0.0000 |   10 |     4 |     6 |    0 |   53.9s |
|                      0.0625 |    9 |     4 |     5 |    0 |   53.6s |
|                      0.1250 |    9 |     4 |     5 |    0 |   52.8s |

Pure arithmetic advanced as the unique tie-break winner. On 100 untouched paired games from opening offset 3,930,000, accepted descendant weight 0.125 won 44 games, split 21 Black and 23 White, with zero caps in 284.8 seconds. Arithmetic also won 44, split 20 Black and 24 White, with zero caps in 279.8 seconds.

The candidate failed to confirm the small screen gains and is rejected. Descendant geometric weight 0.125 remains accepted. The independent research control remains available for reproducibility. The checkpoint, browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Descendant policy temperature

### Hypothesis

Root-policy temperature had been calibrated, but descendant priors remained at temperature 1.0 after full symmetry and FPU reduction 0.5 changed tree allocation. A separate descendant temperature can sharpen or flatten Moka's own symmetry-aggregated priors without changing the root policy, model evaluations, payload, legal moves, or teacher access.

Root evaluators in the arena and both search-data generators were explicitly pinned to the root temperature. Default and explicit descendant temperature 1.0 reproduced identical moves, passes, and outcomes.

### Broad screen

On 20 fresh games from opening offset 3,950,000:

| Descendant temperature | Wins | Black | White | Caps | Runtime |
| ---------------------: | ---: | ----: | ----: | ---: | ------: |
|                   0.70 |    7 |     3 |     4 |    0 |   56.0s |
|                   0.80 |   10 |     5 |     5 |    0 |   56.0s |
|                   0.90 |   11 |     5 |     6 |    0 |   57.0s |
|                   1.00 |    9 |     3 |     6 |    0 |   54.4s |
|                   1.10 |    8 |     3 |     5 |    0 |   55.5s |
|                   1.20 |    6 |     1 |     5 |    0 |   55.2s |

Temperature 0.9 advanced as the unique screen leader. On 100 untouched paired games from opening offset 3,960,000, accepted temperature 1.0 won 48 games, split 24 Black and 24 White, with zero caps in 270.1 seconds. Temperature 0.9 also won 48, split 21 Black and 27 White, with zero caps in 268.7 seconds. It failed to confirm the screen gain and was rejected.

### Local refinement

The broad screen had a clear interior peak, so one bounded local refinement tested the neighboring interval on 20 fresh games from opening offset 3,970,000:

| Descendant temperature | Wins | Black | White | Caps | Runtime |
| ---------------------: | ---: | ----: | ----: | ---: | ------: |
|                   0.85 |   10 |     4 |     6 |    0 |   57.1s |
|                   0.90 |    9 |     3 |     6 |    0 |   58.4s |
|                   0.95 |    7 |     3 |     4 |    0 |   55.7s |
|                   1.00 |    7 |     3 |     4 |    0 |   59.0s |

Temperature 0.85 advanced as the unique local leader. On 100 untouched paired games from opening offset 3,980,000, accepted temperature 1.0 won 44 games, split 22 Black and 22 White, with zero caps in 267.6 seconds. Temperature 0.85 won 41, split 19 Black and 22 White, with zero caps in 266.9 seconds.

Both sharpened candidates failed independent confirmation. Descendant temperature 1.0 remains accepted. The independent research control remains available for reproducibility. The checkpoint, browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Full-symmetry descendant exploration

### Hypothesis

The earlier root/descendant exploration split was rejected when real roots used eight symmetries but descendants used one canonical view. Full descendant symmetry and FPU reduction 0.5 directly changed descendant prior reliability and allocation. The split was therefore re-audited under the accepted topology.

An explicit descendant coefficient overrides exploration only below the real root. Its default inherits root exploration 2.0. Ordinary and sequential-halving paths share the same resolver. Default inheritance and explicit descendant 2.0 reproduced identical moves, passes, outcomes, and runtime.

### Broad screen

The prior experiment's coefficient grid was reused on 20 fresh games from opening offset 4,000,000:

| Descendant exploration | Wins | Black | White | Caps | Runtime |
| ---------------------: | ---: | ----: | ----: | ---: | ------: |
|                  0.750 |    9 |     5 |     4 |    0 |   47.8s |
|                  1.250 |   10 |     5 |     5 |    0 |   54.1s |
|                  1.500 |    9 |     4 |     5 |    0 |   54.5s |
|                  2.000 |    9 |     3 |     6 |    0 |   54.4s |
|                  2.500 |    7 |     2 |     5 |    0 |   56.5s |

Coefficient 1.25 advanced as the unique screen leader. On 100 untouched paired games from opening offset 4,010,000, accepted descendant exploration 2.0 won 43 games, split 22 Black and 21 White, with zero caps in 280.4 seconds. Candidate 1.25 also won 43, split 25 Black and 18 White, with zero caps in 271.6 seconds. It failed to confirm the screen gain and was rejected.

### Local refinement

One bounded local refinement tested the neighborhood around 1.25 on 20 fresh games from opening offset 4,020,000:

| Descendant exploration | Wins | Black | White | Caps | Runtime |
| ---------------------: | ---: | ----: | ----: | ---: | ------: |
|                  1.000 |   12 |     6 |     6 |    0 |   52.5s |
|                  1.125 |   10 |     4 |     6 |    0 |   54.7s |
|                  1.250 |    8 |     4 |     4 |    0 |   52.6s |
|                  1.375 |    9 |     4 |     5 |    0 |   54.8s |
|                  2.000 |   10 |     6 |     4 |    0 |   53.4s |

Coefficient 1.0 advanced as the unique local leader. Two untouched 100-game paired blocks produced:

| Opening offset | Control wins | Candidate wins | Control Black / White | Candidate Black / White | Control / candidate caps | Control / candidate runtime |
| -------------: | -----------: | -------------: | :-------------------- | :---------------------- | -----------------------: | --------------------------: |
|      4,030,000 |           42 |             51 | 22 / 20               | 24 / 27                 |                    0 / 0 |             282.5s / 274.0s |
|      4,040,000 |           44 |             34 | 22 / 22               | 17 / 17                 |                    0 / 0 |             279.0s / 262.7s |
|      **Total** |       **86** |         **85** | **44 / 42**           | **41 / 44**             |                **0 / 0** |         **561.5s / 536.7s** |

The first-block nine-game gain reversed by ten games in the second block. The candidate lost one game in aggregate despite ending games sooner. Descendant exploration 1.0 is rejected.

Exploration 2.0 remains accepted at root and descendants. The independent research control remains available for reproducibility. The checkpoint, browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Low-visit child-Q shrinkage

### Hypothesis

At a 64-evaluation budget, one leaf evaluation immediately becomes a visited child's full Q estimate. A small Bayesian pseudo-count can shrink low-visit child Q toward the parent mean, reducing sensitivity to a noisy first evaluation while decaying as real visits accumulate.

The method changes only PUCT scoring. It uses Moka's existing parent value, child value, and visit count. It adds no evaluation, model tensor, rule heuristic, outcome signal, or teacher access. A zero pseudo-count is exactly disabled. The implementation propagates through ordinary search, batched search, adaptive extensions, and sequential halving. Regression tests cover shrinkage math, invalid negative counts, and the disabled default. Default and explicit zero reproduced identical moves, outcomes, and passes.

### Screen

On 20 fresh games from opening offset 4,060,000:

| Q pseudo-count | Wins | Black | White | Caps | Runtime |
| -------------: | ---: | ----: | ----: | ---: | ------: |
|           0.00 |   11 |     6 |     5 |    0 |   52.1s |
|           0.25 |   11 |     6 |     5 |    0 |   54.9s |
|           0.50 |   12 |     6 |     6 |    0 |   56.6s |
|           1.00 |   11 |     6 |     5 |    0 |   56.2s |
|           2.00 |   10 |     6 |     4 |    0 |   56.5s |
|           4.00 |   11 |     6 |     5 |    0 |   56.2s |

Pseudo-count 0.5 advanced as the unique screen leader.

### Confirmation

On 100 untouched paired games from opening offset 4,070,000, the unshrunk control won 46 games, split 25 Black and 21 White, with zero caps in 284.5 seconds. Pseudo-count 0.5 won 43, split 26 Black and 17 White, with zero caps in 285.6 seconds.

The candidate lost three games and regressed primarily as White. It is rejected without a second confirmation block. Child-Q pseudo-count zero remains accepted. The independent research control remains available for reproducibility. The checkpoint, browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Full-network sibling soups

### Hypothesis

Full-network symmetry-distillation seeds 260–262 used the same initialization, data, exact-QAT recipe, learning rate, and preservation objective. All three passed the original offline gate, but seed 260 alone advanced because it led the 20-game arena screen. Averaging sibling solutions could reduce seed-specific error without changing architecture, parameter count, browser payload, search, or inference cost.

Every candidate was averaged in float parameter space and then materialized through the exact INT8 deployment path before evaluation. The accepted seed-260 checkpoint remained the control.

### Equal-weight blends

Three blends were measured on the test splits of the promoted-model consensus corpus and two independent b18 corpora:

| Exact checkpoint            | Consensus loss / move / value | b18 3,300,000 loss / move / value | b18 3,500,000 loss / move / value |
| :-------------------------- | :---------------------------- | :-------------------------------- | :-------------------------------- |
| Accepted seed 260           | 2.2734 / 75.6% / 0.1214       | 2.9976 / 38.1% / 0.6045           | 2.9491 / 29.3% / 0.6076           |
| 50% seed 260 + 50% seed 261 | 2.2747 / 76.9% / 0.1233       | 2.9996 / 37.3% / 0.6041           | 2.9510 / 30.2% / 0.6063           |
| 50% seed 260 + 50% seed 262 | 2.2729 / 76.4% / 0.1217       | 2.9951 / 36.4% / 0.6018           | 2.9450 / 29.8% / 0.6069           |
| Equal seeds 260–262         | 2.2738 / 77.3% / 0.1222       | 2.9974 / 37.3% / 0.6024           | 2.9469 / 30.7% / 0.6061           |

The seed-260/261 blend regressed loss on every corpus and was rejected offline. The other two advanced to 20 fresh games from opening offset 4,080,000:

| Player                      | Wins | Black | White | Caps | Runtime |
| :-------------------------- | ---: | ----: | ----: | ---: | ------: |
| Accepted seed 260           |    8 |     4 |     4 |    0 |   57.1s |
| 50% seed 260 + 50% seed 262 |    7 |     4 |     3 |    0 |   57.2s |
| Equal seeds 260–262         |    8 |     3 |     5 |    0 |   56.5s |

Neither equal-weight blend uniquely improved the control.

### Shrunk sibling contribution

One bounded refinement reduced the sibling contribution from 50% to 25%. The first candidate used 75% seed 260 and 25% seed 262. The second used 75% seed 260 and 12.5% each of seeds 261 and 262.

| Exact checkpoint                               | Consensus loss / move / value | b18 3,300,000 loss / move / value | b18 3,500,000 loss / move / value |
| :--------------------------------------------- | :---------------------------- | :-------------------------------- | :-------------------------------- |
| Accepted seed 260                              | 2.2734 / 75.6% / 0.1214       | 2.9976 / 38.1% / 0.6045           | 2.9491 / 29.3% / 0.6076           |
| 75% seed 260 + 25% seed 262                    | 2.2732 / 75.6% / 0.1219       | 2.9961 / 38.1% / 0.6031           | 2.9477 / 29.3% / 0.6075           |
| 75% seed 260 + 12.5% seed 261 + 12.5% seed 262 | 2.2731 / 76.4% / 0.1199       | 2.9949 / 37.3% / 0.6021           | 2.9467 / 30.2% / 0.6062           |

Both restrained blends improved loss on every test corpus. On 20 fresh games from opening offset 4,090,000:

| Player                                         | Wins | Black | White | Caps | Runtime |
| :--------------------------------------------- | ---: | ----: | ----: | ---: | ------: |
| Accepted seed 260                              |    9 |     6 |     3 |    0 |   54.3s |
| 75% seed 260 + 25% seed 262                    |    9 |     6 |     3 |    0 |   54.3s |
| 75% seed 260 + 12.5% seed 261 + 12.5% seed 262 |   10 |     7 |     3 |    0 |   58.0s |

The three-seed restrained blend advanced as the unique screen leader. Its exact-dequantized checkpoint has SHA-256 `da5fad01a83d6e411c5a037ece169e5b47a12e4d8feb6ed7d83f49b60be9b996`.

### Confirmation

Two untouched paired 100-game blocks compared the frozen candidate with the accepted checkpoint:

| Opening offset | Control wins | Candidate wins | Control Black / White | Candidate Black / White | Control / candidate caps | Control / candidate runtime |
| -------------: | -----------: | -------------: | :-------------------- | :---------------------- | -----------------------: | --------------------------: |
|      4,100,000 |           38 |             42 | 23 / 15               | 23 / 19                 |                    0 / 0 |             279.9s / 276.7s |
|      4,110,000 |           47 |             44 | 27 / 20               | 24 / 20                 |                    0 / 1 |             287.4s / 422.8s |
|      **Total** |       **85** |         **86** | **50 / 35**           | **47 / 39**             |                **0 / 1** |         **567.3s / 699.5s** |

The candidate's four-game first-block gain reversed to a three-game loss in the second block. Its one-game aggregate edge was accompanied by a new 116-move unique-state cap and 23.3% greater total runtime. The result is indistinguishable from arena variance and fails the no-cap-regression gate.

All sibling soups are rejected. The accepted checkpoint, 104,129-parameter architecture, 111,920-byte browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Sibling prediction-ensemble distillation

### Hypothesis

Parameter soups indiscriminately averaged internal features and failed the arena gate. Prediction-space ensembling can preserve complementary sibling outputs without requiring their internal representations to align. The symmetry-target generator now accepts repeated checkpoints and aggregates every aligned policy and value across checkpoints and board symmetries. A three-checkpoint teacher therefore used 24 predictions per training position offline while retaining one unchanged student at inference.

Exact INT8 checkpoints for full-network seeds 260–262 generated three deterministic corpora:

| Opening offset | Positions | Compressed bytes | SHA-256                                                            |
| -------------: | --------: | ---------------: | :----------------------------------------------------------------- |
|      3,000,000 |     2,181 |        1,125,645 | `d2633e112aa8ca95e9275b92e979971dd98d26ebf2251f3e78e8c162cda76d63` |
|      3,100,000 |     2,154 |        1,099,115 | `3da5a4bc2009c49a087a12ba59dd9a601d6567f478a2e7570d9e7876ab7ee62a` |
|      3,500,000 |     2,145 |        1,040,466 | `b24d69d4599e01210d5cc4cc82f56d31f170bb6a3689c58aee6fbeafc5629a92` |

### Exact-QAT students

Seeds 280–282 started from the accepted float shadow and used one epoch, learning rate 0.000005, policy preservation weight 0.25, and exact INT8-aware training. Every candidate regressed offline:

| Test corpus                  | Accepted loss / move / value | Candidate range                             |
| :--------------------------- | :--------------------------- | :------------------------------------------ |
| Sibling ensemble 3,500,000   | 2.2747 / 75.1% / 0.1191      | 2.2954–2.2968 / 71.6%–72.0% / 0.1352–0.1370 |
| Seed-260 consensus 3,500,000 | 2.2734 / 75.6% / 0.1214      | 2.2935–2.2949 / 72.0%–72.4% / 0.1369–0.1388 |
| b18 3,300,000                | 2.9976 / 38.1% / 0.6045      | 3.0228–3.0254 / 37.3%–38.1% / 0.6023–0.6046 |
| b18 3,500,000                | 2.9491 / 29.3% / 0.6076      | 2.9963–2.9986 / 28.4% / 0.6180–0.6182       |

One bounded step-size check used seed 283 at learning rate 0.000001. It still regressed sibling-ensemble loss to 2.2778, seed-260 consensus loss to 2.2765, b18 3,300,000 loss to 3.0053, and b18 3,500,000 loss to 2.9593. Its value error also regressed on every corpus.

The smaller update did not rescue the objective, identifying the ensemble target rather than excessive step size as the problem. No candidate entered the arena. Multi-checkpoint target generation remains available for future independent teacher ensembles, but sibling prediction ensembling is rejected. The accepted checkpoint, browser artifact, search defaults, and Million website remain unchanged.

## 2026-07-28 — Evaluation-cache accounting

### Audit

The accepted descendant evaluator already caches exact feature states, but search counts tree simulations rather than model cache misses. On 20 games from opening offset 4,120,000, the descendant evaluator received 39,844 requests: 1,855 hits and 37,989 misses, a 4.66% hit rate. The separate real-root evaluator received 685 requests and zero hits.

The arena had retained both evaluator caches across separate games. That is valid memoization but does not reproduce a standalone browser game, which begins without positions from earlier matches. Clearing both caches at each game boundary reduced the descendant hit rate to 3.38% on 20 fresh games from opening offset 4,130,000: 1,179 hits among 34,873 requests. Root hits remained zero. Per-game cache isolation is retained for faithful research timing and prevents paired games from lending inference work to one another.

### Fixed evaluation budget

A disabled research control tracked actual descendant cache misses. After the ordinary 64 simulations, it could add a bounded number of simulations only while the move remained below its original miss budget. Cached and terminal leaves could therefore contribute extra tree visits without another model evaluation.

On 20 fresh games from opening offset 4,140,000:

| Maximum free simulations | Wins | Black | White | Caps | Runtime |
| -----------------------: | ---: | ----: | ----: | ---: | ------: |
|                        0 |    9 |     4 |     5 |    0 |   84.2s |
|                        4 |    8 |     4 |     4 |    0 |   83.4s |
|                        8 |    8 |     4 |     4 |    0 |   87.2s |

Both candidates lost one completed game and were rejected.

### Shared root evaluator

Accepted root and descendant evaluation currently use the same eight symmetries, policy temperature, geometric policy aggregation, value aggregation, and checkpoint. A second disabled control shared their cache only when every output-affecting setting matched. When the real root had already been evaluated as a descendant, the saved root call funded exactly one replacement simulation.

On 20 fresh games from opening offset 4,150,000, control and shared-root search both won 10 games, split five Black and five White, with zero caps. Runtime changed from 76.5 to 81.3 seconds. Exact root reuse tied rather than improved play and is rejected.

Both controls remain disabled for reproducibility. Cache isolation remains enabled in the arena. The accepted model and browser artifact remain unchanged.

## 2026-07-28 — Search-aware causal regret diagnostics

The diagnostic arena now supports the accepted 64-simulation search rather than only raw policy play. It records the search policy, searched root value, teacher-policy agreement, and symmetry value spread at every Moka decision.

The previous diagnostic defined move loss as KataGo's parent value minus its value after Moka's move. That metric could report large loss even when Moka selected KataGo's own top-policy move because the teacher value head is not exactly consistent across parent and child evaluations. The corrected metric evaluates both children and measures KataGo's value after its top-policy move minus its value after Moka's actual move. Teacher-matching moves therefore have exactly zero action regret.

Ten games from opening offset 4,160,000 produced 417 Moka decisions and four wins:

| Phase   | Decisions | Teacher move agreement | Mean causal regret | Maximum causal regret | Regret at least 0.2 |
| :------ | --------: | ---------------------: | -----------------: | --------------------: | ------------------: |
| Opening |        80 |                  58.8% |             0.0179 |                0.2979 |                   2 |
| Middle  |       150 |                  62.7% |             0.0179 |                0.7106 |                  13 |
| Endgame |       187 |                  58.3% |             0.0078 |                0.9210 |                   3 |

Only 18 of 417 decisions had regret of at least 0.2. Symmetry value spread did not identify them:

| Metric                             |  Result |
| :--------------------------------- | ------: |
| Mean spread, critical decisions    |  0.1150 |
| Mean spread, other decisions       |  0.1162 |
| Spread–regret correlation          | -0.0006 |
| Critical ROC AUC                   |   0.508 |
| Highest-spread 20% critical recall |   5.56% |

Value-spread uncertainty is therefore rejected before runtime intervention. The corrected causal diagnostic and raw spread measurement remain available for future objectives.

## 2026-07-28 — Full-symmetry opponent reply width

### Hypothesis

Opponent reply pruning previously improved high-visit search, but its low-budget calibration predated full descendant symmetry and FPU reduction 0.5. Restricting only opponent-to-move nodes to Moka's highest-prior replies can concentrate the fixed 64 simulations while leaving every Moka-to-move branch legal. It uses no teacher query, benchmark label, rule heuristic, extra model, or additional inference.

### Screen

The exact accepted artifact, 64 simulations, full root and descendant symmetry, geometric policy weight 0.125, exploration 2.0, value weight 1.25, FPU reduction 0.5, maximum-visit selection, and margin-60 resignation remained fixed. On 20 fresh paired games from opening offset 4,170,000:

| Opponent branch width | Wins | Black | White | Caps | Runtime |
| --------------------: | ---: | ----: | ----: | ---: | ------: |
|                  Full |    7 |     5 |     2 |    0 |   92.5s |
|                     4 |   10 |     7 |     3 |    0 |   90.7s |
|                     8 |    7 |     4 |     3 |    0 |   92.7s |
|                    16 |    8 |     6 |     2 |    0 |   97.2s |

Width 4 advanced as the unique screen leader.

### Confirmation

Two untouched paired 100-game blocks compared the frozen width with full branching:

| Opening offset | Full wins | Width-4 wins | Full Black / White | Width-4 Black / White | Full / width-4 caps | Full / width-4 runtime |
| -------------: | --------: | -----------: | :----------------- | :-------------------- | ------------------: | ---------------------: |
|      4,180,000 |        41 |           42 | 19 / 22            | 19 / 23               |               0 / 0 |        399.1s / 431.0s |
|      4,190,000 |        46 |           50 | 21 / 25            | 23 / 27               |               0 / 0 |        407.6s / 428.4s |
|      **Total** |    **87** |       **92** | **40 / 47**        | **42 / 50**           |           **0 / 0** |    **806.7s / 859.4s** |

Width 4 improved both independent blocks, adding two Black wins and three White wins in aggregate without a cap. Total runtime increased 6.5% because its games ran longer despite the narrower opponent tree.

Opponent branch width 4 is accepted as the research default. The 104,129-parameter checkpoint, 111,920-byte browser artifact, and their digests remain unchanged. The Million website remains untouched.

The promoted default and explicit `--opponent-branches 4` reproduced the same two-game result at opening offset 4,200,000: one Moka win as White, one KataGo win, four Moka passes, two KataGo passes, and zero caps.

## 2026-07-28 — Local opponent-width refinement

Widths immediately around the accepted opponent reply width were screened on 20 fresh games from opening offset 4,210,000:

| Opponent branch width | Wins | Black | White | Caps |
| --------------------: | ---: | ----: | ----: | ---: |
|                     2 |   10 |     4 |     6 |    0 |
|                     3 |    9 |     4 |     5 |    0 |
|                     4 |    7 |     4 |     3 |    0 |
|                     5 |    8 |     5 |     3 |    0 |
|                     6 |    8 |     4 |     4 |    0 |

Width 2 advanced as the unique screen leader. Two disjoint 100-game blocks compared it with the accepted width 4:

| Opening offset | Width-4 wins | Width-2 wins | Width-4 Black / White | Width-2 Black / White | Width-4 / width-2 caps |
| -------------: | -----------: | -----------: | :-------------------- | :-------------------- | ---------------------: |
|      4,220,000 |           35 |           38 | 22 / 13               | 21 / 17               |                  1 / 1 |
|      4,230,000 |           49 |           47 | 26 / 23               | 27 / 20               |                  0 / 0 |
|      **Total** |       **84** |       **85** | **48 / 36**           | **48 / 37**           |              **1 / 1** |

The three-game first-block gain reversed to a two-game loss in the independent second block. A one-game aggregate difference over 200 games is indistinguishable from arena variance, so width 2 is rejected. Opponent width 4 remains accepted.

### Early-game width schedule

A phase schedule retained width 4 normally and used width 2 only before a fixed move threshold. On 20 fresh games from opening offset 4,240,000:

| Early width-2 end move | Wins | Black | White | Caps |
| ---------------------: | ---: | ----: | ----: | ---: |
|       Disabled control |   11 |     6 |     5 |    0 |
|                     20 |    8 |     4 |     4 |    0 |
|                     30 |    9 |     4 |     5 |    0 |
|                     40 |    9 |     5 |     4 |    0 |
|                     50 |    7 |     4 |     3 |    0 |

Every schedule lost at least two completed games. Early width 2 is rejected, and its unused runtime path was removed.

### PUCT value weight under width 4

The policy–value balance was retuned because opponent reply pruning changed the tree topology. On 20 fresh games from opening offset 4,250,000:

| Value weight | Wins | Black | White | Caps |
| -----------: | ---: | ----: | ----: | ---: |
|        1.000 |   11 |     8 |     3 |    0 |
|        1.125 |    8 |     5 |     3 |    0 |
|        1.250 |   12 |     8 |     4 |    0 |
|        1.375 |    8 |     6 |     2 |    0 |
|        1.500 |    9 |     7 |     2 |    0 |

The accepted value weight 1.25 remained the unique leader. No candidate advanced.

### PUCT exploration under width 4

Exploration was retuned with opponent width 4 and value weight 1.25 fixed. On 20 fresh games from opening offset 4,260,000:

| Exploration | Wins | Black | White | Caps |
| ----------: | ---: | ----: | ----: | ---: |
|        1.50 |   11 |     4 |     7 |    0 |
|        1.75 |   12 |     7 |     5 |    0 |
|        2.00 |   11 |     6 |     5 |    0 |
|        2.25 |    9 |     6 |     3 |    0 |
|        2.50 |    8 |     5 |     3 |    0 |

Exploration 1.75 advanced as the unique screen leader. Two disjoint 100-game blocks produced:

| Opening offset | Exploration-2 wins | Exploration-1.75 wins | Control Black / White | Candidate Black / White | Control / candidate caps | Control / candidate runtime |
| -------------: | -----------------: | --------------------: | :-------------------- | :---------------------- | -----------------------: | --------------------------: |
|      4,270,000 |                 33 |                    37 | 16 / 17               | 22 / 15                 |                    0 / 0 |             302.1s / 305.3s |
|      4,280,000 |                 38 |                    40 | 21 / 17               | 21 / 19                 |                    0 / 0 |             287.7s / 287.8s |
|      **Total** |             **71** |                **77** | **37 / 34**           | **43 / 34**             |                **0 / 0** |         **589.8s / 593.1s** |

Exploration 1.75 improved both independent blocks, added six Black wins while preserving White wins, and introduced no cap. Runtime increased 0.6%. It is accepted as the research default. The checkpoint, 111,920-byte browser artifact, and Million website remain unchanged.

The promoted default and explicit `--search-exploration 1.75` reproduced the same two-game result at opening offset 4,290,000, including six Moka passes, two KataGo passes, and zero caps.

## 2026-07-28 — FPU under exploration 1.75

First-play urgency was retuned because exploration 1.75 changed the relative prior bonus assigned to unvisited children. On 20 fresh games from opening offset 4,300,000:

| FPU reduction | Wins | Black | White | Caps |
| ------------: | ---: | ----: | ----: | ---: |
|         0.250 |    7 |     3 |     4 |    0 |
|         0.375 |    6 |     2 |     4 |    0 |
|         0.500 |    8 |     2 |     6 |    0 |
|         0.625 |    9 |     4 |     5 |    0 |
|         0.750 |   10 |     5 |     5 |    0 |

Reduction 0.75 advanced as the unique screen leader. On 100 untouched games from opening offset 4,310,000, the accepted reduction 0.5 won 52 games, split 26 Black and 26 White, with zero caps in 280.9 seconds. Reduction 0.75 won 48, split 24 Black and 24 White, with zero caps in 277.1 seconds.

The candidate lost four completed games and regressed equally as both colors. It is rejected. FPU reduction 0.5 remains accepted.

## 2026-07-28 — Opponent width under exploration 1.75

Opponent width was rescreened because exploration 1.75 changed how quickly PUCT exhausts the retained reply set. On 20 fresh games from opening offset 4,320,000:

| Opponent width | Wins | Black | White | Caps |
| -------------: | ---: | ----: | ----: | ---: |
|              2 |   10 |     6 |     4 |    0 |
|              3 |   14 |     8 |     6 |    0 |
|              4 |   14 |     8 |     6 |    0 |
|              5 |    9 |     5 |     4 |    0 |
|              6 |   11 |     6 |     5 |    0 |

Widths 3 and 4 tied exactly for the lead, including their color split and cap count. Selecting the smaller width after observing a tie would add selection bias, so no candidate advanced. Opponent width 4 remains accepted.

## 2026-07-28 — Root symmetry blend under accepted pruning

The geometric root-policy blend was retuned under width 4 and exploration 1.75. On 20 fresh games from opening offset 4,330,000:

| Geometric weight | Wins | Black | White | Caps |
| ---------------: | ---: | ----: | ----: | ---: |
|           0.0000 |    8 |     3 |     5 |    0 |
|           0.0625 |    8 |     3 |     5 |    0 |
|           0.1250 |    8 |     3 |     5 |    0 |
|           0.1875 |    9 |     4 |     5 |    0 |
|           0.2500 |    9 |     4 |     5 |    0 |

Weights 0.1875 and 0.25 tied exactly for the lead. No candidate advanced after the tie. The accepted root geometric weight remains 0.125.

## 2026-07-28 — Descendant symmetry blend under accepted pruning

The descendant geometric-policy blend was screened independently because it controls every evaluated leaf prior rather than only the real root. On 20 fresh games from opening offset 4,340,000:

| Geometric weight | Wins | Black | White | Caps |
| ---------------: | ---: | ----: | ----: | ---: |
|           0.0000 |   11 |     3 |     8 |    0 |
|           0.0625 |   11 |     3 |     8 |    0 |
|           0.1250 |   12 |     5 |     7 |    0 |
|           0.1875 |   11 |     5 |     6 |    0 |
|           0.2500 |   10 |     4 |     6 |    0 |

The accepted descendant geometric weight 0.125 remained the unique leader. No candidate advanced. This cycle therefore retains FPU reduction 0.5, opponent width 4, root geometric weight 0.125, and descendant geometric weight 0.125. The model, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Causal reanalysis selection

### Hypothesis

The existing reanalysis selector measures the value drift from the parent position to Moka's played child. A more causal selector can compare the stronger b18 teacher's preferred child directly with Moka's actual child. This should prioritize decisions where the teacher sees a concrete better alternative rather than positions whose value merely changed after a move.

### Controlled corpus

Both selectors used the same 128 Moka-versus-b6 games, seed 284, opening offset 4,350,000, Moka turns only, 25% selection rate, 128 b18 visits, and 1,034 retained roots. The causal corpus has SHA-256 `b8fd7ef0bbba77e10ab3d5c40ecda305b26aa6a3b387d1400e53590501255ebb`; the legacy corpus has SHA-256 `467a38088c2a53bb8e7b4a4ff9474235325347c63e92ebc64c8eb60c24285626`.

| Diagnostic           | Causal selector | Existing selector |
| :------------------- | --------------: | ----------------: |
| b18 coverage         |          86.94% |            88.59% |
| b18 top-move match   |          32.11% |            38.97% |
| Regret at least 0.05 |           8.80% |             9.86% |
| Regret at least 0.10 |           6.77% |             7.93% |
| Regret at least 0.20 |           5.03% |             5.90% |
| Regret at least 0.40 |           2.90% |             3.77% |
| Mean finite regret   |         0.04885 |           0.05495 |
| Winner flips         |              19 |                24 |

The selections overlapped on 697 roots; each selector contributed roughly 330 unique roots, for Jaccard overlap 0.510. Despite its cleaner interpretation, the causal score was worse on every stronger-teacher diagnostic. It is rejected before training. The generated dataset now records each root's selection score so future reanalysis corpora remain auditable without changing training input.

## 2026-07-28 — Opponent progressive widening

### Hypothesis

The accepted fixed opponent width 4 spends visits across all retained replies immediately. Prior-guided progressive widening can retain the same four replies but initially expose only the highest-prior two or three, unlocking the remainder as the node earns visits. This uses the existing model and fixed 64-evaluation budget without teacher access or extra inference.

### Screen

The exact accepted artifact and all accepted search settings were fixed. Five schedules used the same 20 fresh paired games from opening offset 4,360,000:

| Initial replies | Visits per unlocked reply | Wins | Black | White | Caps |
| --------------: | ------------------------: | ---: | ----: | ----: | ---: |
|               4 |                  Disabled |   10 |     7 |     3 |    0 |
|               2 |                         2 |   11 |     8 |     3 |    0 |
|               2 |                         4 |   11 |     7 |     4 |    0 |
|               2 |                         8 |   10 |     6 |     4 |    0 |
|               3 |                         4 |   10 |     7 |     3 |    0 |

Two candidates tied one game above control with different color splits. Neither was a unique screen winner, and selecting between them after observing the tie would add selection bias. Progressive widening is rejected and its unused runtime path is removed. Fixed opponent width 4 remains accepted; the checkpoint and browser artifact remain unchanged.

## 2026-07-28 — Descendant exploration under opponent pruning

Opponent width 4 and root exploration 1.75 changed the accepted tree after the last root/descendant exploration audit. The split was rescreened on the same exact artifact, fixed 64-evaluation budget, and 20 fresh paired games from opening offset 4,370,000:

| Descendant exploration | Wins | Black | White | Caps |
| ---------------------: | ---: | ----: | ----: | ---: |
|                   1.00 |    5 |     2 |     3 |    0 |
|                   1.25 |    9 |     4 |     5 |    0 |
|                   1.50 |   11 |     6 |     5 |    0 |
|         Inherited 1.75 |   11 |     7 |     4 |    0 |
|                   2.00 |   10 |     7 |     3 |    0 |

The only challenger tied the accepted player while trading one Black win for one White win. No candidate advanced. Descendant exploration continues to inherit the accepted root coefficient 1.75.

## 2026-07-28 — Visit budget under opponent pruning

The fixed budget was last calibrated before opponent width 4 and exploration 1.75 changed tree allocation. Budgets around the accepted 64 evaluations were screened on 20 fresh paired games from opening offset 4,380,000:

| Evaluations | Wins | Black | White | Caps | Runtime |
| ----------: | ---: | ----: | ----: | ---: | ------: |
|          48 |    8 |     4 |     4 |    0 |   58.5s |
|          56 |    6 |     3 |     3 |    0 |   66.8s |
|          64 |   10 |     4 |     6 |    0 |   78.0s |
|          72 |    7 |     2 |     5 |    0 |   79.6s |
|          80 |   10 |     5 |     5 |    0 |   87.6s |

Eighty evaluations tied the accepted player while taking 12.3% longer on the same games. Every other candidate lost at least two completed games. Fixed 64-evaluation search remains accepted.

## 2026-07-28 — Terminal evaluation regression audit

Commit `b1e437e` replaced Tromp-Taylor area outcomes at terminal search nodes with Moka's learned value, and used the same estimate to decide whether to accept an opponent pass. The intent was to approximate dead-stone adjudication, but the arena still scores completed games by area. This made search optimize a different terminal rule from the one that determines wins.

The exact accepted artifact and search settings compared the two implementations on identical fresh paired openings:

| Opening offset | Games | Area-terminal wins | Network-terminal wins | Area Black / White | Network Black / White | Area / network caps |
| -------------: | ----: | -----------------: | --------------------: | :----------------- | :-------------------- | ------------------: |
|      4,390,000 |    20 |                 10 |                     9 | 5 / 5              | 4 / 5                 |               0 / 0 |
|      4,400,000 |   100 |                 42 |                    26 | 21 / 21            | 12 / 14               |               0 / 0 |

The learned-terminal player lost 16 completed games on the frozen 100-game block and regressed as both colors. Terminal search, rollout adjudication, and opponent-pass acceptance are restored to the arena's area rule. The separate dead-stone removal API remains available for user-approved adjudication before scoring; it is not inferred from the compact network. The model and browser artifact remain unchanged.

## 2026-07-28 — Current-model global output adapter

### Hypothesis

The zero-initialized global pooling adapter had only been tested on a much older Moka. It adds 1,432 parameters, initially reproduces the accepted model exactly, and lets global board summaries correct policy logits and value without modifying the local trunk. Re-auditing it on the accepted symmetry-distilled checkpoint tests whether current local features can support a small global correction.

### Training

Three adapter-only candidates used the same b18-labeled Moka positions from opening offsets 3,000,000, 3,100,000, and 3,500,000, three epochs, batch size 256, policy-preservation weight 0.25, and learning rates 0.0001, 0.0003, and 0.001. The accepted 104,129-parameter checkpoint and zero-initialized 105,561-parameter adapter produced exactly equal policy logits and values before training.

The 0.001 candidate preserved held-out top-move agreement and improved value MAE from 0.6076 to 0.5963, but worsened policy loss from 2.9491 to 2.9925. Lower-rate candidates regressed more offline, leaving the MCTS tradeoff ambiguous.

### Arena

On 20 fresh games from opening offset 4,420,000:

| Player         | Wins | Black | White | Caps |
| :------------- | ---: | ----: | ----: | ---: |
| Incumbent      |    7 |     3 |     4 |    0 |
| Adapter 0.0001 |   10 |     4 |     6 |    0 |
| Adapter 0.0003 |    7 |     2 |     5 |    0 |
| Adapter 0.001  |    8 |     5 |     3 |    0 |

The lowest-rate adapter was frozen as the unique screen winner. Two untouched 100-game blocks produced:

| Opening offset | Incumbent wins | Adapter wins | Incumbent Black / White | Adapter Black / White | Incumbent / adapter caps | Incumbent / adapter runtime |
| -------------: | -------------: | -----------: | :---------------------- | :-------------------- | -----------------------: | --------------------------: |
|      4,430,000 |             39 |           38 | 18 / 21                 | 18 / 20               |                    0 / 0 |             293.9s / 313.0s |
|      4,440,000 |             45 |           48 | 22 / 23                 | 25 / 23               |                    0 / 0 |             292.3s / 306.4s |
|      **Total** |         **84** |       **86** | **40 / 44**             | **43 / 43**           |                **0 / 0** |         **586.2s / 619.4s** |

The candidate's first-block loss reversed to a three-game gain. Its two-game aggregate edge is smaller than the block swing, and runtime increased 5.7%. The output adapter is rejected. The accepted architecture, checkpoint, browser artifact, and Million website remain unchanged.

## 2026-07-28 — Quantized global residual context

### Hypothesis

The rejected output adapter can only correct the final policy and value heads after local features have already been compressed. KataGo instead injects pooled global context inside its residual trunk. Moka can approximate that method with three zero-initialized adapters after blocks 4, 8, and 12. Each adapter pools the bottleneck's spatial means and maxima, projects 32 pooled values through eight hidden channels, and broadcasts a 16-channel bias before the block's second spatial convolution.

The architecture adds 1,224 parameters, from 104,129 to 105,353. Loading the accepted nested checkpoint into it produces exactly equal logits and values before training. KataGo is used only for offline labels; every arena and browser evaluation uses Moka's own compact model.

### Quantization audit

The accepted float-shadow checkpoint matches only 51.1% of the current model's symmetry-consensus top moves on the held-out offset-3,500,000 bucket, while its exact INT8 materialization matches 75.6%. Training the new adapters in float and quantizing afterward therefore optimizes the wrong player. The adapter-only trainer now permits the existing exact INT8 straight-through path: the complete incumbent is quantized before every forward pass while gradients update only the 12 new adapter tensors.

Adapter-only consensus training was stable under QAT but did not beat the incumbent's held-out loss. It was rejected offline. Direct b18 QAT improved held-out policy loss and value error at all four screened rates:

| Learning rate | Test loss | Top-move agreement | Value MAE |
| ------------: | --------: | -----------------: | --------: |
|     Incumbent |    2.9491 |              29.3% |    0.6076 |
|       0.00001 |    2.9435 |              28.0% |    0.6050 |
|       0.00003 |    2.9387 |              28.9% |    0.6030 |
|       0.00010 |    2.9356 |              28.0% |    0.5968 |
|       0.00030 |    2.9431 |              27.6% |    0.5803 |

The candidates used the same 5,167 training rows from the 3,000,000, 3,100,000, and 3,500,000 b18 corpora, three epochs, batch size 256, policy-preservation weight 0.25, and seeds 306–309. No arena result changed their recipe.

### Arena

On 20 fresh paired games from opening offset 4,450,000:

| Player          | Wins | Black | White | Caps | Runtime |
| :-------------- | ---: | ----: | ----: | ---: | ------: |
| Incumbent       |    9 |     7 |     2 |    0 |   62.1s |
| Adapter 0.00001 |   11 |     6 |     5 |    0 |   66.6s |
| Adapter 0.00003 |    7 |     4 |     3 |    1 |   70.1s |
| Adapter 0.00010 |    6 |     5 |     1 |    1 |   69.2s |
| Adapter 0.00030 |   10 |     6 |     4 |    0 |   72.5s |

The smallest update was frozen as the unique cap-safe screen winner. Two untouched 100-game blocks produced:

| Opening offset | Incumbent wins | Candidate wins | Incumbent Black / White | Candidate Black / White | Incumbent / candidate caps | Incumbent / candidate runtime |
| -------------: | -------------: | -------------: | :---------------------- | :---------------------- | -------------------------: | ----------------------------: |
|      4,460,000 |             39 |             49 | 22 / 17                 | 27 / 22                 |                      0 / 0 |               308.5s / 330.0s |
|      4,470,000 |             39 |             46 | 18 / 21                 | 24 / 22                 |                      0 / 0 |               294.7s / 322.8s |
|      **Total** |         **78** |         **95** | **40 / 38**             | **51 / 44**             |                  **0 / 0** |           **603.2s / 652.8s** |

The candidate improves both independent blocks, adds 11 Black and six White wins, and introduces no cap. Python arena runtime increases 8.2%, partly from longer games. In the actual JavaScript runtime, 500 alternating evaluations are effectively tied: 4.160 ms for the incumbent and 4.165 ms for the candidate.

### Deployment

The promoted browser binary is 113,648 bytes and 100,666 bytes under deterministic gzip, up 1,728 raw bytes. Seven deterministic positions match the exact MLX checkpoint on every top move with finite outputs; numerical differences remain within the incumbent runtime's existing envelope. The promoted digests are:

- float-shadow checkpoint: `9be23c3d60b8ba70ff4a8456704b921e82f0a691ec3e7d305b9830b671687d27`;
- exact INT8 checkpoint: `be72b4e8068cc66788597979ed10414dc3a59f70d3ba19fc817e66302ff27f0a`;
- browser binary: `d808d09f4b9dab959fc2764a16485448ae407bbd90cbdf8de6e8e8605b2c2de9`;
- browser manifest: `94ada54b21e63ca44a8ca8669fbd0ab0ed702942eff060feb31f2f83b51a4cad`.

The global-residual checkpoint and browser artifact are promoted. The accepted 64-evaluation search remains unchanged. The Million website remains untouched.

## 2026-07-28 — Exact INT8 preservation and deeper global-context labels

### Quantized preservation correction

Quantization-aware training previously compared each candidate with the float-shadow reference in its policy-preservation term. That reference is not the deployed player: on the held-out offset-3,500,000 symmetry-consensus bucket, the old float shadow matched only 51.1% of the exact INT8 player's top moves. The trainer now quantizes the frozen reference before computing preservation loss whenever exact INT8-aware training is enabled.

The successful global-residual recipe was replayed from the same seed-260 initialization, data, seed 306, learning rate 0.00001, and three-epoch schedule with only this correction. The replay improved exact held-out loss over the promoted candidate on all seven inspected b18 and symmetry corpora. On 20 fresh paired games from opening offset 4,480,000, however, both players won nine games with zero caps. The promoted model split 4 Black / 5 White in 58.9 seconds; the replay split 5 Black / 4 White in 63.5 seconds. The replay is rejected as a tie.

### Separate 128- and 256-visit labels

Three incremental adapter-only candidates started from the promoted model. They used 2,489 training rows from the offset-3,300,000 128-visit auxiliary corpus, the offset-3,400,000 root and branch corpora, and the offset-3,200,000 256-visit corpus. Each used two epochs, exact INT8 preservation, policy-preservation weight 0.25, and a predeclared learning rate:

| Learning rate | b18 3,000,000 loss | b18 3,100,000 loss | b18 3,500,000 loss | Consensus top move |
| ------------: | -----------------: | -----------------: | -----------------: | -----------------: |
|       Control |             2.8980 |             2.7451 |             2.9435 |              75.6% |
|      0.000003 |             2.8937 |             2.7414 |             2.9380 |              75.1% |
|      0.000010 |             2.8867 |             2.7366 |             2.9306 |              76.0% |
|      0.000030 |             2.8842 |             2.7351 |             2.9279 |              75.1% |

Every candidate also lowered loss on the separate 128- and 256-visit test splits. All three entered the fixed 20-game screen at opening offset 4,490,000:

| Player       | Wins | Black | White | Caps | Runtime |
| :----------- | ---: | ----: | ----: | ---: | ------: |
| Control      |    9 |     6 |     3 |    1 |   57.2s |
| Adapter 3e-6 |    8 |     6 |     2 |    0 |   55.7s |
| Adapter 1e-5 |    8 |     5 |     3 |    0 |   57.7s |
| Adapter 3e-5 |    8 |     7 |     1 |    0 |   55.8s |

The deeper-label family is rejected without confirmation. Exact preservation remains in the trainer because it aligns the regularizer with the deployed player, but no checkpoint or browser artifact changes.

## 2026-07-28 — Global-context search recalibration

The promoted global-residual model changes policy and value estimates inside search, so the two most important PUCT coefficients were re-audited without combining changes. Every run used the exact promoted INT8 checkpoint, 64 evaluations, full root and descendant symmetry, opponent width 4, FPU reduction 0.5, and the accepted area-scored terminal rule.

### Exploration

Five exploration coefficients used the same 20 paired games from opening offset 4,500,000:

| Exploration | Wins | Black | White | Caps |
| ----------: | ---: | ----: | ----: | ---: |
|        1.25 |    7 |     3 |     4 |    0 |
|        1.50 |   10 |     4 |     6 |    0 |
|        1.75 |    6 |     3 |     3 |    0 |
|        2.00 |    6 |     3 |     3 |    1 |
|        2.25 |    8 |     3 |     5 |    0 |

Exploration 1.50 was frozen as the unique screen winner. Two untouched confirmation blocks produced:

| Opening offset | Exploration 1.75 | Exploration 1.50 | Control Black / White | Candidate Black / White | Control / candidate caps | Control / candidate runtime |
| -------------: | ---------------: | ---------------: | :-------------------- | :---------------------- | -----------------------: | --------------------------: |
|      4,510,000 |               40 |               45 | 20 / 20               | 20 / 25                 |                    0 / 1 |             301.7s / 306.3s |
|      4,520,000 |               38 |               37 | 19 / 19               | 20 / 17                 |                    0 / 0 |             307.7s / 309.4s |
|      **Total** |           **78** |           **82** | **39 / 39**           | **40 / 42**             |                **0 / 1** |         **609.4s / 615.7s** |

The five-game first-block gain reversed to a one-game loss. The candidate also introduced a capped loss. Exploration 1.50 is rejected; 1.75 remains accepted.

### Value weight

With exploration restored to 1.75, five value weights used 20 fresh games from opening offset 4,530,000:

| Value weight | Wins | Black | White | Caps |
| -----------: | ---: | ----: | ----: | ---: |
|        1.000 |    9 |     3 |     6 |    0 |
|        1.125 |    6 |     2 |     4 |    0 |
|        1.250 |    6 |     2 |     4 |    0 |
|        1.375 |   10 |     4 |     6 |    0 |
|        1.500 |   12 |     6 |     6 |    0 |

Value weight 1.50 was the unique screen winner. Its two frozen confirmation blocks produced:

| Opening offset | Value 1.25 | Value 1.50 | Control Black / White | Candidate Black / White | Control / candidate caps | Control / candidate runtime |
| -------------: | ---------: | ---------: | :-------------------- | :---------------------- | -----------------------: | --------------------------: |
|      4,540,000 |         41 |         45 | 27 / 14               | 31 / 14                 |                    0 / 0 |             467.5s / 468.1s |
|      4,550,000 |         51 |         48 | 29 / 22               | 27 / 21                 |                    0 / 0 |             442.1s / 439.6s |
|      **Total** |     **92** |     **93** | **56 / 36**           | **58 / 35**             |                **0 / 0** |         **909.6s / 907.7s** |

The four-game first-block gain reversed to a three-game loss. A one-game aggregate edge with opposite block directions is not evidence of stronger play. Value weight 1.50 is rejected; 1.25 remains accepted. The promoted global-residual model and browser artifact remain unchanged.

## 2026-07-28 — Child-Q policy ranking

### Hypothesis

The previous direct-teacher updates repeatedly lowered average policy loss without improving play. Dense cross-entropy treats every probability error as relevant, while MCTS is especially sensitive to whether materially worse actions outrank the teacher's preferred action. A targeted pairwise objective can protect this decision structure without adding an inference head or changing browser cost.

For each position, the objective takes b18's most-visited move and compares it only with child actions that:

- have at least two b18 visits;
- are at least 0.05 root-value worse than the preferred move.

The loss is an advantage-weighted logistic ranking loss over the existing policy logits. It activates on 22.5%, 25.9%, and 28.0% of the offset-3,000,000, 3,100,000, and 3,500,000 corpora, respectively, with about two material comparisons per active position. KataGo values are used only during offline training.

### Training and offline gate

Three global-residual adapter-only candidates started from the promoted checkpoint. They used the same 5,167 training rows, three epochs, learning rate 0.00001, seed 320, exact INT8 preservation weight 0.25, and ranking weights 0.05, 0.10, and 0.20.

| Player      | b18 3,000,000 loss / Q-rank | b18 3,100,000 loss / Q-rank | b18 3,500,000 loss / Q-rank | Consensus top move |
| :---------- | --------------------------: | --------------------------: | --------------------------: | -----------------: |
| Control     |             2.8980 / 0.7545 |             2.7451 / 0.8797 |             2.9435 / 0.9381 |              75.6% |
| Q-rank 0.05 |             2.8830 / 0.7465 |             2.7331 / 0.8703 |             2.9233 / 0.9192 |              75.1% |
| Q-rank 0.10 |             2.8831 / 0.7465 |             2.7329 / 0.8700 |             2.9232 / 0.9190 |              75.1% |
| Q-rank 0.20 |             2.8831 / 0.7463 |             2.7327 / 0.8694 |             2.9232 / 0.9190 |              75.1% |

All candidates also lowered standard b18 loss on the independent 128- and 256-visit test splits. The fixed 20-game screen used opening offset 4,560,000:

| Player      | Wins | Black | White | Caps |
| :---------- | ---: | ----: | ----: | ---: |
| Control     |    8 |     4 |     4 |    0 |
| Q-rank 0.05 |   10 |     6 |     4 |    0 |
| Q-rank 0.10 |    9 |     6 |     3 |    0 |
| Q-rank 0.20 |   10 |     6 |     4 |    0 |

Weights 0.05 and 0.20 tied exactly for the lead, including color split and cap count. Selecting one after observing the tie would add bias, so neither advances. The child-Q objective remains available for reproducible future experiments, but the checkpoint and browser artifact remain unchanged.
