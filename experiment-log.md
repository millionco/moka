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
