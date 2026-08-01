# Moka strength-research handoff

## Objective

Improve Moka's real Go strength without hidden teacher access or rule cheating. Use Moka's own policy/value network and legitimate MCTS test-time compute. Promote only gains that reproduce on paired openings and both colors.

## Repositories

- Moka: `https://github.com/millionco/moka.git`
- Website: `https://github.com/millionco/million-website.git`
- Research branch: `codex/moka-value-search`

The current research belongs only in Moka. Do not modify or deploy the Million website until a candidate passes every offline, arena, confirmation, exact-INT8, payload, and browser gate.

## Clean-Mac setup

```sh
mkdir -p ~/Developer
cd ~/Developer
git clone https://github.com/millionco/moka.git
cd moka
git fetch origin codex/moka-value-search
git switch --track -c codex/moka-value-search origin/codex/moka-value-search
uv sync
ni
mkdir -p checkpoints data teachers
cp handoff-assets/moka-global-soup-exact-q50-int8-roundtrip.safetensors checkpoints/
cp handoff-assets/moka-current-search256-offset9550k-64.npz data/
cp handoff-assets/moka-global-soup-search128-offset5400k-128.npz data/
cp handoff-assets/moka-global-b18-onpolicy-offset4720k-64x128.npz data/
cp handoff-assets/katago-b6c96.onnx teachers/
nr check
uv run python tests/search.py

cd ~/Developer
git clone https://github.com/millionco/million-website.git
cd million-website
ni
git switch main
git pull --ff-only origin main
```

Install `uv`, Node.js, pnpm, and `@antfu/ni` first if the clean machine lacks them. Use `ni` and `nr`; do not substitute npm or pnpm commands inside either repository.

## Required assets

| Asset                                                   | SHA-256                                                            |
| :------------------------------------------------------ | :----------------------------------------------------------------- |
| `katago-b6c96.onnx`                                     | `e921df41ea38f69e622ad779295661126d7012807b051fac286191f1f118388b` |
| `moka-global-soup-exact-q50-int8-roundtrip.safetensors` | `90db3d02bb1fe3f850c32b6c4b5b864f049220d1e05d128c4efd686dd5b0d954` |
| `moka-current-search256-offset9550k-64.npz`             | `3054722bc167ee6b51f8649bc629e5853d67293721c2d7f79a02bb881a84b885` |
| `moka-global-soup-search128-offset5400k-128.npz`        | `6d000bba283dc48a42864abd6aa0a571f64af42dad04e50548abcfbaa4b3118e` |
| `moka-global-b18-onpolicy-offset4720k-64x128.npz`       | `e263dfae8829d8b9bd195ce04057760598113b3dfa6d31bb2495a0a431739f10` |

Verify them after copying:

```sh
shasum -a 256 checkpoints/moka-global-soup-exact-q50-int8-roundtrip.safetensors data/moka-current-search256-offset9550k-64.npz data/moka-global-soup-search128-offset5400k-128.npz data/moka-global-b18-onpolicy-offset4720k-64x128.npz teachers/katago-b6c96.onnx
```

## Accepted player

- Checkpoint: `moka-global-soup-exact-q50-int8-roundtrip.safetensors`
- Sequential simulations: 256
- Exploration: 1.75
- Value weight: 1.25
- FPU reduction: 0.25
- Opponent width: 4
- Root and descendant symmetry: all eight views
- Root and descendant geometric policy blend: 0.125
- Rules: positional superko, 7-point komi, area scoring
- Resignation area margin: 60 points

Two independent 128-versus-256 comparisons each gave 256 visits exactly nine more wins. Pooled across 100 games, 256 visits scored 64 wins versus 46 for 128, split 30 Black and 34 White versus 22 and 24, with zero caps.

Do not change these accepted settings while evaluating a model candidate.

## Recent rejected search experiments

- Sequential 512 visits tied 256 at 11/20 and lost one Black win.
- Exploration 2.0 led 23–19 across two screens but improved only one game in the independent audit, below its frozen gate.
- Value weight 1.0 matched native b18's top move on 77/128 roots versus 80/128 for weight 1.25 and regressed both colors.
- Post-move pondering retained 85.5% of resolved reply subtrees and added 92,416 simulations, but lost 8–11 on the paired screen. Its implementation was removed.
- Visit-margin, symmetry-spread, and learned-uncertainty adaptive compute were already rejected.
- Score blending, retained action-value priors, child-Q shrinkage, LCB root selection, dynamic cPUCT, logarithmic cPUCT, RAVE, graph search, subtree bias, handcrafted area/liberty/eye values, and terminal proof propagation were already tested and rejected.

Do not repeat these experiments without a new causal signal and a preregistered reason.

## In-progress experiment

The next experiment is frozen in `experiment-log.md` under “Preregistered 256-visit policy iteration.” Data collection is complete; training has not started.

The new corpus contains 2,205 Moka-turn positions from 64 paired-color games at opening offset 9,550,000. Targets blend 75% of Moka's accepted 256-visit distribution with 25% of its legal eight-view root policy. KataGo b6c96 supplied opponent moves and scalar values only.

Train exactly one candidate:

```sh
uv run python -m go_model.train \
  --data data/moka-current-search256-offset9550k-64.npz \
  --supplemental-data data/moka-global-soup-search128-offset5400k-128.npz \
  --initial-checkpoint checkpoints/moka-global-soup-exact-q50-int8-roundtrip.safetensors \
  --checkpoint checkpoints/moka-search256-adapter-seed530.safetensors \
  --epochs 1 \
  --batch-size 256 \
  --learning-rate 0.00001 \
  --seed 530 \
  --game-pair-size 2 \
  --policy-preservation-weight 0.25 \
  --global-residual \
  --global-residual-adapter-only \
  --int8-quantization-aware
```

Materialize only the 12 trained adapter tensors into the exact accepted base:

```sh
uv run python -m go_model.quantization \
  --checkpoint checkpoints/moka-search256-adapter-seed530.safetensors \
  --output checkpoints/moka-search256-adapter-seed530-exact.safetensors \
  --base-checkpoint checkpoints/moka-global-soup-exact-q50-int8-roundtrip.safetensors \
  --parameter-prefix residual_blocks.3.global_ \
  --parameter-prefix residual_blocks.7.global_ \
  --parameter-prefix residual_blocks.11.global_
```

Before arena play, prove all of the following:

1. New-corpus test policy loss improves by at least 0.005 and top-move agreement does not fall.
2. Prior 128-visit test loss stays within 0.001 and top agreement is unchanged or better.
3. Offset-4,720,000 policy loss and value MAE each stay within 0.001.
4. Every tensor outside the three `residual_blocks.*.global_` prefixes is byte-identical to the accepted checkpoint.

Use paired-game test buckets and the existing `go_model.train.evaluate` implementation. Write exact metrics, commands, hashes, and decisions to `experiment-log.md` before proceeding.

If every offline gate passes, run the frozen 20-game screen:

```sh
uv run python -m go_model.arena \
  --checkpoint checkpoints/moka-global-soup-exact-q50-int8-roundtrip.safetensors \
  --teacher teachers/katago-b6c96.onnx \
  --games 20 \
  --simulations 256 \
  --opening-offset 9560000 \
  --global-residual

uv run python -m go_model.arena \
  --checkpoint checkpoints/moka-search256-adapter-seed530-exact.safetensors \
  --teacher teachers/katago-b6c96.onnx \
  --games 20 \
  --simulations 256 \
  --opening-offset 9560000 \
  --global-residual
```

The candidate must gain at least two wins, lose no wins as either color, add no cap or resignation, and remain within 5% runtime. If it passes, preregister and run two untouched 40-game confirmations. If it fails, reject it without another seed, learning rate, epoch, or target blend.

## Research discipline

- Read the complete `experiment-log.md` before proposing another mechanism.
- Write hypotheses, constants, offsets, and gates before results.
- Use paired openings and report Black and White separately.
- Treat caps and resignation changes as first-class evidence.
- Never select a model from the same arena block used to tune it.
- Do not claim rank or win rate without an exact reproducible match.
- Do not deploy to `million-website` until the user explicitly asks after confirmation.
- Keep browser payload and Lighthouse work separate from Python research promotion.
