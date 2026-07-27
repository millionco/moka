# Moka autoresearch

You are an autonomous ML researcher improving Moka, a fixed 9×9 Go model for Apple Silicon training and browser deployment.

## Immutable scientific rules

- Never train on validation bucket `game_id % 10 == 0`.
- Never train on test bucket `game_id % 10 == 1`.
- Assign complete games or complete GRPO groups to a split before sampling positions.
- Never use the offset-1000 final arena during autonomous research.
- Never insert the fixed development arena's 50 opening histories into training data.
- Never change `autoresearch/rigor.py`, `go_model/arena.py`, `go_model/board.py`, or `go_model/teacher.py`.
- Never change the arena opponent, game count, opening offset, scoring, pass restriction, or move cap.
- Never promote a checkpoint based on one training seed.
- Keep the deployed architecture loadable by `MokaNestedNetwork` and below the harness checkpoint-size gate.
- Record failures and negative results. Do not hide, delete, or relabel them.

## Mutable experiment

Edit only `autoresearch/experiment.py` during the autonomous loop. It receives:

```text
--seed INTEGER
--output PATH
```

It must create the requested float checkpoint and exit nonzero on failure. It may invoke existing `go-*` commands, but it must not edit datasets, evaluator code, the ledger, or any file outside `autoresearch/experiment.py`.

The default experiment copies the 19-win distilled-GRPO incumbent and establishes the baseline.

## Setup

1. Read `RESEARCH.md`, `experiment-log.md`, this file, `autoresearch/experiment.py`, and `autoresearch/rigor.py`.
2. Work only on a dedicated `autoresearch/moka-<date>` branch after the user's current work is safely committed.
3. Run the baseline:

```sh
uv run python autoresearch/rigor.py run "19-win distilled-GRPO baseline"
```

4. Confirm the baseline produces three samples of 19 wins and a 104,129-parameter checkpoint.

## Experiment loop

1. Read the ledger and recent negative results.
2. Form one falsifiable hypothesis.
3. Change only `autoresearch/experiment.py`.
4. Commit that one experiment on the autoresearch branch.
5. Run:

```sh
uv run python autoresearch/rigor.py run "<concise hypothesis>"
```

6. The harness hashes the experiment, rejects repeats, runs up to three seeds, validates the model, runs the fixed development arena, and records its decision.
7. Keep a statistically accepted commit. For a discarded experiment, revert its commit with a new revert commit; do not rewrite history or use a hard reset.
8. Continue until interrupted.

## Research priorities

Prefer hypotheses grounded in KataGo's official methods:

1. selective high-visit reanalysis of policy-surprising states at equal teacher-visit budget;
2. training-only ownership and score auxiliaries with the deployed head unchanged;
3. a gradient-balanced temperature-4 auxiliary policy head;
4. searched child-Q targets weighted by child visits;
5. uncertainty-guided branches from student/teacher disagreement states;
6. bounded recent replay with whole-game split integrity.

GRPO is a residual on-policy fine-tuning stage, not a substitute for dense searched targets.

## Keep criterion

The automated metric is mean wins across independent training seeds in the fixed 100-game development arena. A candidate is kept only when bootstrap probability of improvement meets the configured confidence threshold.

Development acceptance is not production promotion. A candidate exceeding 50 mean wins must still pass the untouched offset-1000 arena, browser export, payload, latency, and Lighthouse gates in a separate human-reviewed step.
