# Moka autoresearch

This is a Moka-specific adaptation of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx).

The upstream language-model dataset, tokenizer, training script, and `val_bpb` metric are intentionally not copied. Moka already uses MLX and has a stricter experimental protocol:

- one mutable file: `experiment.py`
- fixed whole-game train, validation, and test buckets
- fixed 100-game color-balanced development arena
- three training seeds by default
- bootstrap-gated keep/discard decisions
- exact nested-checkpoint and size validation
- an offset-1000 final arena that automation must never run

Initialize the baseline from the repository root:

```sh
uv run python autoresearch/rigor.py run "19-win distilled-GRPO baseline"
```

Then give an agent `autoresearch/program.md`. The ledger is local and ignored by Git.

This harness is designed for model-training experiments. Teacher-data generation has a separate compute budget and must preserve whole games or GRPO groups before splitting.
