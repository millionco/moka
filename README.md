# Moka

A very cute Go-playing model.

Moka v1 is a 109,569-parameter policy and value network for 9×9 Go. Its 112 KB INT8 weights run entirely in the browser. We estimate it plays around 10 kyu.

Moka was distilled from KataGo b6c96 using teacher games and positions reached by Moka’s own rollouts. It matches KataGo’s preferred move on 46.3% of held-out positions.

## Arena

| Set       | Moka | KataGo | Black wins | White wins | Move caps |
| --------- | ---: | -----: | ---------: | ---------: | --------: |
| 100 games |   52 |     48 |         26 |         26 |         8 |

| Checkpoint         | Search     | Opponent replies | Symmetries |
| ------------------ | ---------- | ---------------: | ---------: |
| 104,129 parameters | 512 visits |                8 |          8 |

## Browser payload

| Path                | Weights | Runtime | Total load |
| ------------------- | ------: | ------: | ---------: |
| Moka v1 · INT8      |  100 KB |    8 KB |     108 KB |
| KataGo b6c96 · ONNX |  4.1 MB | 12.8 MB |    17.0 MB |

Moka’s browser path is about 157× smaller than the teacher path. The point is not to replace KataGo. It is to put a learned Go player inside an ordinary webpage.

## Browser performance

| Metric         | Latency |
| -------------- | ------: |
| Initialization | 10.0 ms |
| Mean inference |  9.6 ms |
| p50 inference  |  9.4 ms |
| p95 inference  | 10.6 ms |

These are the medians of five Chromium runs on Apple Silicon. Each run measured 100 positions after 10 warmups. Latency includes Worker messaging; inference stays off the main thread.

## Setup

```sh
uv sync
ni
```

Place KataGo b6c96 at `teachers/katago-b6c96.onnx`.

## Train

```sh
uv run go-generate --positions 50000
uv run go-train --data data/katago-distillation.npz --epochs 30
uv run go-export
```

## Build

```sh
nr build
```

## Test

```sh
nr check
nr test
```

## License

MIT
