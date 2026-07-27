# Moka

A very cute Go-playing model.

Moka v1 is a 104,129-parameter policy and value network for 9×9 Go. Its 112 KB INT8 weights run entirely in the browser. With 512-visit search, it wins 52 of 100 games against KataGo b6c96.

Moka was distilled from KataGo b6c96 using teacher games and positions reached by Moka’s own rollouts. It matches KataGo’s preferred move on 46.3% of held-out positions.

## Browser payload

| Path                | Weights | Runtime | Total load |
| ------------------- | ------: | ------: | ---------: |
| Moka v1 · INT8      |  100 KB |    4 KB |     103 KB |
| KataGo b6c96 · ONNX |  4.1 MB | 12.8 MB |    17.0 MB |

Moka’s browser path is about 165× smaller than the teacher path. The point is not to replace KataGo. It is to put a learned Go player inside an ordinary webpage.

## Browser performance

| Metric         | Latency |
| -------------- | ------: |
| Initialization | 10.1 ms |
| Mean inference |  9.0 ms |
| p50 inference  |  8.8 ms |
| p95 inference  |  9.6 ms |

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
