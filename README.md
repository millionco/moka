# Moka

A very cute Go-playing model.

Moka v1 is a 109,569-parameter policy and value network for 9×9 Go. Its 112 KB INT8 weights run entirely in the browser. We estimate it plays around 10 kyu.

Moka was distilled from KataGo b6c96 using teacher games and positions reached by Moka’s own rollouts. It matches KataGo’s preferred move on 46.3% of held-out positions.

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
