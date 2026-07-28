# Moka

A cute Go-playing model.

Moka v1 is a 104,129-parameter policy and value network for 9×9 Go. Its 112 KB INT8 weights run entirely in the browser. With 64-visit search, it won 91 of 200 fresh games against KataGo b6c96.

Moka was distilled from KataGo b6c96 using teacher games and positions reached by Moka’s own rollouts.

## Browser payload

| Path                | Weights | Runtime | Total load |
| ------------------- | ------: | ------: | ---------: |
| Moka v1 · INT8      |  100 KB |    4 KB |     103 KB |
| KataGo b6c96 · ONNX |  4.1 MB | 12.8 MB |    17.0 MB |

Moka’s browser path is about 165× smaller than the teacher path. The point is not to replace KataGo. It is to put a learned Go player inside an ordinary webpage.

## Browser performance

Moka runs inference in a Web Worker, which keeps model execution off the main thread.

| Metric         | Latency |
| -------------- | ------: |
| Initialization | 10.1 ms |
| Mean inference |  9.0 ms |
| p50 inference  |  8.8 ms |
| p95 inference  |  9.6 ms |

These are the medians of five Chromium runs on Apple Silicon. Each run measured 100 positions after 10 warmups. Latency includes Worker messaging; inference stays off the main thread.

## Embed Moka on a site

Embed Moka by serving its browser client, worker, manifest, and weights with your site. The example below runs without a framework or bundler.

### Build and copy the browser assets

Build Moka from this repository, then copy its four runtime files into your site’s public directory.

```sh
ni
nr build
mkdir -p ../your_site/public/moka
cp dist/index.js ../your_site/public/moka/index.js
cp dist/worker.js ../your_site/public/moka/worker.js
cp model/go-model.json ../your_site/public/moka/go-model.json
cp model/go-model.bin ../your_site/public/moka/go-model.bin
```

The deployed site should serve these files from `/moka`. Use different URLs in the initialization options if you choose another directory.

### Initialize Moka in the browser

Run this code from a JavaScript module, such as an `app.js` file loaded with `type="module"`.

```js
import {
  GoModelWorkerClient,
  createGameState,
  encodeStudentFeatures,
  playMove,
  selectHighestLegalMove,
} from "/moka/index.js";

const mokaWorker = new Worker("/moka/worker.js", {
  type: "module",
});
const moka = new GoModelWorkerClient(mokaWorker);

await moka.initialize({
  manifestUrl: "/moka/go-model.json",
  weightsUrl: "/moka/go-model.bin",
});

let gameState = createGameState();
```

`initialize` downloads the manifest and weights, verifies the model digest, and loads the model inside the worker.

### Ask Moka to play a move

Encode the current game state, run inference, and select the highest-rated legal move.

```js
const inference = await moka.infer(encodeStudentFeatures(gameState));
const mokaMove = selectHighestLegalMove(gameState, inference.policyLogits);
const nextGameState = playMove(gameState, mokaMove);

if (nextGameState) {
  gameState = nextGameState;
}

const currentPlayerWinProbability = (inference.value + 1) / 2;
```

The `value` result ranges from `-1` to `1` for the player who was next to move. The final line converts it to a probability from `0` to `1`.

Map board intersections to move indexes with `row * 9 + column`. Indexes `0` through `80` represent board points in row-major order, starting at the top-left. Index `81` represents a pass. `playMove` returns `null` when a move is illegal.

Call `moka.dispose()` when your page or component no longer needs the model. This terminates the worker and rejects pending requests.

### Configure asset security

Serve the assets over HTTP or HTTPS because browsers do not start module workers from `file://` pages. A Content Security Policy (CSP) must allow the worker URL through `worker-src`. Cross-origin asset URLs also need Cross-Origin Resource Sharing (CORS) headers.

## Set up the training environment

Install the JavaScript and Python dependencies before training or building Moka.

```sh
uv sync
ni
```

Place KataGo b6c96 at `teachers/katago-b6c96.onnx`.

## Train Moka

Generate positions, train the network, and export its browser weights.

```sh
uv run go-generate --positions 50000
uv run go-train --data data/katago-distillation.npz --epochs 30
uv run go-export
```

## Build the browser package

Build the browser client and worker into `dist`.

```sh
nr build
```

## Run checks and tests

Run the TypeScript checks and Python test suite before publishing changes.

```sh
nr check
nr test
```

## License

MIT
