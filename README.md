# Moka

A cute Go-playing model.

Moka v2 is a 244,529-parameter policy and value network for 9×9 Go. Its 260 KB INT8 weights run entirely in the browser. With 64-visit full-symmetry search, Moka v2 won 159 of 200 fresh games against KataGo b6c96. Moka v1 won 98 of the same 200 games. In 100 direct games, Moka v2 won 85 against Moka v1.

The KataGo b6c96 opponent plays the top move of its raw policy. It does not search.

Moka v2 was distilled from the native 9×9 KataGo b18 network. The training data has two parts: b18 self-play games, and games of Moka against KataGo b6c96. KataGo b18 labelled each position in both parts.

## Browser payload

| Path                | Weights | Runtime | Total load |
| ------------------- | ------: | ------: | ---------: |
| Moka v2 · INT8      |  235 KB |    4 KB |     239 KB |
| Moka v1 · INT8      |  100 KB |    4 KB |     103 KB |
| KataGo b6c96 · ONNX |  4.1 MB | 12.8 MB |    17.0 MB |

Weights sizes are gzipped. The Moka v2 browser path is about 70× smaller than the teacher path. The point is not to replace KataGo. It is to put a learned Go player inside an ordinary webpage.

## Browser performance

Moka runs inference in a Web Worker, which keeps model execution off the main thread.

| Metric         | Latency |
| -------------- | ------: |
| Initialization | 10.1 ms |
| Mean inference |  9.0 ms |
| p50 inference  |  8.8 ms |
| p95 inference  |  9.6 ms |

These are the Moka v1 medians of five Chromium runs on Apple Silicon. Each run measured 100 positions after 10 warmups. Latency includes Worker messaging; inference stays off the main thread.

Moka v2 needs more compute for each inference. In Node.js on an Apple M5 Max, the same runtime took 10.8 ms for Moka v2 and 4.1 ms for Moka v1 (mean of 100 positions, median of five runs). Expect browser latency for Moka v2 to be approximately 2.6× the values in the table.

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
  getAutomaticallyDeadMoves,
  getAreaScore,
  playMove,
  removeDeadStones,
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

### Adjudicate the final score

Two passes stop play, but dead stones must be removed before area scoring. Moka combines conservative pass-alive analysis with a capture aftermath to identify them automatically:

```js
const deadMoves = getAutomaticallyDeadMoves(gameState);
const adjudicatedState = removeDeadStones(gameState, deadMoves);
const blackLead = getAreaScore(adjudicatedState);
```

Pass-alive groups and unresolved seki remain on the board. `removeDeadStones` returns a copy and leaves the played position unchanged. Positive scores lead for Black; negative scores lead for White, including 7 points of komi.

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

Moka v2 needs KataGo with a Metal or CUDA backend and the native 9×9 b18 network at `teachers/kata9x9-native/kata9x9-b18c384nbt-20231025.bin.gz`.

1. Generate b18 self-play positions:

   ```sh
   uv run python -m go_model.selfplay_generate --games 3000 --visits 64
   ```

2. Generate on-policy positions. Moka plays KataGo b6c96, and b18 labels each position:

   ```sh
   uv run python -m go_model.onpolicy_generate \
     --moka-checkpoints checkpoints/moka-global-soup-exact-q50-int8-roundtrip.safetensors \
     --games 3000 --visits 32
   ```

3. Train the network:

   ```sh
   uv run python -m go_model.scaled_train \
     --data "data/selfplay/*.npz" "data/onpolicy/*.npz" \
     --checkpoint checkpoints/moka-v2.safetensors --epochs 10
   ```

4. Round the weights to INT8 and export the browser weights:

   ```sh
   uv run python -m go_model.quantization \
     --checkpoint checkpoints/moka-v2.safetensors \
     --output checkpoints/moka-v2-int8.safetensors
   uv run go-export --checkpoint checkpoints/moka-v2-int8.safetensors
   ```

To get more on-policy data, do step 2 again with the new checkpoint added to `--moka-checkpoints`. Moka v2 used two rounds.

Compare two checkpoints in direct games:

```sh
uv run python -m go_model.headtohead --first NEW.safetensors --second OLD.safetensors \
  --games 100 --simulations 64 --opening-offset 20100000
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
