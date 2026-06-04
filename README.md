# OCC-RAG WebGPU

A WebGPU chat demo for [`occ-ai/OCC-RAG-0.6B`](https://huggingface.co/occ-ai/OCC-RAG-0.6B) —
a 0.6B-parameter retrieval-augmented model (built on Qwen3) tuned for faithful,
citation-anchored question answering with calibrated abstention. Everything runs
entirely in the browser with [🤗 Transformers.js](https://huggingface.co/docs/transformers.js)
and ONNX Runtime Web, so no data is sent to a server.

The ONNX checkpoint is loaded at runtime from the Hugging Face Hub
([`mrsndmn/OCC-RAG-0.6B-ONNX`](https://huggingface.co/mrsndmn/OCC-RAG-0.6B-ONNX),
the `q4f16` variant) and cached in the browser, so the site itself is just static
files — perfect for GitHub Pages.

## Live demo

Deployed automatically to GitHub Pages: **https://mrsndmn.github.io/occ-js/**

A WebGPU-capable browser is required (recent Chrome / Edge, or Firefox Nightly).

## Develop locally

```sh
npm i
npm run dev
```

Open `http://localhost:5173`, click **Load model**, and chat. Use the
**Context sources** box to paste reference passages (separate multiple sources
with a blank line) — they are attached to your next question as numbered sources,
so the model can answer from them and cite them.

> The model weights (~534 MB) are downloaded from the Hub on first **Load model**
> and then cached by the browser.

## Deploy (GitHub Pages)

Deployment is fully automated by
[`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml): every push to
`main` builds the Vite app and publishes `dist/` to GitHub Pages.

One-time setup in the repo: **Settings → Pages → Build and deployment → Source →
GitHub Actions**.

The Vite `base` defaults to `/occ-js/` and the workflow overrides it with the
actual repo name via the `BASE_PATH` env var, so assets resolve at
`https://<user>.github.io/<repo>/`. For a custom domain or a user/organization
page, set `BASE_PATH=/`.

## How the ONNX checkpoint was built

`OCC-RAG-0.6B` ships only `safetensors` weights, so it is exported to ONNX and
quantized (the demo uses `q4f16`) before Transformers.js can load it. The
reproducible conversion lives in [`scripts/convert.sh`](./scripts/convert.sh)
(it uses [`uv`](https://github.com/astral-sh/uv) and the
[Transformers.js conversion script](https://github.com/huggingface/transformers.js)).
It downloads the model, normalizes the (transformers v5) `config.json` /
`tokenizer_config.json` for the exporter, exports with KV-cache, and inlines the
chat template into `tokenizer_config.json`. The resulting
`OCC-RAG-0.6B-ONNX` folder is what was uploaded to the Hub.

To serve the model locally instead of from the Hub, run `./scripts/convert.sh`
(it writes `public/models/OCC-RAG-0.6B-ONNX/`), then in
[`src/worker.js`](./src/worker.js) set `MODEL_ID = "OCC-RAG-0.6B-ONNX"` and
`env.allowRemoteModels = false`.
