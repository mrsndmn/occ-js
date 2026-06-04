#!/usr/bin/env bash
#
# Convert occ-ai/OCC-RAG-0.6B (safetensors) to the ONNX layout that
# Transformers.js loads, and place it under public/models/ for the demo.
#
# Requires: uv (https://github.com/astral-sh/uv) and git.
#
# Usage:
#   ./scripts/convert.sh
#
set -euo pipefail

MODEL_REPO="occ-ai/OCC-RAG-0.6B"
TJS_REF="3.7.1"  # tag of huggingface/transformers.js that hosts the conversion script
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # occ-rag-webgpu/
WORK="${HERE}/.convert"
SRC="${WORK}/src"
OUT_NAME="OCC-RAG-0.6B-ONNX"
DEST="${HERE}/public/models/${OUT_NAME}"

mkdir -p "${WORK}/scripts" "${SRC}"

echo ">> [1/6] Creating Python env with uv"
uv venv "${WORK}/.venv" --python 3.12
PY="${WORK}/.venv/bin/python"
uv pip install --python "${PY}" \
  "transformers==4.51.3" \
  "optimum>=2.1.0" "optimum-onnx[onnxruntime]>=0.1.0" \
  "onnx==1.17.0" "onnxruntime>=1.22" "onnx_ir" "onnxslim>=0.1.48" \
  "numpy<2.3" tqdm sentencepiece protobuf "huggingface_hub[cli]"
# NOTE: onnxruntime>=1.22 is required for the q4f16 rebuild in step 5b
# (model_type="qwen3" in the transformers optimizer; onnx_ir for MatMulNBits).

echo ">> [2/6] Fetching the Transformers.js conversion script (@ ${TJS_REF})"
BASE="https://raw.githubusercontent.com/huggingface/transformers.js/${TJS_REF}/scripts"
for f in convert.py quantize.py float16.py utils.py; do
  curl -fsSL -o "${WORK}/scripts/${f}" "${BASE}/${f}"
done
touch "${WORK}/scripts/__init__.py"

echo ">> [3/6] Downloading ${MODEL_REPO}"
"${PY}" - "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    "occ-ai/OCC-RAG-0.6B",
    local_dir=sys.argv[1],
    allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"],
)
PY

echo ">> [4/6] Normalizing config & tokenizer for the (older) exporter"
"${PY}" - "$SRC" <<'PY'
import json, os, sys
src = sys.argv[1]

# config.json: surface v5-style fields the older exporter doesn't read
cp = os.path.join(src, "config.json")
c = json.load(open(cp))
rp = c.pop("rope_parameters", None)
if rp:
    c["rope_theta"] = rp.get("rope_theta", 1000000)
    if rp.get("rope_type", "default") not in (None, "default"):
        c["rope_scaling"] = {k: v for k, v in rp.items() if k != "rope_theta"}
c.setdefault("rope_theta", 1000000)
if "dtype" in c and "torch_dtype" not in c:
    c["torch_dtype"] = c.pop("dtype")
c.pop("layer_types", None)
c["transformers_version"] = "4.51.3"
# The rebuilt q4f16/fp16 graphs (step 5b) have float16 KV-cache I/O, so
# Transformers.js must feed a float16 past_key_values. Without this it defaults
# to float32 and inference fails on a dtype mismatch.
c["transformers.js_config"] = {
    "kv_cache_dtype": {"q4f16": "float16", "fp16": "float16"}
}
json.dump(c, open(cp, "w"), indent=2)

# tokenizer_config.json: drop the v5 list-form `extra_special_tokens`
# (these tokens are already special-marked added tokens in tokenizer.json)
tp = os.path.join(src, "tokenizer_config.json")
t = json.load(open(tp))
if isinstance(t.get("extra_special_tokens"), list):
    t.pop("extra_special_tokens")
json.dump(t, open(tp, "w"), indent=2)
print("normalized config.json (rope_theta=%s) and tokenizer_config.json" % c["rope_theta"])
PY

echo ">> [5/6] Exporting to ONNX (q4f16, fp16, q4, q8)"
( cd "${WORK}" && HF_HUB_OFFLINE=1 "${PY}" -m scripts.convert \
    --model_id src \
    --quantize --modes q4f16 fp16 q4 q8 \
    --task text-generation-with-past \
    --device cpu --skip_validation \
    --output_parent_dir "${WORK}/out" )

CONV="${WORK}/out/src"

echo ">> [5b] Rebuilding q4f16 as a pre-fused fp16 graph (WebGPU-loadable)"
# The stock q4f16 export aborts ORT Web's session init (SimplifiedLayerNormFusion
# vs ORT's inserted precision casts). Rebuild it from the fp32 model with the
# qwen3 transformer optimizer so RMSNorm is pre-fused. See fuse_q4f16.py.
HERE_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${PY}" "${HERE_SCRIPTS}/fuse_q4f16.py" "${CONV}/onnx/model.onnx" "${CONV}/onnx/model_q4f16.onnx"

echo ">> [6/6] Inlining chat template, writing generation_config, and installing to public/models"
"${PY}" - "$SRC" "$CONV" <<'PY'
import json, os, sys
src, conv = sys.argv[1], sys.argv[2]

# Transformers.js reads `chat_template` from tokenizer_config.json (not the .jinja file)
jinja = os.path.join(src, "chat_template.jinja")
tcp = os.path.join(conv, "tokenizer_config.json")
tc = json.load(open(tcp))
if os.path.exists(jinja):
    template = open(jinja, encoding="utf-8").read()
    # The @huggingface/jinja runtime bundled with Transformers.js does not
    # implement the `| string` filter. Use the `~` concat operator (which
    # auto-stringifies) so source ids render in the browser.
    template = template.replace(
        "'<|source_start|><|source_id|>' + (loop.index | string) + ' ' + doc['text'] + '<|source_end|>\\n'",
        "'<|source_start|><|source_id|>' ~ loop.index ~ ' ' ~ doc['text'] ~ '<|source_end|>\\n'",
    )
    tc["chat_template"] = template
    json.dump(tc, open(tcp, "w"), ensure_ascii=False, indent=2)
    print("inlined chat_template into tokenizer_config.json")

# Write a valid generation_config.json (upstream one is rejected by transformers
# 4.51.3 because do_sample=False with temperature=0.0)
gc = {
    "do_sample": False,
    "eos_token_id": [151643, 151645, 151683],
    "pad_token_id": 151643,
    "max_new_tokens": 2048,
    "transformers_version": "4.51.3",
}
json.dump(gc, open(os.path.join(conv, "generation_config.json"), "w"), indent=2)
print("wrote generation_config.json")
PY

rm -rf "${DEST}"
mkdir -p "$(dirname "${DEST}")"
cp -r "${CONV}" "${DEST}"

echo ""
echo "Done. Converted model installed at:"
echo "  ${DEST}"
echo "Files:"
( cd "${DEST}" && find . -type f | sort )
