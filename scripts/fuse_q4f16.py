#!/usr/bin/env python
"""
Rebuild the q4f16 ONNX variant as a PRE-FUSED, full-fp16 graph.

Why this exists
---------------
The plain transformers.js export + `quantize_fp16` (keep_io_types=True) produces a
q4f16 graph with float32 I/O and RMSNorm left as primitive ops (Pow/ReduceMean/...).
When ONNX Runtime Web loads it on WebGPU, ORT inserts its own precision casts
("InsertedPrecisionFreeCast_...") and then its `SimplifiedLayerNormFusion` pass
trips over them, aborting session init with an opaque `uncaught exception: <number>`
(the demo gets stuck on "Loading model...").

onnx-community's Qwen3 ONNX models avoid this because the RMSNorm is *already*
fused into com.microsoft `(Skip)SimplifiedLayerNormalization` nodes, so ORT never
runs the buggy fusion at load time.

This script reproduces that: it runs ONNX Runtime's `qwen3` transformer optimizer
(which fuses (Skip)SimplifiedLayerNormalization), converts to full fp16
(keep_io_types=False -> fp16 KV-cache I/O), then re-quantizes the MatMuls to 4-bit
(MatMulNBits). The result loads at ORT's default ("all") optimization level.

Requires onnxruntime >= 1.22 (model_type="qwen3" in the transformers optimizer)
plus onnx_ir (MatMulNBitsQuantizer). Usage:
    python fuse_q4f16.py <fp32_model.onnx> <out_q4f16.onnx>
"""
import sys
import onnx
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

try:
    from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer
except ImportError:  # renamed in onnxruntime >= 1.23
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer as MatMul4BitsQuantizer,
    )

# Qwen3-0.6B geometry (decoupled head_dim, GQA: 16 q heads / 8 kv heads).
NUM_HEADS = 16
HIDDEN_SIZE = 1024
BLOCK_SIZE = 32  # matches quantize_config.json


def main(src: str, out: str) -> None:
    print(f">> optimize_model(model_type=qwen3) on {src}")
    m = optimizer.optimize_model(
        src,
        model_type="qwen3",
        num_heads=NUM_HEADS,
        hidden_size=HIDDEN_SIZE,
        optimization_options=FusionOptions("qwen3"),
        opt_level=0,  # skip the ORT graph optimizer (avoids CPU-only nchwc bloat)
        use_gpu=True,
    )

    print(">> convert to float16 (keep_io_types=False -> fp16 KV-cache I/O)")
    m.convert_float_to_float16(keep_io_types=False)

    fp16_tmp = out + ".fp16.tmp.onnx"
    m.save_model_to_file(fp16_tmp, use_external_data_format=False)

    print(">> quantize MatMul -> MatMulNBits (4-bit, symmetric)")
    q = MatMul4BitsQuantizer(
        model=onnx.load_model(fp16_tmp), block_size=BLOCK_SIZE, is_symmetric=True
    )
    q.process()
    onnx.save_model(q.model.model, out)

    import os

    os.remove(fp16_tmp)
    print(f">> wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
