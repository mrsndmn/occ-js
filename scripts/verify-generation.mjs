// Node smoke-test for the converted OCC-RAG-0.6B ONNX model.
// Exercises the same Transformers.js path the browser worker uses
// (tokenizer + custom chat template + autoregressive generation), but on the
// CPU backend so it can run headless (the browser demo uses q4f16 on WebGPU).
//
//   node scripts/verify-generation.mjs
//
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  env,
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
} from "@huggingface/transformers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Load the converted model from the local public/models folder.
env.allowRemoteModels = false;
env.localModelPath = path.join(__dirname, "..", "public", "models");

const MODEL_ID = "OCC-RAG-0.6B-ONNX";
// CPU-friendly variant for headless verification (browser uses "q4f16" on WebGPU).
const DTYPE = process.env.DTYPE || "q8";

console.log(`Loading ${MODEL_ID} (dtype=${DTYPE}) from ${env.localModelPath}`);

const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
  dtype: DTYPE,
  device: "cpu",
});
console.log("✓ tokenizer and model loaded");

// A retrieval-augmented prompt: a question grounded in two numbered sources.
const messages = [
  {
    role: "user",
    content: "Which planet is the largest, and what is it mostly made of?",
  },
];
const documents = [
  { text: "Jupiter is the largest planet in the Solar System." },
  { text: "Jupiter is a gas giant composed mainly of hydrogen and helium." },
];

const inputs = tokenizer.apply_chat_template(messages, {
  add_generation_prompt: true,
  return_dict: true,
  documents,
});
console.log(`✓ chat template applied (prompt tokens: ${inputs.input_ids.dims.at(-1)})`);

let generated = "";
const streamer = new TextStreamer(tokenizer, {
  skip_prompt: true,
  skip_special_tokens: false,
  callback_function: (t) => {
    generated += t;
  },
});

const t0 = performance.now();
const { sequences } = await model.generate({
  ...inputs,
  do_sample: false,
  max_new_tokens: 96,
  streamer,
  return_dict_in_generate: true,
});
const dt = (performance.now() - t0) / 1000;

const promptLen = inputs.input_ids.dims.at(-1);
const totalLen = sequences.dims.at(-1);
const newTokens = totalLen - promptLen;

const fullDecoded = tokenizer.batch_decode(sequences, {
  skip_special_tokens: false,
})[0];

console.log("\n----- generated text -----");
console.log(generated.trim());
console.log("--------------------------\n");

console.log(`new tokens : ${newTokens}`);
console.log(`time       : ${dt.toFixed(2)}s  (${(newTokens / dt).toFixed(1)} tok/s)`);

// Basic correctness assertions.
const problems = [];
if (newTokens < 1) problems.push("no tokens were generated");
if (generated.trim().length === 0) problems.push("decoded output is empty");
// The model should mention the grounded answer from the supplied sources.
if (!/jupiter/i.test(fullDecoded)) {
  problems.push("output does not reference the answer ('Jupiter') from the sources");
}

if (problems.length) {
  console.error("\n✗ VERIFICATION FAILED:");
  for (const p of problems) console.error("  - " + p);
  process.exit(1);
}

console.log("\n✓ VERIFICATION PASSED: model loads and generates coherent, grounded tokens.");
