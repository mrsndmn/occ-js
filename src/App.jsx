import { useEffect, useState, useRef } from "react";

import Chat from "./components/Chat";
import ArrowRightIcon from "./components/icons/ArrowRightIcon";
import StopIcon from "./components/icons/StopIcon";
import Progress from "./components/Progress";

const IS_WEBGPU_AVAILABLE = !!navigator.gpu;
const STICKY_SCROLL_THRESHOLD = 120;
// Each example prefills both the Context sources box and the question, so the
// model can answer (and cite) from grounded passages. The last one asks
// something the sources do not cover, to showcase calibrated abstention.
const EXAMPLES = [
  {
    question: "Who designed the Eiffel Tower, and how tall is it?",
    context:
      "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by the engineer Gustave Eiffel and built between 1887 and 1889 for the 1889 World's Fair.\n\nThe tower stands 330 metres (1,083 ft) tall and was the tallest man-made structure in the world until the Chrysler Building was completed in 1930.",
  },
  {
    question:
      "What does a plant take in, and what does it release during photosynthesis?",
    context:
      "Photosynthesis is the process by which green plants convert light energy into chemical energy. Using sunlight, they combine carbon dioxide from the air with water drawn up from the soil.\n\nThe process produces glucose, which the plant uses for energy, and releases oxygen into the atmosphere as a by-product.",
  },
  {
    question: "Who is the company's CEO?",
    context:
      "Our store accepts returns within 30 days of purchase with a valid receipt. Refunds are issued to the original payment method within 5–7 business days.\n\nItems marked as final sale cannot be returned or exchanged.",
  },
];

// Split the free-form context box into individual numbered sources.
// Sources are separated by a blank line.
function parseDocuments(context) {
  return context
    .split(/\n\s*\n/)
    .map((text) => text.trim())
    .filter((text) => text.length > 0)
    .map((text) => ({ text }));
}

function App() {
  // Create a reference to the worker object.
  const worker = useRef(null);

  const textareaRef = useRef(null);
  const chatContainerRef = useRef(null);

  // Model loading and progress
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState([]);
  const [isRunning, setIsRunning] = useState(false);

  // Inputs and outputs
  const [input, setInput] = useState("");
  const [context, setContext] = useState("");
  const [messages, setMessages] = useState([]);
  const [tps, setTps] = useState(null);
  const [numTokens, setNumTokens] = useState(null);

  // This is a single-turn demo: each context allows exactly one question.
  // Once a question has been asked, the inputs lock until the user resets.
  const hasAsked = messages.length > 0;

  function onEnter(message) {
    // Guard against asking a follow-up in the same context.
    if (hasAsked || isRunning) return;
    setMessages((prev) => [...prev, { role: "user", content: message }]);
    setTps(null);
    setIsRunning(true);
    setInput("");
  }

  // Prefill the context sources, then ask the example question. Both state
  // updates are batched, so the generate effect sees the new context.
  function onExample(example) {
    if (hasAsked || isRunning) return;
    setContext(example.context);
    onEnter(example.question);
  }

  function onInterrupt() {
    // NOTE: We do not set isRunning to false here because the worker
    // will send a 'complete' message when it is done.
    worker.current.postMessage({ type: "interrupt" });
  }

  useEffect(() => {
    resizeInput();
  }, [input]);

  function resizeInput() {
    if (!textareaRef.current) return;

    const target = textareaRef.current;
    target.style.height = "auto";
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    // Create the worker if it does not yet exist.
    if (!worker.current) {
      worker.current = new Worker(new URL("./worker.js", import.meta.url), {
        type: "module",
      });
      worker.current.postMessage({ type: "check" }); // Do a feature check
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e) => {
      switch (e.data.status) {
        case "loading":
          // Model file start load: add a new progress item to the list.
          setStatus("loading");
          setLoadingMessage(e.data.data);
          break;

        case "initiate":
          setProgressItems((prev) => [...prev, e.data]);
          break;

        case "progress":
          // Model file progress: update one of the progress items.
          setProgressItems((prev) =>
            prev.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data };
              }
              return item;
            }),
          );
          break;

        case "done":
          // Model file loaded: remove the progress item from the list.
          setProgressItems((prev) =>
            prev.filter((item) => item.file !== e.data.file),
          );
          break;

        case "ready":
          // Pipeline ready: the worker is ready to accept messages.
          setStatus("ready");
          break;

        case "start":
          {
            // Start generation
            setMessages((prev) => [
              ...prev,
              { role: "assistant", content: "" },
            ]);
          }
          break;

        case "update":
          {
            // Generation update: update the output text.
            const { output, tps, numTokens } = e.data;
            setTps(tps);
            setNumTokens(numTokens);
            setMessages((prev) => {
              const cloned = [...prev];
              const last = cloned.at(-1);
              cloned[cloned.length - 1] = {
                ...last,
                content: last.content + output,
                // OCC-RAG emits a structured answer directly (no separate
                // hidden reasoning block to collapse), so render everything.
                answerIndex: 0,
              };
              return cloned;
            });
          }
          break;

        case "complete":
          // Generation complete: re-enable the "Generate" button
          setIsRunning(false);
          break;

        case "error":
          setError(e.data.data);
          break;
      }
    };

    const onErrorReceived = (e) => {
      console.error("Worker error:", e);
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);
    worker.current.addEventListener("error", onErrorReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current.removeEventListener("message", onMessageReceived);
      worker.current.removeEventListener("error", onErrorReceived);
    };
  }, []);

  // Send the messages to the worker thread whenever the `messages` state changes.
  useEffect(() => {
    if (messages.filter((x) => x.role === "user").length === 0) {
      // No user messages yet: do nothing.
      return;
    }
    if (messages.at(-1).role === "assistant") {
      // Do not update if the last message is from the assistant
      return;
    }
    setTps(null);
    worker.current.postMessage({
      type: "generate",
      data: { messages, documents: parseDocuments(context) },
    });
  }, [messages, isRunning]);

  useEffect(() => {
    if (!chatContainerRef.current || !isRunning) return;
    const element = chatContainerRef.current;
    if (
      element.scrollHeight - element.scrollTop - element.clientHeight <
      STICKY_SCROLL_THRESHOLD
    ) {
      element.scrollTop = element.scrollHeight;
    }
  }, [messages, isRunning]);

  return IS_WEBGPU_AVAILABLE ? (
    <div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">
      {status === null && messages.length === 0 && (
        <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
          <div className="flex flex-col items-center mb-1 max-w-[400px] text-center">
            <img
              src="logo.png"
              width="80%"
              height="auto"
              className="block drop-shadow-lg bg-transparent"
            ></img>
            <h1 className="text-4xl font-bold my-1">OCC-RAG WebGPU</h1>
            <h2 className="font-semibold">
              A context-grounded, citation-anchored RAG model that runs locally
              in your browser with WebGPU acceleration.
            </h2>
          </div>

          <div className="flex flex-col items-center px-4">
            <p className="max-w-[514px] mb-4">
              <br />
              You are about to load{" "}
              <a
                href="https://huggingface.co/occ-ai/OCC-RAG-0.6B"
                target="_blank"
                rel="noreferrer"
                className="font-medium underline"
              >
                OCC-RAG-0.6B
              </a>
              , a 0.6B parameter retrieval-augmented model (built on Qwen3)
              optimized for faithful, source-grounded question answering.
              Everything runs entirely in your browser with{" "}
              <a
                href="https://huggingface.co/docs/transformers.js"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                🤗&nbsp;Transformers.js
              </a>{" "}
              and ONNX Runtime Web, meaning no data is sent to a server. Once
              loaded, it can even be used offline. The source code for the demo
              is available on{" "}
              <a
                href="https://github.com/huggingface/transformers.js-examples/tree/main/occ-rag-webgpu"
                target="_blank"
                rel="noreferrer"
                className="font-medium underline"
              >
                GitHub
              </a>
              .
            </p>

            {error && (
              <div className="text-red-500 text-center mb-2">
                <p className="mb-1">
                  Unable to load model due to the following error:
                </p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            <button
              className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 cursor-pointer disabled:cursor-not-allowed select-none"
              onClick={() => {
                worker.current.postMessage({ type: "load" });
                setStatus("loading");
              }}
              disabled={status !== null || error !== null}
            >
              Load model
            </button>
          </div>
        </div>
      )}
      {status === "loading" && (
        <>
          <div className="w-full max-w-[500px] text-left mx-auto p-4 bottom-0 mt-auto">
            <p className="text-center mb-1">{loadingMessage}</p>
            {progressItems.map(({ file, progress, total }, i) => (
              <Progress
                key={i}
                text={file}
                percentage={progress}
                total={total}
              />
            ))}
          </div>
        </>
      )}

      {status === "ready" && (
        <div
          ref={chatContainerRef}
          className="overflow-y-auto scrollbar-thin w-full flex flex-col items-center h-full"
        >
          <Chat messages={messages} />
          {messages.length === 0 && (
            <div>
              {EXAMPLES.map((example, i) => (
                <div
                  key={i}
                  className="m-1 border border-gray-300 dark:border-gray-600 rounded-md p-2 bg-gray-100 dark:bg-gray-700 cursor-pointer max-w-[500px]"
                  onClick={() => onExample(example)}
                  title={example.context}
                >
                  <div>{example.question}</div>
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Prefills {parseDocuments(example.context).length} context
                    source
                    {parseDocuments(example.context).length === 1 ? "" : "s"}
                  </div>
                </div>
              ))}
            </div>
          )}
          <p className="text-center text-sm min-h-6 text-gray-500 dark:text-gray-300">
            {tps && messages.length > 0 && (
              <>
                {!isRunning && (
                  <span>
                    Generated {numTokens} tokens in{" "}
                    {(numTokens / tps).toFixed(2)} seconds&nbsp;&#40;
                  </span>
                )}
                {
                  <>
                    <span className="font-medium text-center mr-1 text-black dark:text-white">
                      {tps.toFixed(2)}
                    </span>
                    <span className="text-gray-500 dark:text-gray-300">
                      tokens/second
                    </span>
                  </>
                }
                {!isRunning && (
                  <>
                    <span className="mr-1">&#41;.</span>
                    <span
                      className="underline cursor-pointer"
                      onClick={() => {
                        worker.current.postMessage({ type: "reset" });
                        setMessages([]);
                      }}
                    >
                      Reset
                    </span>
                  </>
                )}
              </>
            )}
          </p>
        </div>
      )}

      <div className="w-[600px] max-w-[80%] mx-auto mt-2 mb-3">
        <details className="mb-2 text-sm">
          <summary className="cursor-pointer text-gray-500 dark:text-gray-300 select-none">
            Context sources {context.trim() ? "(in use)" : "(optional)"}
          </summary>
          <textarea
            className="scrollbar-thin w-full mt-1 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-500 dark:bg-gray-700 bg-transparent outline-hidden text-gray-800 dark:text-gray-200 placeholder-gray-400 resize-none"
            placeholder={
              "Paste reference passages here. Separate multiple sources with a blank line.\n\nThey are attached to your next question as numbered sources for grounded answering."
            }
            rows={4}
            value={context}
            disabled={status !== "ready" || hasAsked}
            onChange={(e) => setContext(e.target.value)}
          />
        </details>
        <div className="border border-gray-300 dark:border-gray-500 dark:bg-gray-700 rounded-lg max-h-[200px] relative flex">
          <textarea
            ref={textareaRef}
            className="scrollbar-thin w-[550px] px-3 py-4 rounded-lg bg-transparent border-none outline-hidden text-gray-800 disabled:text-gray-400 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-300 disabled:placeholder-gray-200 dark:disabled:placeholder-gray-500 resize-none disabled:cursor-not-allowed"
            placeholder={
              hasAsked
                ? "One question per context — press Reset to ask another."
                : "Type your message..."
            }
            type="text"
            rows={1}
            value={input}
            disabled={status !== "ready" || hasAsked}
            title={
              status !== "ready"
                ? "Model not loaded yet"
                : hasAsked
                  ? "One question per context — press Reset to ask another"
                  : "Model is ready"
            }
            onKeyDown={(e) => {
              if (
                input.length > 0 &&
                !isRunning &&
                !hasAsked &&
                e.key === "Enter" &&
                !e.shiftKey
              ) {
                e.preventDefault(); // Prevent default behavior of Enter key
                onEnter(input);
              }
            }}
            onInput={(e) => setInput(e.target.value)}
          />
          {isRunning ? (
            <div className="cursor-pointer" onClick={onInterrupt}>
              <StopIcon className="h-8 w-8 p-1 rounded-md text-gray-800 dark:text-gray-100 absolute right-3 bottom-3" />
            </div>
          ) : input.length > 0 && !hasAsked ? (
            <div className="cursor-pointer" onClick={() => onEnter(input)}>
              <ArrowRightIcon
                className={`h-8 w-8 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-3 bottom-3`}
              />
            </div>
          ) : (
            <div>
              <ArrowRightIcon
                className={`h-8 w-8 p-1 bg-gray-200 dark:bg-gray-600 text-gray-50 dark:text-gray-800 rounded-md absolute right-3 bottom-3`}
              />
            </div>
          )}
        </div>
      </div>
      <p className="text-xs text-gray-400 text-center mb-3">
        Disclaimer: Generated content may be inaccurate or false.
      </p>
    </div>
  ) : (
    <div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">
      WebGPU is not supported
      <br />
      by this browser :&#40;
    </div>
  );
}

export default App;
