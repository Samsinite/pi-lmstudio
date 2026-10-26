import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";

import extension, { mapModels } from "../extensions/pi-lmstudio/index.ts";

function model(overrides = {}) {
  return {
    type: "llm",
    publisher: "local",
    key: "test-model",
    display_name: "Test Model",
    size_bytes: 1,
    params_string: null,
    loaded_instances: [],
    max_context_length: 262144,
    format: "gguf",
    variants: [],
    selected_variant: "",
    ...overrides,
  };
}

test("uses the theoretical context window while a model is unloaded", () => {
  const [mapped] = mapModels([model()], "lmstudio");

  assert.equal(mapped.contextWindow, 262144);
  assert.equal(mapped.maxTokens, 262144);
});

test("caps output tokens to the loaded context window", () => {
  const loadedModel = model({
    loaded_instances: [
      {
        id: "test-model",
        config: {
          context_length: 40192,
          eval_batch_size: 512,
          flash_attention: true,
          num_experts: 0,
          offload_kv_cache_to_gpu: true,
        },
      },
    ],
  });
  const [mapped] = mapModels([loadedModel], "lmstudio");

  assert.equal(mapped.contextWindow, 40192);
  assert.equal(mapped.maxTokens, 40192);
});

test("refreshes model limits for idle input before preflight", async () => {
  const originalFetch = globalThis.fetch;
  const unloadedModel = model();
  const loadedModel = model({
    loaded_instances: [
      {
        id: "test-model",
        config: {
          context_length: 40192,
          eval_batch_size: 512,
          flash_attention: true,
          num_experts: 0,
          offload_kv_cache_to_gpu: true,
        },
      },
    ],
  });
  const responses = [unloadedModel, loadedModel];
  const handlers = new Map();
  const registeredContextWindows = [];

  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [responses.shift()] }),
  });

  const pi = {
    on: (event, handler) => handlers.set(event, handler),
    registerProvider: (provider, config) => {
      registeredContextWindows.push(config.models[0].contextWindow);
    },
    unregisterProvider: () => {},
  };

  try {
    await extension(pi);
    const inputHandler = handlers.get("input");
    const input = { type: "input", text: "prompt", source: "interactive" };
    await inputHandler(input, { model: { provider: "lmstudio" }, isIdle: () => true });
    await inputHandler(input, { model: { provider: "lmstudio" }, isIdle: () => false });

    assert.deepEqual(registeredContextWindows, [262144, 40192]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("refreshes only the selected LM Studio server before an idle prompt", async () => {
  const originalFetch = globalThis.fetch;
  const originalHome = process.env.HOME;
  const home = mkdtempSync(path.join(os.tmpdir(), "pi-lmstudio-"));
  const configDirectory = path.join(home, ".pi", "agent");
  mkdirSync(configDirectory, { recursive: true });
  writeFileSync(
    path.join(configDirectory, "lmstudio.json"),
    JSON.stringify({
      urls: [
        { name: "selected", url: "http://selected.test" },
        { name: "other", url: "http://other.test" },
      ],
    }),
  );
  process.env.HOME = home;

  const requestedUrls = [];
  const handlers = new Map();
  const configuredExtension = (await import("../extensions/pi-lmstudio/index.ts?selected-server-test")).default;
  globalThis.fetch = async (url) => {
    requestedUrls.push(url);
    return {
      ok: true,
      json: async () => ({ models: [model()] }),
    };
  };

  const pi = {
    on: (event, handler) => handlers.set(event, handler),
    registerProvider: () => {},
    unregisterProvider: () => {},
  };

  try {
    await configuredExtension(pi);
    await handlers.get("input")(
      { type: "input", text: "prompt", source: "interactive" },
      { model: { provider: "lmstudio/selected" }, isIdle: () => true },
    );

    assert.deepEqual(requestedUrls.sort(), [
      "http://other.test/api/v1/models",
      "http://selected.test/api/v1/models",
      "http://selected.test/api/v1/models",
    ]);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalHome === undefined) delete process.env.HOME;
    else process.env.HOME = originalHome;
    rmSync(home, { recursive: true, force: true });
  }
});

test("refreshes model limits after the first assistant message", async () => {
  const originalFetch = globalThis.fetch;
  const unloadedModel = model();
  const loadedModel = model({
    loaded_instances: [
      {
        id: "test-model",
        config: {
          context_length: 40192,
          eval_batch_size: 512,
          flash_attention: true,
          num_experts: 0,
          offload_kv_cache_to_gpu: true,
        },
      },
    ],
  });
  const responses = [unloadedModel, loadedModel];
  const handlers = new Map();
  const registeredContextWindows = [];

  globalThis.fetch = async () => ({
    ok: true,
    json: async () => ({ models: [responses.shift()] }),
  });

  const pi = {
    on: (event, handler) => handlers.set(event, handler),
    registerProvider: (_provider, config) => {
      registeredContextWindows.push(config.models[0].contextWindow);
    },
    unregisterProvider: () => {},
  };

  try {
    await extension(pi);
    await handlers.get("agent_start")();
    await handlers.get("message_end")({ message: { role: "user" } });
    await handlers.get("message_end")({ message: { role: "assistant" } });
    await handlers.get("message_end")({ message: { role: "assistant" } });

    assert.deepEqual(registeredContextWindows, [262144, 40192]);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
