// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import Module from "node:module";
import test from "node:test";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const extensionPath = require.resolve("../dist/extension.js");

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((ok, fail) => {
    resolve = ok;
    reject = fail;
  });
  return { promise, resolve, reject };
}

function loadExtension({ trusted = true, startHook, stopHook } = {}) {
  const callbacks = {};
  const clients = [];
  const errors = [];
  let compilerPath = "/first/nmath-compiler";
  const output = { error() {} };
  const vscode = {
    commands: {
      registerCommand(_name, callback) {
        callbacks.restart = callback;
        return { dispose() {} };
      },
    },
    workspace: {
      isTrusted: trusted,
      workspaceFolders: [],
      getConfiguration() {
        return { get: () => compilerPath };
      },
      onDidGrantWorkspaceTrust(callback) {
        callbacks.trust = callback;
        return { dispose() {} };
      },
      onDidChangeConfiguration(callback) {
        callbacks.configuration = callback;
        return { dispose() {} };
      },
    },
    window: {
      createOutputChannel: () => output,
      showErrorMessage: (message) => {
        errors.push(message);
        return Promise.resolve();
      },
    },
  };
  class FakeClient {
    constructor(_id, _name, serverOptions, clientOptions) {
      this.serverOptions = serverOptions;
      this.clientOptions = clientOptions;
      this.active = false;
      this.state = 1;
      this.stopCalls = 0;
      clients.push(this);
    }

    async start() {
      try {
        await startHook?.(this);
      } catch (error) {
        this.state = 4;
        throw error;
      }
      this.state = 2;
      this.active = true;
    }

    async stop() {
      this.stopCalls += 1;
      await stopHook?.(this);
      if (this.state === 4) {
        throw new Error("Client is not running and can't be stopped");
      }
      this.active = false;
      this.state = 1;
    }
  }
  const originalLoad = Module._load;
  Module._load = function (request, parent, isMain) {
    if (request === "vscode") return vscode;
    if (request === "vscode-languageclient/node") {
      return { LanguageClient: FakeClient, State: { StartFailed: 4 } };
    }
    return originalLoad.call(this, request, parent, isMain);
  };
  delete require.cache[extensionPath];
  let extension;
  try {
    extension = require(extensionPath);
  } finally {
    Module._load = originalLoad;
  }
  return {
    ...extension,
    callbacks,
    clients,
    errors,
    output,
    vscode,
    setCompilerPath(value) {
      compilerPath = value;
    },
    configurationChanged() {
      return callbacks.configuration({
        affectsConfiguration: (name) => name === "nmath.compilerPath",
      });
    },
  };
}

async function flush() {
  await new Promise((resolve) => setImmediate(resolve));
}

test("overlapping restarts keep one client owned until deactivation", async () => {
  const firstStop = deferred();
  const host = loadExtension({
    stopHook: (client) =>
      client === host.clients[0] ? firstStop.promise : undefined,
  });
  await host.activate({ subscriptions: [] });
  const first = host.callbacks.restart();
  await flush();
  const second = host.callbacks.restart();
  await flush();
  assert.equal(host.clients.length, 1);
  assert.equal(host.clients[0].stopCalls, 1);

  firstStop.resolve();
  await Promise.all([first, second]);
  assert.equal(host.clients.length, 3);
  assert.deepEqual(
    host.clients.map((client) => client.active),
    [false, false, true],
  );
  assert.deepEqual(host.clients[2].serverOptions, {
    command: "/first/nmath-compiler",
    args: ["lsp"],
  });
  assert.equal(host.clients[2].clientOptions.outputChannel, host.output);

  await host.deactivate();
  assert.ok(host.clients.every((client) => !client.active));
});

test("restart and configuration change use the latest compiler path", async () => {
  const firstStop = deferred();
  const host = loadExtension({
    stopHook: (client) =>
      client === host.clients[0] ? firstStop.promise : undefined,
  });
  await host.activate({ subscriptions: [] });
  const restart = host.callbacks.restart();
  await flush();
  host.setCompilerPath("/second/nmath-compiler");
  const configuration = host.configurationChanged();
  firstStop.resolve();
  await Promise.all([restart, configuration]);
  assert.equal(
    host.clients.at(-1).serverOptions.command,
    "/second/nmath-compiler",
  );
  assert.equal(host.clients.filter((client) => client.active).length, 1);
  await host.deactivate();
  assert.ok(host.clients.every((client) => !client.active));
});

test("deactivation drains pending startup and prevents later launches", async () => {
  const startup = deferred();
  const host = loadExtension({ startHook: () => startup.promise });
  const activation = host.activate({ subscriptions: [] });
  await flush();
  const deactivation = host.deactivate();
  const restart = host.callbacks.restart();
  assert.equal(host.clients.length, 1);
  startup.resolve();
  await Promise.all([activation, deactivation, restart]);
  assert.ok(host.clients.every((client) => !client.active));
  assert.equal(host.clients.length, 1);
  await host.callbacks.restart();
  assert.equal(host.clients.length, 1);
});

test("a failed stop retains ownership and a later restart can retry", async () => {
  const firstStop = deferred();
  const host = loadExtension({
    stopHook: (client) => {
      if (client === host.clients[0] && client.stopCalls === 1) {
        return firstStop.promise;
      }
    },
  });
  await host.activate({ subscriptions: [] });
  const failedRestart = host.callbacks.restart();
  await flush();
  const queuedRestart = host.callbacks.restart();
  firstStop.reject(new Error("stop failed"));
  await assert.rejects(failedRestart, /stop failed/);
  await queuedRestart;
  assert.match(host.errors[0], /stop failed/);
  assert.deepEqual(
    host.clients.map((client) => client.active),
    [false, true],
  );
  await host.deactivate();
});

test("a pending failed startup is cleaned up before a queued restart", async () => {
  const firstStart = deferred();
  const host = loadExtension({
    startHook: (client) => {
      if (client === host.clients[0]) {
        return firstStart.promise;
      }
    },
  });
  const activation = host.activate({ subscriptions: [] });
  await flush();
  const restart = host.callbacks.restart();
  firstStart.reject(new Error("start failed"));
  await Promise.all([activation, restart]);
  assert.equal(host.clients[0].stopCalls, 0);
  assert.match(host.errors[0], /start failed/);
  assert.match(host.errors[0], /nmath\.compilerPath/);
  assert.deepEqual(
    host.clients.map((client) => client.active),
    [false, true],
  );
  await host.deactivate();
});

test("a failed startup waits for its spawned server to exit", async () => {
  const firstStart = deferred();
  const server = new EventEmitter();
  server.pid = 42;
  server.exitCode = null;
  server.signalCode = null;
  const host = loadExtension({
    startHook: (client) => {
      if (client === host.clients[0]) {
        client.serverProcess = server;
        client.active = true;
        return firstStart.promise;
      }
    },
  });
  const activation = host.activate({ subscriptions: [] });
  await flush();
  const restart = host.callbacks.restart();
  firstStart.reject(new Error("initialization failed"));
  await flush();
  assert.equal(host.clients.length, 1);
  assert.equal(host.clients[0].stopCalls, 1);
  server.exitCode = 0;
  host.clients[0].active = false;
  server.emit("exit", 0, null);
  await Promise.all([activation, restart]);
  assert.deepEqual(
    host.clients.map((client) => client.active),
    [false, true],
  );
  await host.deactivate();
});

test("untrusted workspaces stay idle until trust is granted", async () => {
  const host = loadExtension({ trusted: false });
  await host.activate({ subscriptions: [] });
  await host.callbacks.restart();
  await host.configurationChanged();
  assert.equal(host.clients.length, 0);
  host.vscode.workspace.isTrusted = true;
  await host.callbacks.trust();
  assert.equal(host.clients.length, 1);
  await host.callbacks.restart();
  assert.deepEqual(
    host.clients.map((client) => client.active),
    [false, true],
  );
  await host.deactivate();
});
