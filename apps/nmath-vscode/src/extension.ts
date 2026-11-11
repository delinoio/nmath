// SPDX-License-Identifier: Apache-2.0
import { existsSync } from "node:fs";
import type { ChildProcess } from "node:child_process";
import * as vscode from "vscode";
import {
  LanguageClient,
  State,
  type LanguageClientOptions,
  type ServerOptions,
} from "vscode-languageclient/node";

import { resolveCompilerPath, startFailureMessage } from "./resolve";

let client: LanguageClient | undefined;
let output: vscode.LogOutputChannel | undefined;
let lifecycle: Promise<void> = Promise.resolve();
let deactivated = false;

function queue(operation: () => Promise<void>): Promise<void> {
  const next = lifecycle.then(operation);
  lifecycle = next.catch((error: unknown) => {
    const cause = error instanceof Error ? error.message : String(error);
    void vscode.window.showErrorMessage(
      `The NMath language server could not restart: ${cause}. Run 'NMath: Restart Language Server' to try again.`,
    );
  });
  return next;
}

function compilerCommand(): string {
  return resolveCompilerPath({
    setting: vscode.workspace
      .getConfiguration("nmath")
      .get<string>("compilerPath", ""),
    workspaceFolders: (vscode.workspace.workspaceFolders ?? []).map(
      (f) => f.uri.fsPath,
    ),
    platform: process.platform,
    exists: existsSync,
  });
}

function waitForServerExit(server: ChildProcess): Promise<void> {
  if (server.exitCode !== null || server.signalCode !== null) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const onExit = () => {
      clearTimeout(timeout);
      resolve();
    };
    const timeout = setTimeout(() => {
      server.removeListener("exit", onExit);
      reject(new Error("The failed language server process did not exit"));
    }, 5000);
    server.once("exit", onExit);
  });
}

async function cleanUpFailedStart(next: LanguageClient): Promise<void> {
  if (next.state !== State.StartFailed) {
    await stop();
    return;
  }

  const server = next.serverProcess;
  if (server?.pid !== undefined) {
    // vscode-languageclient 11 rejects stop() after a failed start, but its
    // Node client still schedules termination of the spawned server. Keep
    // ownership until that process exits. Remove this path when the client
    // can await shutdown directly from StartFailed.
    try {
      await next.stop();
    } catch {
      // The process exit below establishes whether cleanup succeeded.
    }
    await waitForServerExit(server);
  }
  if (client === next) {
    client = undefined;
  }
}

async function start(): Promise<void> {
  if (deactivated || !vscode.workspace.isTrusted || client) {
    return;
  }
  output ??= vscode.window.createOutputChannel("NMath Language Server", {
    log: true,
  });
  const command = compilerCommand();
  const serverOptions: ServerOptions = {
    command,
    args: ["lsp"],
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: "file", language: "nts" }],
    outputChannel: output,
  };
  const next = new LanguageClient(
    "nmath",
    "NMath Language Server",
    serverOptions,
    clientOptions,
  );
  client = next;
  try {
    await next.start();
  } catch (error) {
    const cause = error instanceof Error ? error.message : String(error);
    void vscode.window.showErrorMessage(startFailureMessage(command, cause));
    await cleanUpFailedStart(next);
  }
}

async function stop(): Promise<void> {
  const current = client;
  await current?.stop();
  if (client === current) {
    client = undefined;
  }
}

async function restart(): Promise<void> {
  if (deactivated) {
    return;
  }
  await stop();
  await start();
}

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand("nmath.restartLanguageServer", () =>
      queue(restart),
    ),
    vscode.workspace.onDidGrantWorkspaceTrust(() => queue(start)),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("nmath.compilerPath")) {
        return queue(restart);
      }
    }),
  );
  await queue(start);
}

export function deactivate(): Promise<void> {
  deactivated = true;
  return queue(stop);
}
