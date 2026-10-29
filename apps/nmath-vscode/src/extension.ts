// SPDX-License-Identifier: Apache-2.0
import { existsSync } from "node:fs";
import * as vscode from "vscode";
import {
  LanguageClient,
  type LanguageClientOptions,
  type ServerOptions,
} from "vscode-languageclient/node";

import { resolveCompilerPath, startFailureMessage } from "./resolve";

let client: LanguageClient | undefined;
let output: vscode.LogOutputChannel | undefined;

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

async function start(): Promise<void> {
  if (!vscode.workspace.isTrusted) {
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
  client = new LanguageClient(
    "nmath",
    "NMath Language Server",
    serverOptions,
    clientOptions,
  );
  try {
    await client.start();
  } catch (error) {
    client = undefined;
    const cause = error instanceof Error ? error.message : String(error);
    void vscode.window.showErrorMessage(startFailureMessage(command, cause));
  }
}

async function stop(): Promise<void> {
  const current = client;
  client = undefined;
  await current?.stop();
}

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  context.subscriptions.push(
    vscode.commands.registerCommand("nmath.restartLanguageServer", async () => {
      await stop();
      await start();
    }),
    vscode.workspace.onDidGrantWorkspaceTrust(() => start()),
    vscode.workspace.onDidChangeConfiguration(async (event) => {
      if (event.affectsConfiguration("nmath.compilerPath")) {
        await stop();
        await start();
      }
    }),
  );
  await start();
}

export function deactivate(): Promise<void> {
  return stop();
}
