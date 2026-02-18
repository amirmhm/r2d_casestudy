#!/usr/bin/env node

/**
 * Baseline Sanity Check
 *
 * Verifies that the unpatched mini-game builds and the baseline
 * API is functional. This MUST pass before any patches are applied.
 *
 * Usage:
 *   node eval/checks/baseline_sanity.mjs [--workdir ./path]
 */

import { execSync } from "node:child_process";
import { strict as assert } from "node:assert";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { existsSync } from "node:fs";

// Parse args
const args = process.argv.slice(2);
let workDir = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--workdir" && args[i + 1]) {
    workDir = args[i + 1];
    i++;
  }
}

const miniGameDir = workDir ? resolve(workDir) : resolve("ggf-mini-game");

console.log("=== Baseline Sanity Check ===\n");

// 1. Check directory exists
console.log("1. Checking directory...");
assert.ok(existsSync(miniGameDir), `Directory not found: ${miniGameDir}`);
assert.ok(existsSync(join(miniGameDir, "package.json")), "package.json not found");
assert.ok(existsSync(join(miniGameDir, "tsconfig.json")), "tsconfig.json not found");
console.log("   OK: Directory structure valid");

// 2. Build
console.log("2. Building project...");
try {
  execSync("npm run build", { cwd: miniGameDir, stdio: "pipe", shell: true });
  console.log("   OK: Build succeeded");
} catch (e) {
  console.error("   FAIL: Build failed");
  console.error(e.stderr?.toString());
  process.exit(1);
}

// 3. Import and check exports
console.log("3. Checking exports...");
const distDir = join(miniGameDir, "dist");
const indexUrl = pathToFileURL(join(distDir, "index.js")).href;

let mod;
try {
  mod = await import(indexUrl);
} catch (e) {
  console.error(`   FAIL: Cannot import dist/index.js: ${e.message}`);
  process.exit(1);
}

// Check core exports
const requiredExports = [
  "createInitialState",
  "gameReducer",
  "getKeyForAction",
  "getAllMappings",
  "isKeyBound",
  "getActionForKey",
  "isPaused",
  "pauseGame",
  "resumeGame",
  "addScore",
  "resetScore",
  "getScore",
  "formatScore",
  "updateEnemyAI",
  "updateAllEnemies",
  "createEnemy",
  "distance",
  "serializeState",
  "deserializeState",
  "isValidState",
  "getCurrentVersion",
];

for (const name of requiredExports) {
  assert.ok(typeof mod[name] === "function", `Missing export: ${name}`);
}
console.log(`   OK: ${requiredExports.length} exports verified`);

// 4. Functional checks
console.log("4. Running functional checks...");

// Create initial state
const state = mod.createInitialState();
assert.ok(state, "createInitialState should return a state");
assert.strictEqual(typeof state.tick, "number");
assert.strictEqual(state.paused, false);
assert.ok(state.player);
assert.ok(Array.isArray(state.enemies));
assert.ok(state.inputMap);
assert.ok(state.settings);

// Game reducer
const ticked = mod.gameReducer(state, { type: "TICK" });
assert.strictEqual(ticked.tick, 1);

// Pause
const paused = mod.pauseGame(state);
assert.strictEqual(paused.paused, true);
const resumed = mod.resumeGame(paused);
assert.strictEqual(resumed.paused, false);

// Score
const scored = mod.addScore(state, 100);
assert.strictEqual(mod.getScore(scored), 100);
const pausedScore = mod.addScore(paused, 999);
assert.strictEqual(mod.getScore(pausedScore), 0); // Should not add when paused

// Enemy
const enemy = mod.createEnemy("test", 10, 10, 2);
assert.strictEqual(enemy.id, "test");
assert.strictEqual(enemy.speed, 2);

// Save/Load roundtrip
const saved = mod.serializeState(state);
const loaded = mod.deserializeState(saved);
assert.ok(loaded !== null);
assert.strictEqual(loaded.tick, state.tick);

console.log("   OK: All functional checks passed");

console.log("\n=== Baseline Sanity: ALL PASS ===");
process.exit(0);
