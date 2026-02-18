#!/usr/bin/env node

/**
 * Generic check runner for GGF LLM Systems Case.
 *
 * Usage:
 *   node eval/checks/run_check.mjs --task task_01
 *   node eval/checks/run_check.mjs --task task_01 --workdir ./some/path
 *
 * This script:
 * 1. Builds the project (npm run build in workdir)
 * 2. Dynamically imports the compiled dist/
 * 3. Runs task-specific assertions
 */

import { execSync } from "node:child_process";
import { strict as assert } from "node:assert";
import { resolve, join } from "node:path";
import { pathToFileURL } from "node:url";
import { existsSync } from "node:fs";

// Parse CLI args
const args = process.argv.slice(2);
let taskId = null;
let workDir = null;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--task" && args[i + 1]) {
    taskId = args[i + 1];
    i++;
  } else if (args[i] === "--workdir" && args[i + 1]) {
    workDir = args[i + 1];
    i++;
  }
}

if (!taskId) {
  console.error("Usage: node run_check.mjs --task <task_id> [--workdir <path>]");
  process.exit(1);
}

// Resolve working directory
const miniGameDir = workDir
  ? resolve(workDir)
  : resolve("ggf-mini-game");

if (!existsSync(miniGameDir)) {
  console.error(`Mini-game directory not found: ${miniGameDir}`);
  process.exit(1);
}

// Build
console.log(`Building project in ${miniGameDir}...`);
try {
  execSync("npm run build", { cwd: miniGameDir, stdio: "pipe", shell: true });
} catch (e) {
  console.error("Build failed:");
  console.error(e.stderr?.toString() || e.message);
  process.exit(1);
}

// Import compiled modules
const distDir = join(miniGameDir, "dist");
const indexUrl = pathToFileURL(join(distDir, "index.js")).href;

let mod;
try {
  mod = await import(indexUrl);
} catch (e) {
  console.error(`Failed to import dist/index.js: ${e.message}`);
  process.exit(1);
}

// ============================================================
// TASK CHECKS
// ============================================================

const checks = {
  // --------------------------------------------------------
  // Task 01: Pause Toggle Behavior
  // --------------------------------------------------------
  task_01() {
    console.log("Checking task_01: Pause Toggle Behavior");

    assert.ok(typeof mod.togglePause === "function", "togglePause must be exported as a function");

    const state = mod.createInitialState();
    assert.strictEqual(state.paused, false, "Initial state should not be paused");

    const paused = mod.togglePause(state);
    assert.strictEqual(paused.paused, true, "togglePause on unpaused should set paused=true");
    assert.strictEqual(state.paused, false, "Original state must not be mutated");

    const unpaused = mod.togglePause(paused);
    assert.strictEqual(unpaused.paused, false, "togglePause on paused should set paused=false");

    console.log("  PASS: togglePause works correctly");
  },

  // --------------------------------------------------------
  // Task 02: Input Remapping
  // --------------------------------------------------------
  task_02() {
    console.log("Checking task_02: Input Remapping");

    assert.ok(typeof mod.remapKey === "function", "remapKey must be exported as a function");

    const state = mod.createInitialState();
    const original = state.inputMap.jump;

    const remapped = mod.remapKey(state, "jump", "KeyK");
    assert.strictEqual(remapped.inputMap.jump, "KeyK", "jump should be remapped to KeyK");
    assert.strictEqual(state.inputMap.jump, original, "Original state must not be mutated");

    // Non-existent action
    const unchanged = mod.remapKey(state, "nonexistent_action", "KeyX");
    assert.deepStrictEqual(unchanged.inputMap, state.inputMap, "Non-existent action should return unchanged state");

    console.log("  PASS: remapKey works correctly");
  },

  // --------------------------------------------------------
  // Task 03: Score Combo Multiplier
  // --------------------------------------------------------
  task_03() {
    console.log("Checking task_03: Score Combo Multiplier");

    assert.ok(typeof mod.addComboScore === "function", "addComboScore must be exported");

    const state = mod.createInitialState();

    // streak=0 -> 1.0x
    const s0 = mod.addComboScore(state, 100, 0);
    assert.strictEqual(s0.player.score, 100, "streak=0: 100 * 1.0 = 100");
    assert.strictEqual(s0.player.combo, 0, "combo should be set to streak");

    // streak=5 -> 1.5x
    const s5 = mod.addComboScore(state, 100, 5);
    assert.strictEqual(s5.player.score, 150, "streak=5: 100 * 1.5 = 150");
    assert.strictEqual(s5.player.combo, 5);

    // streak=10 -> 2.0x
    const s10 = mod.addComboScore(state, 100, 10);
    assert.strictEqual(s10.player.score, 200, "streak=10: 100 * 2.0 = 200");

    // streak=20 -> 3.0x (cap)
    const s20 = mod.addComboScore(state, 100, 20);
    assert.strictEqual(s20.player.score, 300, "streak=20: 100 * 3.0 = 300 (capped)");

    // streak=100 -> still 3.0x
    const s100 = mod.addComboScore(state, 100, 100);
    assert.strictEqual(s100.player.score, 300, "streak=100: still 3.0x cap");

    // Paused state
    const pausedState = mod.pauseGame(state);
    const sp = mod.addComboScore(pausedState, 100, 5);
    assert.strictEqual(sp.player.score, 0, "Should not add score when paused");

    console.log("  PASS: addComboScore works correctly");
  },

  // --------------------------------------------------------
  // Task 04: Enemy Patrol Mode
  // --------------------------------------------------------
  task_04() {
    console.log("Checking task_04: Enemy Patrol Mode");

    // createEnemy should include patrolRadius
    const enemy = mod.createEnemy("e1", 500, 500, 2);
    assert.ok("patrolRadius" in enemy, "Enemy must have patrolRadius property");
    assert.strictEqual(enemy.patrolRadius, 50, "Default patrolRadius should be 50");

    // Far enemy -> patrol state (not idle)
    const state = mod.createInitialState();
    state.enemies = [enemy];
    const updated = mod.updateEnemyAI(enemy, state);
    assert.strictEqual(updated.state, "patrol", "Far enemy should be in patrol state");

    // Close enemy -> chase
    const closeEnemy = mod.createEnemy("e2", 5, 5, 2);
    const updatedClose = mod.updateEnemyAI(closeEnemy, state);
    assert.strictEqual(updatedClose.state, "chase", "Close enemy should chase");

    // Custom chaseThreshold
    const midEnemy = mod.createEnemy("e3", 80, 0, 2);
    const withCustomThreshold = mod.updateEnemyAI(midEnemy, state, 50);
    assert.strictEqual(withCustomThreshold.state, "patrol", "Enemy at 80 with threshold 50 should patrol");

    const withDefaultThreshold = mod.updateEnemyAI(midEnemy, state, 200);
    assert.strictEqual(withDefaultThreshold.state, "chase", "Enemy at 80 with threshold 200 should chase");

    console.log("  PASS: Enemy patrol mode works correctly");
  },

  // --------------------------------------------------------
  // Task 05: Save System V2
  // --------------------------------------------------------
  task_05() {
    console.log("Checking task_05: Save System V2");

    assert.strictEqual(mod.getCurrentVersion(), 2, "getCurrentVersion should return 2");

    const state = mod.createInitialState();
    const serialized = mod.serializeState(state);
    const parsed = JSON.parse(serialized);
    assert.strictEqual(parsed.version, 2, "Serialized save should be version 2");
    assert.ok(parsed.metadata, "V2 save should have metadata");
    assert.strictEqual(typeof parsed.metadata.playTime, "number", "metadata.playTime should be number");
    assert.strictEqual(typeof parsed.metadata.saveSlot, "number", "metadata.saveSlot should be number");

    // Deserialize v2
    const loaded = mod.deserializeState(serialized);
    assert.ok(loaded !== null, "Should deserialize v2 saves");

    // Backward compat: v1 save
    const v1Save = JSON.stringify({ version: 1, timestamp: Date.now(), state });
    const loadedV1 = mod.deserializeState(v1Save);
    assert.ok(loadedV1 !== null, "Should deserialize v1 saves (backward compat)");

    console.log("  PASS: Save system v2 works correctly");
  },

  // --------------------------------------------------------
  // Task 06: Difficulty Affects Enemy Speed
  // --------------------------------------------------------
  task_06() {
    console.log("Checking task_06: Difficulty Speed Multiplier");

    assert.ok(typeof mod.getDifficultySpeedMultiplier === "function",
      "getDifficultySpeedMultiplier must be exported");

    // Formula: 0.5 + (difficulty / 10) * 1.5
    const eps = 0.01;
    assert.ok(Math.abs(mod.getDifficultySpeedMultiplier(1) - 0.65) < eps,
      "difficulty 1 -> ~0.65");
    assert.ok(Math.abs(mod.getDifficultySpeedMultiplier(5) - 1.25) < eps,
      "difficulty 5 -> 1.25");
    assert.ok(Math.abs(mod.getDifficultySpeedMultiplier(10) - 2.0) < eps,
      "difficulty 10 -> 2.0");

    // Test that enemy movement is affected by difficulty
    const stateHigh = mod.gameReducer(mod.createInitialState(), { type: "UPDATE_SETTINGS", settings: { difficulty: 10 } });
    const enemy = mod.createEnemy("e1", 50, 0, 2);
    const updatedHigh = mod.updateEnemyAI(enemy, stateHigh);

    const stateLow = mod.gameReducer(mod.createInitialState(), { type: "UPDATE_SETTINGS", settings: { difficulty: 1 } });
    const updatedLow = mod.updateEnemyAI(enemy, stateLow);

    // Higher difficulty should move enemy more
    const distHigh = mod.distance(enemy.position, updatedHigh.position);
    const distLow = mod.distance(enemy.position, updatedLow.position);
    assert.ok(distHigh > distLow, "Higher difficulty should make enemies move faster");

    console.log("  PASS: Difficulty speed multiplier works correctly");
  },

  // --------------------------------------------------------
  // Task 07: Event Log System
  // --------------------------------------------------------
  task_07() {
    console.log("Checking task_07: Event Log System");

    assert.ok(typeof mod.createEventLog === "function", "createEventLog must be exported");
    assert.ok(typeof mod.logEvent === "function", "logEvent must be exported");
    assert.ok(typeof mod.getRecentEvents === "function", "getRecentEvents must be exported");

    // Create log
    const log = mod.createEventLog(5);
    assert.strictEqual(log.maxSize, 5);
    assert.ok(Array.isArray(log.events));
    assert.strictEqual(log.events.length, 0);

    // Default maxSize
    const defaultLog = mod.createEventLog();
    assert.strictEqual(defaultLog.maxSize, 50, "Default maxSize should be 50");

    // Log events
    let current = log;
    for (let i = 0; i < 7; i++) {
      current = mod.logEvent(current, { type: `event_${i}`, timestamp: i });
    }
    assert.strictEqual(current.events.length, 5, "Should trim to maxSize (5)");
    assert.strictEqual(current.events[0].type, "event_2", "FIFO: oldest trimmed");

    // getRecentEvents
    const recent = mod.getRecentEvents(current, 3);
    assert.strictEqual(recent.length, 3);
    assert.strictEqual(recent[recent.length - 1].type, "event_6", "Most recent should be last");

    // GameState should have eventLog
    const state = mod.createInitialState();
    assert.ok(state.eventLog, "GameState should have eventLog field");
    assert.ok(state.eventLog.events, "eventLog should have events array");

    console.log("  PASS: Event log system works correctly");
  },

  // --------------------------------------------------------
  // Task 08: Ability Cooldown Mechanism
  // --------------------------------------------------------
  task_08() {
    console.log("Checking task_08: Ability Cooldown Mechanism");

    assert.ok(typeof mod.createCooldownManager === "function", "createCooldownManager must be exported");
    assert.ok(typeof mod.startCooldown === "function", "startCooldown must be exported");
    assert.ok(typeof mod.tickCooldowns === "function", "tickCooldowns must be exported");
    assert.ok(typeof mod.isOnCooldown === "function", "isOnCooldown must be exported");
    assert.ok(typeof mod.getCooldownRemaining === "function", "getCooldownRemaining must be exported");

    // Create
    let mgr = mod.createCooldownManager();
    assert.ok(Array.isArray(mgr), "Manager should be an array");
    assert.strictEqual(mgr.length, 0, "Should start empty");

    // Start cooldown
    mgr = mod.startCooldown(mgr, "fireball", 10);
    assert.strictEqual(mgr.length, 1);
    assert.strictEqual(mod.isOnCooldown(mgr, "fireball"), true);
    assert.strictEqual(mod.getCooldownRemaining(mgr, "fireball"), 10);

    // Non-existent ability
    assert.strictEqual(mod.isOnCooldown(mgr, "ice_blast"), false);
    assert.strictEqual(mod.getCooldownRemaining(mgr, "ice_blast"), 0);

    // Tick cooldowns
    mgr = mod.tickCooldowns(mgr);
    assert.strictEqual(mod.getCooldownRemaining(mgr, "fireball"), 9);

    // Tick to zero
    for (let i = 0; i < 20; i++) {
      mgr = mod.tickCooldowns(mgr);
    }
    assert.strictEqual(mod.getCooldownRemaining(mgr, "fireball"), 0, "Should not go below 0");
    assert.strictEqual(mod.isOnCooldown(mgr, "fireball"), false, "Should be off cooldown at 0");

    // Reset existing cooldown
    mgr = mod.startCooldown(mgr, "fireball", 5);
    assert.strictEqual(mod.getCooldownRemaining(mgr, "fireball"), 5, "Should reset to new duration");

    console.log("  PASS: Cooldown mechanism works correctly");
  },

  // --------------------------------------------------------
  // Task 09: Deterministic RNG
  // --------------------------------------------------------
  task_09() {
    console.log("Checking task_09: Deterministic RNG");

    assert.ok(typeof mod.createRNG === "function", "createRNG must be exported");
    assert.ok(typeof mod.randomInt === "function", "randomInt must be exported");
    assert.ok(typeof mod.randomChoice === "function", "randomChoice must be exported");

    // Determinism: same seed -> same sequence
    const rng1 = mod.createRNG(42);
    const rng2 = mod.createRNG(42);
    const seq1 = [rng1(), rng1(), rng1(), rng1(), rng1()];
    const seq2 = [rng2(), rng2(), rng2(), rng2(), rng2()];
    assert.deepStrictEqual(seq1, seq2, "Same seed must produce same sequence");

    // Range check: all values in [0, 1)
    const rng3 = mod.createRNG(123);
    for (let i = 0; i < 100; i++) {
      const val = rng3();
      assert.ok(val >= 0 && val < 1, `Value ${val} must be in [0, 1)`);
    }

    // Different seeds -> different sequences
    const rng4 = mod.createRNG(1);
    const rng5 = mod.createRNG(999);
    const s4 = [rng4(), rng4(), rng4()];
    const s5 = [rng5(), rng5(), rng5()];
    assert.notDeepStrictEqual(s4, s5, "Different seeds should produce different sequences");

    // randomInt
    const rng6 = mod.createRNG(42);
    for (let i = 0; i < 50; i++) {
      const val = mod.randomInt(rng6, 1, 10);
      assert.ok(val >= 1 && val <= 10 && Number.isInteger(val),
        `randomInt(1, 10) returned ${val}`);
    }

    // randomChoice
    const rng7 = mod.createRNG(42);
    const choices = ["a", "b", "c"];
    for (let i = 0; i < 20; i++) {
      const pick = mod.randomChoice(rng7, choices);
      assert.ok(choices.includes(pick), `randomChoice returned '${pick}' not in choices`);
    }

    // Deterministic randomChoice
    const rng8 = mod.createRNG(42);
    const rng9 = mod.createRNG(42);
    const picks1 = Array.from({ length: 5 }, () => mod.randomChoice(rng8, choices));
    const picks2 = Array.from({ length: 5 }, () => mod.randomChoice(rng9, choices));
    assert.deepStrictEqual(picks1, picks2, "randomChoice must be deterministic");

    console.log("  PASS: Deterministic RNG works correctly");
  },

  // --------------------------------------------------------
  // Task 10: Settings Validation
  // --------------------------------------------------------
  task_10() {
    console.log("Checking task_10: Settings Validation");

    assert.ok(typeof mod.validateSettings === "function", "validateSettings must be exported");

    // Empty object -> all defaults
    const defaults = mod.validateSettings({});
    assert.strictEqual(defaults.difficulty, 5, "Default difficulty = 5");
    assert.strictEqual(defaults.soundVolume, 0.8, "Default soundVolume = 0.8");
    assert.strictEqual(defaults.musicVolume, 0.6, "Default musicVolume = 0.6");
    assert.strictEqual(defaults.showFps, false, "Default showFps = false");

    // Valid values preserved
    const valid = mod.validateSettings({ difficulty: 3, soundVolume: 0.5, musicVolume: 0.3, showFps: true });
    assert.strictEqual(valid.difficulty, 3);
    assert.strictEqual(valid.soundVolume, 0.5);
    assert.strictEqual(valid.musicVolume, 0.3);
    assert.strictEqual(valid.showFps, true);

    // Out of range
    const outOfRange = mod.validateSettings({ difficulty: 15, soundVolume: -1, musicVolume: 2 });
    assert.strictEqual(outOfRange.difficulty, 5, "Out-of-range difficulty -> default");
    assert.strictEqual(outOfRange.soundVolume, 0.8, "Negative volume -> default");
    assert.strictEqual(outOfRange.musicVolume, 0.6, "Volume > 1 -> default");

    // Wrong types
    const wrongTypes = mod.validateSettings({ difficulty: "high", showFps: "yes" });
    assert.strictEqual(wrongTypes.difficulty, 5, "Non-number difficulty -> default");
    assert.strictEqual(wrongTypes.showFps, false, "Non-boolean showFps -> default");

    console.log("  PASS: Settings validation works correctly");
  },
};

// ============================================================
// RUN
// ============================================================

if (!(taskId in checks)) {
  console.error(`Unknown task: ${taskId}`);
  console.error(`Available tasks: ${Object.keys(checks).join(", ")}`);
  process.exit(1);
}

try {
  checks[taskId]();
  console.log(`\n[PASS] ${taskId}`);
  process.exit(0);
} catch (e) {
  console.error(`\n[FAIL] ${taskId}`);
  console.error(e.message);
  if (e.stack) {
    console.error(e.stack.split("\n").slice(0, 5).join("\n"));
  }
  process.exit(1);
}
