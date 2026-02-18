/**
 * Demo: runs a tiny 5-tick simulation and prints output.
 */

import {
  createInitialState,
  gameReducer,
  createEnemy,
  updateAllEnemies,
  addScore,
  serializeState,
  deserializeState,
  isPaused,
  pauseGame,
  getScore,
  formatScore,
} from "./index.js";

function main(): void {
  console.log("=== GGF Mini-Game Demo ===\n");

  // Initialize
  let state = createInitialState();
  console.log(`Initial tick: ${state.tick}, score: ${getScore(state)}`);

  // Spawn enemies
  state = gameReducer(state, {
    type: "SPAWN_ENEMY",
    enemy: createEnemy("goblin_1", 50, 50, 3),
  });
  state = gameReducer(state, {
    type: "SPAWN_ENEMY",
    enemy: createEnemy("goblin_2", 200, 200, 1),
  });
  console.log(`Spawned ${state.enemies.length} enemies`);

  // Run 5 ticks
  for (let i = 0; i < 5; i++) {
    state = gameReducer(state, { type: "TICK" });
    state = updateAllEnemies(state);
    state = addScore(state, 10);
    console.log(
      `  Tick ${state.tick}: score=${formatScore(getScore(state))}, ` +
        `enemies=${state.enemies.map((e) => `${e.id}(${e.state})`).join(", ")}`
    );
  }

  // Test pause
  state = pauseGame(state);
  console.log(`\nPaused: ${isPaused(state)}`);
  state = addScore(state, 999); // Should NOT add (paused)
  console.log(`Score after paused addScore: ${getScore(state)} (should be 50)`);

  // Save & Load
  const saved = serializeState(state);
  const loaded = deserializeState(saved);
  console.log(`\nSave/Load roundtrip: ${loaded !== null ? "OK" : "FAIL"}`);
  console.log(`Loaded score: ${loaded ? getScore(loaded) : "N/A"}`);

  console.log("\n=== Demo Complete ===");
}

main();
