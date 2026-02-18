/**
 * GGF Mini-Game — Public API
 * All exports for the game-system modules.
 */

// Core
export {
  createInitialState,
  gameReducer,
  type GameState,
  type GameAction,
  type GameSettings,
  type Enemy,
  type Player,
  type Vec2,
} from "./core/gameState.js";

// Systems
export {
  getKeyForAction,
  getAllMappings,
  isKeyBound,
  getActionForKey,
  type ActionName,
} from "./systems/input.js";

export {
  isPaused,
  pauseGame,
  resumeGame,
} from "./systems/pause.js";

export {
  addScore,
  resetScore,
  getScore,
  formatScore,
} from "./systems/score.js";

export {
  updateEnemyAI,
  updateAllEnemies,
  createEnemy,
  distance,
} from "./systems/enemyAI.js";

export {
  serializeState,
  deserializeState,
  isValidState,
  getCurrentVersion,
  type SaveData,
} from "./systems/save.js";
