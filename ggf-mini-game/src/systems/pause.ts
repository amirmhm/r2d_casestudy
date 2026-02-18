/**
 * Pause management system.
 * Controls game pause state.
 */

import type { GameState } from "../core/gameState.js";

/**
 * Returns whether the game is currently paused.
 */
export function isPaused(state: GameState): boolean {
  return state.paused;
}

/**
 * Returns a new state with paused set to true.
 */
export function pauseGame(state: GameState): GameState {
  return { ...state, paused: true };
}

/**
 * Returns a new state with paused set to false.
 */
export function resumeGame(state: GameState): GameState {
  return { ...state, paused: false };
}
