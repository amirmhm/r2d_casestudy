/**
 * Score management system.
 * Handles points, high scores, and display formatting.
 */

import type { GameState } from "../core/gameState.js";

/**
 * Adds points to the player's score.
 * Does NOT apply any combo multiplier — just raw points.
 */
export function addScore(state: GameState, points: number): GameState {
  if (state.paused) return state;
  return {
    ...state,
    player: {
      ...state.player,
      score: state.player.score + points,
    },
  };
}

/**
 * Resets score to zero.
 */
export function resetScore(state: GameState): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      score: 0,
      combo: 0,
    },
  };
}

/**
 * Returns the current score.
 */
export function getScore(state: GameState): number {
  return state.player.score;
}

/**
 * Formats score for display (e.g., 1500 -> "1,500").
 */
export function formatScore(score: number): string {
  return score.toLocaleString("en-US");
}
