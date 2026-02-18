/**
 * Save / Load system.
 * Serializes and deserializes game state to JSON.
 */

import type { GameState } from "../core/gameState.js";
import { createInitialState } from "../core/gameState.js";

export interface SaveData {
  version: 1;
  timestamp: number;
  state: GameState;
}

/**
 * Serializes a game state into a save-data JSON string.
 */
export function serializeState(state: GameState): string {
  const saveData: SaveData = {
    version: 1,
    timestamp: Date.now(),
    state,
  };
  return JSON.stringify(saveData);
}

/**
 * Deserializes a save-data JSON string back into game state.
 * Returns null if the data is invalid.
 */
export function deserializeState(json: string): GameState | null {
  try {
    const data = JSON.parse(json) as SaveData;
    if (!data || typeof data.version !== "number") return null;
    if (data.version === 1 && data.state) {
      return data.state;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Validates that a state object has the expected shape.
 */
export function isValidState(state: unknown): state is GameState {
  if (!state || typeof state !== "object") return false;
  const s = state as Record<string, unknown>;
  return (
    typeof s.tick === "number" &&
    typeof s.paused === "boolean" &&
    s.player !== undefined &&
    Array.isArray(s.enemies)
  );
}

/**
 * Returns the current save data version.
 */
export function getCurrentVersion(): number {
  return 1;
}
