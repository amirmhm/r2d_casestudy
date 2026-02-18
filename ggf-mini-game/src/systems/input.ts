/**
 * Input mapping system.
 * Maps logical actions to physical keys.
 */

import type { GameState } from "../core/gameState.js";

export type ActionName =
  | "moveUp"
  | "moveDown"
  | "moveLeft"
  | "moveRight"
  | "jump"
  | "attack"
  | "pause";

/**
 * Returns the current key bound to a logical action.
 */
export function getKeyForAction(
  state: GameState,
  action: ActionName
): string | undefined {
  return state.inputMap[action];
}

/**
 * Returns all current input mappings.
 */
export function getAllMappings(
  state: GameState
): Record<string, string> {
  return { ...state.inputMap };
}

/**
 * Checks if a given key code is already bound to any action.
 */
export function isKeyBound(
  state: GameState,
  keyCode: string
): boolean {
  return Object.values(state.inputMap).includes(keyCode);
}

/**
 * Returns the action name bound to a specific key, if any.
 */
export function getActionForKey(
  state: GameState,
  keyCode: string
): string | undefined {
  for (const [action, key] of Object.entries(state.inputMap)) {
    if (key === keyCode) return action;
  }
  return undefined;
}
