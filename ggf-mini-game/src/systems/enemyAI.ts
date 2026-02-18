/**
 * Enemy AI system.
 * Simple behavior functions for enemy entities.
 */

import type { Enemy, GameState, Vec2 } from "../core/gameState.js";

/**
 * Calculates distance between two points.
 */
export function distance(a: Vec2, b: Vec2): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/**
 * Updates a single enemy's AI for one tick.
 * Current behavior: if player is within 100 units, move toward player.
 * Otherwise, stay idle.
 */
export function updateEnemyAI(
  enemy: Enemy,
  state: GameState
): Enemy {
  if (enemy.state === "dead") return enemy;

  const dist = distance(enemy.position, state.player.position);

  if (dist < 100) {
    // Move toward player
    const dx = state.player.position.x - enemy.position.x;
    const dy = state.player.position.y - enemy.position.y;
    const norm = Math.sqrt(dx * dx + dy * dy) || 1;
    return {
      ...enemy,
      state: "chase",
      position: {
        x: enemy.position.x + (dx / norm) * enemy.speed,
        y: enemy.position.y + (dy / norm) * enemy.speed,
      },
    };
  }

  return { ...enemy, state: "idle" };
}

/**
 * Updates all enemies in the state.
 */
export function updateAllEnemies(state: GameState): GameState {
  if (state.paused) return state;
  return {
    ...state,
    enemies: state.enemies.map((e) => updateEnemyAI(e, state)),
  };
}

/**
 * Creates a new enemy with default values.
 */
export function createEnemy(
  id: string,
  x: number,
  y: number,
  speed: number = 2
): Enemy {
  return {
    id,
    position: { x, y },
    speed,
    hp: 50,
    maxHp: 50,
    state: "idle",
  };
}
