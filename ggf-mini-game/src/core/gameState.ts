/**
 * Core game state shape and reducer.
 * All game state lives here — systems read and transform it.
 */

export interface Vec2 {
  x: number;
  y: number;
}

export interface Enemy {
  id: string;
  position: Vec2;
  speed: number;
  hp: number;
  maxHp: number;
  state: "idle" | "patrol" | "chase" | "dead";
}

export interface Player {
  position: Vec2;
  hp: number;
  maxHp: number;
  score: number;
  combo: number;
}

export interface GameState {
  tick: number;
  paused: boolean;
  player: Player;
  enemies: Enemy[];
  inputMap: Record<string, string>;
  settings: GameSettings;
}

export interface GameSettings {
  difficulty: number;       // 1-10 scale
  soundVolume: number;      // 0.0-1.0
  musicVolume: number;      // 0.0-1.0
  showFps: boolean;
}

export type GameAction =
  | { type: "TICK" }
  | { type: "SET_PAUSED"; paused: boolean }
  | { type: "ADD_SCORE"; points: number }
  | { type: "SPAWN_ENEMY"; enemy: Enemy }
  | { type: "REMOVE_ENEMY"; enemyId: string }
  | { type: "MOVE_PLAYER"; position: Vec2 }
  | { type: "UPDATE_SETTINGS"; settings: Partial<GameSettings> };

/**
 * Creates a fresh default game state.
 */
export function createInitialState(): GameState {
  return {
    tick: 0,
    paused: false,
    player: {
      position: { x: 0, y: 0 },
      hp: 100,
      maxHp: 100,
      score: 0,
      combo: 0,
    },
    enemies: [],
    inputMap: {
      moveUp: "KeyW",
      moveDown: "KeyS",
      moveLeft: "KeyA",
      moveRight: "KeyD",
      jump: "Space",
      attack: "KeyJ",
      pause: "Escape",
    },
    settings: {
      difficulty: 5,
      soundVolume: 0.8,
      musicVolume: 0.6,
      showFps: false,
    },
  };
}

/**
 * Pure reducer — applies an action to state and returns new state.
 */
export function gameReducer(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case "TICK":
      if (state.paused) return state;
      return { ...state, tick: state.tick + 1 };

    case "SET_PAUSED":
      return { ...state, paused: action.paused };

    case "ADD_SCORE":
      return {
        ...state,
        player: {
          ...state.player,
          score: state.player.score + action.points,
        },
      };

    case "SPAWN_ENEMY":
      return { ...state, enemies: [...state.enemies, action.enemy] };

    case "REMOVE_ENEMY":
      return {
        ...state,
        enemies: state.enemies.filter((e) => e.id !== action.enemyId),
      };

    case "MOVE_PLAYER":
      return {
        ...state,
        player: { ...state.player, position: action.position },
      };

    case "UPDATE_SETTINGS":
      return {
        ...state,
        settings: { ...state.settings, ...action.settings },
      };

    default:
      return state;
  }
}
