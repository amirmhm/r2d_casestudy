# GGF LLM Systems Case v2.0 — DETAILED GUIDE
# GGF LLM Sistemleri Case v2.0 — DETAYLI REHBER

> This document explains every component of the case in extreme detail.
> Bu dokuman case'in her bileseni asiri detayli olarak aciklar.

---

## TABLE OF CONTENTS / ICINDEKILER

1. [What Is This? / Bu Nedir?](#1-what-is-this--bu-nedir)
2. [Architecture Overview / Mimari Genel Bakis](#2-architecture-overview--mimari-genel-bakis)
3. [The Mini-Game Codebase / Mini-Oyun Kod Tabani](#3-the-mini-game-codebase--mini-oyun-kod-tabani)
4. [The 10 Tasks in Detail / 10 Gorev Detayli](#4-the-10-tasks-in-detail--10-gorev-detayli)
5. [The Evaluation System / Degerlendirme Sistemi](#5-the-evaluation-system--degerlendirme-sistemi)
6. [The Python Solution / Python Cozum Iskeleti](#6-the-python-solution--python-cozum-iskeleti)
7. [RAG System Explained / RAG Sistemi Aciklamasi](#7-rag-system-explained--rag-sistemi-aciklamasi)
8. [Patch Generation Pipeline / Yama Uretim Hatti](#8-patch-generation-pipeline--yama-uretim-hatti)
9. [Diff Guard / Fark Korumasi](#9-diff-guard--fark-korumasi)
10. [How to Run Everything / Her Seyi Nasil Calistirilir](#10-how-to-run-everything--her-seyi-nasil-calistirilir)
11. [Evaluation Flow Diagram / Degerlendirme Akis Semasi](#11-evaluation-flow-diagram--degerlendirme-akis-semasi)
12. [Configuration Reference / Konfigurasyon Referansi](#12-configuration-reference--konfigurasyon-referansi)
13. [The 4 Evaluation Phases / 4 Degerlendirme Fazi](#13-the-4-evaluation-phases--4-degerlendirme-fazi)
14. [Fine-Tuning Guide / Fine-Tuning Rehberi](#14-fine-tuning-guide--fine-tuning-rehberi)
15. [Retrieval Metrics Explained / Geri Getirme Metrikleri](#15-retrieval-metrics-explained--geri-getirme-metrikleri)
16. [Troubleshooting / Sorun Giderme](#16-troubleshooting--sorun-giderme)

---

## 1. What Is This? / Bu Nedir?

### English

The **GGF LLM Systems Case v2.0** is a comprehensive take-home technical assessment that evaluates four key competencies through a **4-phase, 100-point scoring system**:

1. **RAG Pipeline & Retrieval Quality** (30 pts) — BM25, hybrid retrieval, AST-aware chunking
2. **Prompt Engineering & Structured Output** (20 pts) — CoT templates, Pydantic models, JSON extraction
3. **Fine-Tuning & Training Data Curation** (30 pts) — Data preparation, OpenAI API, model comparison
4. **Analytics, Experiment Design & Failure Analysis** (20 pts) — Statistical testing, root cause analysis

The candidate receives:
- **A small TypeScript codebase** (`ggf-mini-game/`) representing game system modules
- **10 modification tasks** defined in `eval/tasks.json` (tagged by phase and difficulty)
- **A Python solution skeleton** (`solution/`) with baseline implementations
- **Gold labels** (`eval/gold_labels.json`) for measuring retrieval quality
- **Training data** (`eval/training_data/`) for fine-tuning experiments
- **An automated evaluation suite** with per-phase checks

### Turkce

**GGF LLM Sistemleri Case v2.0**, dort temel yetkinligi **4 fazli, 100 puanlik puanlama sistemi** ile degerlendiren kapsamli bir teknik degerlendirmedir:

1. **RAG Hatti ve Geri Getirme Kalitesi** (30 puan) — BM25, hibrit geri getirme, AST parcalama
2. **Prompt Muhendisligi ve Yapilandirilmis Cikti** (20 puan) — CoT sablonlari, Pydantic modeller
3. **Fine-Tuning ve Egitim Verisi Duzenleme** (30 puan) — Veri hazirlama, OpenAI API, model karsilastirma
4. **Analitik, Deney Tasarimi ve Hata Analizi** (20 puan) — Istatistiksel test, kok neden analizi

Test edilen beceriler: **RAG uzmanligi, model fine-tuning yetkinligi, prompt muhendisligi, analitik dusunme**.

---

## 2. Architecture Overview / Mimari Genel Bakis

### System Architecture / Sistem Mimarisi

```
+-------------------+     +------------------+     +------------------+
|   ggf-mini-game   |     |     solution     |     |       eval       |
|   (TypeScript)    |     |     (Python)     |     |    (Node.js)     |
+-------------------+     +------------------+     +------------------+
|                   |     |                  |     |                  |
| src/              |---->| RAG Indexer      |     | tasks.json       |
|   core/           |     |   - chunk files  |     |   - 10 tasks     |
|   systems/        |     |   - extract syms |     |   - criteria     |
|   index.ts        |     |                  |     |                  |
|                   |     | RAG Retriever    |     | checks/          |
| dist/ (compiled)  |     |   - keyword      |     |   run_check.mjs  |
|                   |     |   - embedding    |     |   baseline.mjs   |
+-------------------+     |                  |     |                  |
                          | LLM Client       |     | outputs/         |
                          |   - OpenAI compat |     |   (gitignored)   |
                          |   - patch prompt  |     |                  |
                          |                  |     | run_eval.sh      |
                          | Diff Guard       |     | run_eval.ps1     |
                          |   - size check   |     +------------------+
                          |   - file count   |
                          |                  |
                          | Patch Applier    |
                          |   - git apply    |
                          |   - manual fb    |
                          |                  |
                          | Eval Runner      |
                          |   - loop tasks   |
                          |   - collect data |
                          +------------------+
```

### Data Flow / Veri Akisi

```
1. INDEX:  TypeScript files --> Chunked index (JSON)
2. TASK:   tasks.json --> Pick task --> Extract query
3. RETRIEVE: Query --> Index --> Top-K code snippets
4. PROMPT: Task + Context --> LLM prompt
5. GENERATE: LLM --> Unified diff (patch)
6. GUARD:  Diff --> Size check (max 250 lines, 6 files)
7. APPLY:  Diff --> git apply --> Modified codebase
8. BUILD:  npm run build --> Compiled JS
9. CHECK:  node run_check.mjs --> PASS/FAIL
10. REPORT: Collect all results --> summary.json
```

---

## 3. The Mini-Game Codebase / Mini-Oyun Kod Tabani

### English

The `ggf-mini-game/` directory contains a **minimal TypeScript project** representing game system modules. It is NOT a playable game — it's a collection of **pure functions** that manage game state.

### Turkce

`ggf-mini-game/` dizini, oyun sistemi modullerini temsil eden **minimal bir TypeScript projesi** icerir. Oynanabilir bir oyun DEGILDIR — oyun durumunu yoneten **saf fonksiyonlar** koleksiyonudur.

### File-by-File Breakdown / Dosya Dosya Aciklama

#### `src/core/gameState.ts`
The heart of the system. Defines:
- **`GameState`** interface: The complete state shape (tick, paused, player, enemies, inputMap, settings)
- **`GameSettings`** interface: difficulty, volumes, showFps
- **`GameAction`** type: Union of all possible actions (TICK, SET_PAUSED, ADD_SCORE, etc.)
- **`createInitialState()`**: Factory function returning default state
- **`gameReducer(state, action)`**: Pure reducer that applies actions to state

Sistemin kalbi. Tanimlar:
- **`GameState`** arayuzu: Tam durum sekli (tick, paused, player, enemies, inputMap, settings)
- **`gameReducer`**: Aksiyonlari duruma uygulayan saf reducer

#### `src/systems/input.ts`
Input mapping system / Giris esleme sistemi:
- **`getKeyForAction(state, action)`**: Returns the key code bound to a logical action
- **`getAllMappings(state)`**: Returns a copy of all input mappings
- **`isKeyBound(state, keyCode)`**: Checks if a key is already used
- **`getActionForKey(state, keyCode)`**: Reverse lookup — key to action

#### `src/systems/pause.ts`
Pause management / Duraklama yonetimi:
- **`isPaused(state)`**: Returns boolean
- **`pauseGame(state)`**: Returns new state with paused=true
- **`resumeGame(state)`**: Returns new state with paused=false

Note: There is NO `togglePause` function — Task 01 asks candidates to add it!

Not: `togglePause` fonksiyonu YOK — Gorev 01 adaylardan bunu eklemesini istiyor!

#### `src/systems/score.ts`
Score management / Skor yonetimi:
- **`addScore(state, points)`**: Adds raw points (respects pause)
- **`resetScore(state)`**: Resets score and combo to 0
- **`getScore(state)`**: Returns current score
- **`formatScore(score)`**: Formats for display (e.g., 1,500)

Note: No combo multiplier — Task 03 asks to add `addComboScore()`

#### `src/systems/enemyAI.ts`
Enemy AI behaviors / Dusman yapay zekasi:
- **`distance(a, b)`**: Euclidean distance between two Vec2 points
- **`updateEnemyAI(enemy, state)`**: Updates one enemy (chase if <100 units, else idle)
- **`updateAllEnemies(state)`**: Updates all enemies (respects pause)
- **`createEnemy(id, x, y, speed)`**: Factory for new enemies

Note: No patrol mode, no difficulty scaling — Tasks 04, 06 address this

#### `src/systems/save.ts`
Save/Load serialization / Kayit/Yukleme:
- **`serializeState(state)`**: Converts state to JSON string (version 1)
- **`deserializeState(json)`**: Parses JSON back to GameState
- **`isValidState(state)`**: Type guard
- **`getCurrentVersion()`**: Returns 1

Note: No v2, no backward compat — Task 05 requires this upgrade

#### `src/index.ts`
Public API — re-exports everything from all modules. This is what the eval checks import.

#### `src/demo.ts`
A 5-tick simulation that demonstrates all systems working together. Not imported by checks.

---

## 4. The 10 Tasks in Detail / 10 Gorev Detayli

### Task 01: Pause Toggle / Duraklama Gecisi

**What to do / Ne yapilmali:**
Add a `togglePause(state)` function to `src/systems/pause.ts` that flips `state.paused`. Export it from `index.ts`.

**Why it matters / Neden onemli:**
Tests the simplest possible code modification — adding one new function and one export.

**Acceptance checks / Kabul kontrolleri:**
- Function exists and is exported
- `togglePause(unpaused).paused === true`
- `togglePause(paused).paused === false`
- Pure function (no mutation)

**Difficulty / Zorluk:** Easy / Kolay

---

### Task 02: Input Remapping / Giris Yeniden Esleme

**What to do / Ne yapilmali:**
Add `remapKey(state, action, newKeyCode)` to `input.ts`. Updates the inputMap for the given action. Returns unchanged state if action doesn't exist.

**Why it matters / Neden onemli:**
Tests ability to modify state with validation (action must exist).

**Acceptance checks:**
- Function exists/exported
- Remaps correctly
- Handles non-existent action
- No mutation

**Difficulty:** Easy / Kolay

---

### Task 03: Score Combo Multiplier / Skor Kombo Carpani

**What to do:**
Add `addComboScore(state, basePoints, streak)` to `score.ts`. Multiplier formula: `1 + Math.floor(streak / 5) * 0.5`, capped at 3.0x. Update player.combo to streak. Respect pause.

**Why it matters:**
Tests mathematical formula implementation with edge cases (cap, pause).

**Key formula / Onemli formul:**
```
multiplier = Math.min(3.0, 1 + Math.floor(streak / 5) * 0.5)
finalPoints = Math.round(basePoints * multiplier)
```

| Streak | Multiplier | 100 points becomes |
|--------|------------|-------------------|
| 0 | 1.0x | 100 |
| 5 | 1.5x | 150 |
| 10 | 2.0x | 200 |
| 15 | 2.5x | 250 |
| 20+ | 3.0x (cap) | 300 |

**Difficulty:** Medium / Orta

---

### Task 04: Enemy Patrol Mode / Dusman Devriye Modu

**What to do:**
- Add `patrolRadius: number` to Enemy interface (default 50)
- Update `createEnemy` to include patrolRadius
- Modify `updateEnemyAI` to set state="patrol" (instead of "idle") when far from player
- Add optional `chaseThreshold` parameter (default 100)

**Why it matters:**
Tests multi-file modification (gameState.ts + enemyAI.ts + index.ts).

**Difficulty:** Medium / Orta

---

### Task 05: Save System V2 / Kayit Sistemi V2

**What to do:**
- Add `SaveDataV2` with metadata: `{ playTime: number; saveSlot: number }`
- Serialize to v2, deserialize v1 AND v2
- `getCurrentVersion()` returns 2

**Why it matters:**
Tests backward compatibility — a critical real-world skill.

**Difficulty:** Medium / Orta

---

### Task 06: Difficulty Speed / Zorluk Hizi

**What to do:**
- Add `getDifficultySpeedMultiplier(difficulty)`: `0.5 + (difficulty / 10) * 1.5`
- Apply multiplier in `updateEnemyAI` to enemy movement

**Why it matters:**
Tests integration between settings and game logic.

| Difficulty | Multiplier |
|-----------|------------|
| 1 | 0.65x |
| 5 | 1.25x |
| 10 | 2.0x |

**Difficulty:** Medium / Orta

---

### Task 07: Event Log / Olay Gunlugu

**What to do:**
- Create new file `src/systems/eventLog.ts`
- `GameEvent` interface: `{ type, timestamp, data? }`
- `createEventLog(maxSize=50)`, `logEvent(log, event)` (FIFO trim), `getRecentEvents(log, count)`
- Add `eventLog` to `GameState` and `createInitialState`

**Why it matters:**
Tests creating entirely new files + modifying existing ones (most complex change type).

**Difficulty:** Hard / Zor

---

### Task 08: Cooldown System / Bekleme Suresi Sistemi

**What to do:**
- Create `src/systems/cooldown.ts`
- `CooldownEntry`: `{ name, durationTicks, remainingTicks }`
- Functions: create, start, tick, isOnCooldown, getCooldownRemaining

**Why it matters:**
Another new-file task with specific behavior requirements.

**Difficulty:** Medium-Hard / Orta-Zor

---

### Task 09: Deterministic RNG / Deterministik Rastgele Sayi

**What to do:**
- Create `src/systems/rng.ts`
- Implement mulberry32 PRNG: `createRNG(seed)` returns `() => number` in [0, 1)
- `randomInt(rng, min, max)`, `randomChoice(rng, array)`

**Why it matters:**
Tests algorithm implementation (mulberry32) and determinism verification.

**Mulberry32 algorithm / Mulberry32 algoritmasi:**
```typescript
function createRNG(seed: number): () => number {
  let state = seed | 0;
  return () => {
    state = (state + 0x6D2B79F5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
```

**Difficulty:** Hard / Zor

---

### Task 10: Settings Validation / Ayar Dogrulama

**What to do:**
- Add `validateSettings(settings)` to `gameState.ts`
- Validate: difficulty 1-10, volumes 0-1, showFps boolean
- Replace invalid with defaults

**Why it matters:**
Tests defensive programming patterns.

**Difficulty:** Easy-Medium / Kolay-Orta

---

## 5. The Evaluation System / Degerlendirme Sistemi

### How `run_check.mjs` Works / Nasil Calisir

```
1. Parse CLI args (--task, --workdir)
2. Resolve mini-game directory
3. Run `npm run build` (compile TS -> JS)
4. Dynamically import dist/index.js
5. Look up task-specific check function
6. Run assertions using node:assert/strict
7. Exit 0 (PASS) or non-zero (FAIL)
```

### Check Structure / Kontrol Yapisi

Each task check follows this pattern / Her gorev kontrolu bu kaliba uyar:

```javascript
task_XX() {
  // 1. Assert function exists and is exported
  assert.ok(typeof mod.functionName === "function");

  // 2. Create test state
  const state = mod.createInitialState();

  // 3. Run the function with known inputs
  const result = mod.functionName(state, ...args);

  // 4. Assert expected outputs
  assert.strictEqual(result.field, expectedValue);

  // 5. Assert immutability (original state unchanged)
  assert.strictEqual(state.field, originalValue);
}
```

### Why Checks Run Against `dist/` / Neden `dist/` Uzerinde Calisir

The checks import from the compiled `dist/index.js` rather than TypeScript source because:
1. No extra runtime dependencies (no ts-node needed)
2. Verifies that the patch produces **compilable** code
3. Tests the actual output that would run in production

Kontroller TypeScript kaynagi yerine derlenmis `dist/index.js` dosyasindan import eder cunku:
1. Ekstra runtime bagimliligi gerekmez
2. Yamanin **derlenebilir** kod urettigini dogrular
3. Uretimde calisacak gercek ciktiyi test eder

---

## 6. The Python Solution / Python Cozum Iskeleti

### Package Structure / Paket Yapisi

```
solution/
  pyproject.toml          -- Project metadata & dependencies (v2.0.0)
  src/
    ggf_case/
      __init__.py          -- Package version
      config.py            -- Pydantic settings from .env (extended for v2.0)
      cli.py               -- Typer CLI commands (extended for all phases)
      rag/
        __init__.py
        indexer.py          -- Code chunking & indexing (fixed + AST)
        retriever.py        -- Multi-strategy retrieval
        bm25.py             -- BM25 implementation (NEW in v2.0)
        hybrid.py           -- Hybrid retrieval with RRF (NEW in v2.0)
        reranker.py         -- Cross-encoder reranking (NEW in v2.0)
      llm/
        __init__.py
        openai_compat.py    -- HTTP client for LLM API
        prompts.py          -- System & user prompt templates
        structured_output.py -- Pydantic models, CoT templates (NEW in v2.0)
      patch/
        __init__.py
        diff_guard.py       -- Patch size validation
        apply_patch.py      -- git apply wrapper
      eval/
        __init__.py
        runner.py           -- Main evaluation loop
      metrics/              -- (NEW in v2.0)
        __init__.py
        retrieval_metrics.py -- precision@k, recall@k, MRR, NDCG
        patch_metrics.py    -- exact_match, hunk_match, diff scoring
      finetune/             -- (NEW in v2.0)
        __init__.py
        data_curator.py     -- Training data curation & formatting
        trainer.py          -- OpenAI fine-tuning API integration
        evaluator.py        -- Base vs fine-tuned model comparison
      analytics/            -- (NEW in v2.0)
        __init__.py
        failure_analyzer.py -- Failure classification & attribution
        experiment.py       -- A/B experiment framework
```

### Module Details / Modul Detaylari

#### `config.py` — Configuration / Konfigrasyon
Uses `pydantic-settings` to load from environment variables and `.env` file.

Key settings / Temel ayarlar:
| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | LLM endpoint |
| `OPENAI_API_KEY` | (empty) | API key |
| `OPENAI_MODEL` | `Qwen2.5-Coder-7B-Instruct` | Model name |
| `TOP_K` | 8 | Retrieval result count |
| `DIFF_MAX_LINES` | 250 | Max changed lines |
| `DIFF_MAX_FILES` | 6 | Max files in patch |

#### `rag/indexer.py` — Codebase Indexing / Kod Tabani Indeksleme

**What it does / Ne yapar:**
1. Scans all `.ts` files in the mini-game source
2. Filters out `node_modules/` and `dist/`
3. Splits each file into overlapping chunks (60 lines, 10 line overlap)
4. Extracts symbols (function names, interface names, class names)
5. Computes content hashes for change detection
6. Saves index as JSON

**Data structures / Veri yapilari:**
```python
@dataclass
class CodeChunk:
    file_path: str      # e.g., "systems/pause.ts"
    start_line: int     # e.g., 1
    end_line: int       # e.g., 30
    content: str        # The actual code
    content_hash: str   # SHA-256 prefix
    language: str       # "typescript"
    symbols: list[str]  # ["isPaused", "pauseGame", ...]
```

#### `rag/retriever.py` — Code Retrieval / Kod Geri Getirme

**Two modes / Iki mod:**

1. **Keyword retrieval (baseline)** — Always works, no extra deps
   - Scores based on term frequency, symbol matches, file path matches
   - Symbol exact match: +10 points
   - Symbol partial match: +5 points
   - Content term frequency: +1 per occurrence (capped at 5)
   - File path match: +3 points

2. **Embedding retrieval (optional)** — Requires `sentence-transformers`
   - Uses `all-MiniLM-L6-v2` by default
   - Encodes query and all chunks
   - Cosine similarity ranking
   - Falls back to keyword if not installed

**Candidates should improve / Adaylar iyilestirmeli:**
- Better chunking strategies
- Hybrid keyword + embedding
- Qdrant vector DB integration
- Query expansion
- Re-ranking

#### `llm/openai_compat.py` — LLM Client / LLM Istemcisi

Works with any OpenAI-compatible API / Herhangi bir OpenAI-uyumlu API ile calisir:
- **OpenAI** (api.openai.com)
- **vLLM** (local server)
- **Ollama** (localhost:11434/v1)
- **Together.ai**, **Groq**, etc.

Key features / Temel ozellikler:
- Timeout handling (120s)
- Clear error messages
- Health check endpoint (`/models`)

#### `llm/prompts.py` — Prompt Templates / Prompt Sablonlari

**System prompt instructs the LLM to:**
1. Generate ONLY valid unified diff format
2. Produce MINIMAL changes
3. Preserve formatting
4. Not rename files
5. Ensure TypeScript strict mode compatibility

**User prompt includes:**
- Task title and description
- Acceptance criteria (bullet list)
- Suggested files to modify
- Retrieved code context (from RAG)

**Candidates should improve / Adaylar iyilestirmeli:**
- Few-shot examples of good patches
- Chain-of-thought reasoning before diff
- File-specific instructions
- Error recovery prompts

#### `patch/diff_guard.py` — Patch Validator / Yama Dogrulayici

Parses unified diffs and enforces constraints / Unified diff'leri ayrıstirir ve kisitlamalari uygular:

| Constraint | Default | Override? |
|-----------|---------|-----------|
| Max changed lines | 250 | Yes (flag) |
| Max files touched | 6 | Yes (flag) |
| Empty patch | Rejected | No |

Also handles LLM response cleaning / Ayrica LLM yanit temizleme:
- Strips markdown code blocks (```diff ... ```)
- Finds diff-like content in mixed responses
- Falls back to raw response if nothing detected

#### `patch/apply_patch.py` — Patch Application / Yama Uygulama

Primary method: `git apply` / Ana yontem: `git apply`
Fallback: Manual line-based application / Yedek: Manuel satir bazli uygulama

Also provides `create_working_copy()` to clone the mini-game directory for each task.

#### `eval/runner.py` — Evaluation Loop / Degerlendirme Dongusu

**Per-task flow / Gorev basi akis:**
1. Create fresh working copy of mini-game
2. Install npm dependencies
3. Retrieve relevant code via RAG
4. Generate patch via LLM
5. Validate with diff guard
6. Apply patch to working copy
7. Build (`npm run build`)
8. Run acceptance check
9. Collect metrics

**Output files / Cikti dosyalari:**
- `summary.json` — Full results with per-task data
- `logs.jsonl` — One JSON line per task result
- `index.json` — The codebase index used

#### `cli.py` — CLI Interface / Komut Satiri Arayuzu

Commands / Komutlar:
| Command | Description |
|---------|-------------|
| `ggf-case index` | Index the codebase |
| `ggf-case run-eval` | Run all 10 tasks |
| `ggf-case run-task task_01` | Run a single task |
| `ggf-case check-health` | Verify LLM endpoint |

---

## 7. RAG System Explained / RAG Sistemi Aciklamasi

### What is RAG? / RAG Nedir?

**RAG = Retrieval-Augmented Generation**

Instead of feeding the ENTIRE codebase to the LLM (which may exceed context limits), we:
1. **Index** the codebase into searchable chunks
2. **Retrieve** only the most relevant chunks for each task
3. **Augment** the LLM prompt with these chunks
4. **Generate** a patch based on the relevant context

Tum kod tabanini LLM'e vermek yerine (baglam sinirlarini asabilir):
1. Kod tabanini aranabilir parcalara **indeksle**
2. Her gorev icin en alakali parcalari **getir**
3. LLM prompt'unu bu parcalarla **zenginlestir**
4. Alakali baglama dayali bir yama **uret**

### Why RAG? / Neden RAG?

- LLMs have limited context windows (4K-128K tokens)
- The mini-game is small (~500 lines), but real codebases are millions of lines
- RAG lets you scale to any codebase size
- Better retrieval = better patches

### Your Task / Gorev

The baseline provides keyword matching with fixed-window chunking.
You are expected to improve the retrieval pipeline significantly.

Baseline keyword esleme ve sabit pencere parcalama saglar.
Geri getirme hattini onemli olcude iyilestirmeniz beklenmektedir.

---

## 8. Patch Generation Pipeline / Yama Uretim Hatti

### Prompt Design / Prompt Tasarimi

You should design effective prompts that instruct the LLM to generate valid unified diffs.
A baseline prompt is provided in `llm/prompts.py` — you can improve it.

Gecerli unified diff uretmek icin LLM'e talimat veren etkili prompt'lar tasarlamaniz gerekmektedir.
`llm/prompts.py` dosyasinda bir baseline prompt saglanmistir — iyilestirebilirsiniz.

### Unified Diff Format / Unified Diff Formati

```diff
--- a/src/systems/pause.ts
+++ b/src/systems/pause.ts
@@ -25,3 +25,11 @@ export function resumeGame(state: GameState): GameState {
   return { ...state, paused: false };
 }
+
+/**
+ * Toggles the pause state.
+ */
+export function togglePause(state: GameState): GameState {
+  return { ...state, paused: !state.paused };
+}
```

Each patch has:
- `--- a/file` — original file path
- `+++ b/file` — modified file path
- `@@ -start,count +start,count @@` — hunk header
- Lines starting with `-` — removed lines
- Lines starting with `+` — added lines
- Lines starting with ` ` (space) — unchanged context lines

---

## 9. Diff Guard / Fark Korumasi

### Why Constraints? / Neden Kisitlamalar?

Without constraints, LLMs tend to:
- Rewrite entire files instead of making small changes
- Add unnecessary refactoring
- Generate patches that are hard to review
- Break unrelated code

Kisitlamalar olmadan, LLM'ler:
- Kucuk degisiklikler yerine tum dosyalari yeniden yazarlar
- Gereksiz yeniden duzenleme eklerler
- Incelenmesi zor yamalar uretirler
- Ilgisiz kodu bozarlar

### Guard Rules / Koruma Kurallari

| Rule | Default | Rationale |
|------|---------|-----------|
| Max changed lines | 250 | Prevents full-file rewrites |
| Max files touched | 6 | Prevents scope creep |
| Empty patch | Rejected | No-op patches are useless |
| Override flag | Available | For exceptional cases |

---

## 10. How to Run Everything / Her Seyi Nasil Calistirilir

### Full Setup (Step-by-step) / Tam Kurulum (Adim adim)

#### Step 1: Clone and Enter / Klonla ve Gir
```bash
cd ggf-llm-systems-case
```

#### Step 2: Node Setup / Node Kurulumu
```bash
cd ggf-mini-game
npm install          # Install TypeScript compiler
npm run build        # Compile TS to JS in dist/
npm run demo         # Verify: should print 5-tick simulation
cd ..
```

#### Step 3: Baseline Sanity / Temel Dogrulama
```bash
node eval/checks/baseline_sanity.mjs
# Should print: === Baseline Sanity: ALL PASS ===
```

#### Step 4: Python Setup / Python Kurulumu
```bash
cd solution
python -m venv .venv

# Windows:
.\.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

pip install -e .
cd ..
```

#### Step 5: Configure .env / .env Yapilandir
```bash
cp .env.example .env
# Edit .env with your actual API key and endpoint
```

#### Step 6: Optional Qdrant / Opsiyonel Qdrant
```bash
docker compose up -d
```

#### Step 7: Run Evaluation / Degerlendirmeyi Calistir
```bash
# Option A: Use the runner scripts
./eval/run_eval.sh        # Linux/macOS
.\eval\run_eval.ps1       # Windows

# Option B: Use Python CLI directly
ggf-case run-eval
ggf-case run-task task_01  # Single task
```

---

## 11. Evaluation Flow Diagram / Degerlendirme Akis Semasi

```
START
  |
  v
[Load .env configuration]
  |
  v
[Index ggf-mini-game/src/ into chunks]
  |
  v
[Initialize LLM client]
  |
  v
[For each task in tasks.json:]
  |
  +---> [Create fresh working copy of mini-game]
  |       |
  |       v
  |     [npm install in working copy]
  |       |
  |       v
  |     [RAG: Retrieve top-K relevant code chunks]
  |       |
  |       v
  |     [Build prompt: system + task + context]
  |       |
  |       v
  |     [Send to LLM -> Get raw response]
  |       |
  |       v
  |     [Extract unified diff from response]
  |       |
  |       v
  |     [Diff Guard: Check size constraints]
  |       |  FAIL -> Log error, skip task
  |       |  PASS v
  |     [Apply patch: git apply]
  |       |  FAIL -> Log error, skip task
  |       |  PASS v
  |     [npm run build (compile patched TS)]
  |       |  FAIL -> Log error, skip task
  |       |  PASS v
  |     [Run check: node run_check.mjs --task XX]
  |       |  FAIL -> Log failure
  |       |  PASS -> Log success!
  |       v
  +---> [Collect TaskResult]
  |
  v
[Generate summary.json and logs.jsonl]
  |
  v
[Print results table]
  |
  v
END
```

---

## 12. Configuration Reference / Konfigurasyon Referansi

### Environment Variables / Ortam Degiskenleri

| Variable | Type | Default | Required | Description (EN) | Aciklama (TR) |
|----------|------|---------|----------|-------------------|---------------|
| `OPENAI_BASE_URL` | string | `http://localhost:8000/v1` | No | LLM API base URL | LLM API temel URL'i |
| `OPENAI_API_KEY` | string | (empty) | **Yes** | API authentication key | API dogrulama anahtari |
| `OPENAI_MODEL` | string | `Qwen2.5-Coder-7B-Instruct` | No | Model identifier | Model tanimlayicisi |
| `TOP_K` | int | 8 | No | Number of retrieved chunks | Getirilen parca sayisi |
| `EMBEDDING_MODEL` | string | `all-MiniLM-L6-v2` | No | Sentence-transformers model | Embedding modeli |
| `DIFF_MAX_LINES` | int | 250 | No | Max changed lines per patch | Yama basina max satir |
| `DIFF_MAX_FILES` | int | 6 | No | Max files per patch | Yama basina max dosya |
| `QDRANT_URL` | string | `http://localhost:6333` | No | Qdrant vector DB URL | Qdrant vektor DB URL'i |
| `QDRANT_COLLECTION` | string | `ggf_codebase` | No | Qdrant collection name | Qdrant koleksiyon adi |
| `RETRIEVAL_STRATEGY` | string | `keyword` | No | Retrieval: keyword, bm25, embedding, hybrid | Geri getirme stratejisi |
| `CHUNK_STRATEGY` | string | `fixed` | No | Chunking: fixed, ast, hybrid | Parcalama stratejisi |
| `RERANKER_ENABLED` | bool | `false` | No | Enable cross-encoder reranking | Yeniden siralama aktif |
| `FINETUNE_MODEL` | string | (empty) | No | Fine-tuned model ID | Fine-tune model ID |
| `EXPERIMENT_RUNS` | int | 5 | No | Runs per experiment variant | Deney varyanti basina calistirma |
| `CHAIN_OF_THOUGHT` | bool | `false` | No | Enable CoT prompting | CoT prompting aktif |
| `EVAL_TIMEOUT_SECONDS` | int | 120 | No | Per-task timeout | Gorev basi zaman asimi |
| `LOG_LEVEL` | string | `INFO` | No | Python logging level | Loglama seviyesi |

---

## 13. The 4 Evaluation Phases / 4 Degerlendirme Fazi

### Phase 1: RAG Pipeline & Retrieval Quality (30 pts)
### Faz 1: RAG Hatti ve Geri Getirme Kalitesi (30 puan)

This phase tests your ability to build a production-quality retrieval system.
Bu faz uretim kalitesinde bir geri getirme sistemi olusturma becerinizi test eder.

**What you need to implement / Ne uygulamaniz gerekiyor:**

1. **BM25 Retrieval / BM25 Geri Getirme** — Implement Okapi BM25 with tokenization, inverted index, and IDF computation. / Tokenizasyon, ters indeks ve IDF hesaplama ile Okapi BM25 uygulayin. The formula is / Formul:
   ```
   score(D,Q) = sum_i(IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1*(1-b+b*|D|/avgdl)))
   ```
   Where k1=1.5 and b=0.75 are standard parameters. / k1=1.5 ve b=0.75 standart parametrelerdir.

2. **Hybrid Retrieval / Hibrit Geri Getirme** — Combine multiple retrieval strategies using Reciprocal Rank Fusion (RRF). / Coklu geri getirme stratejilerini Reciprocal Rank Fusion (RRF) ile birlestirin:
   ```
   RRF_score(d) = sum_i(1 / (k + rank_i(d)))
   ```
   Where k=60 is the standard RRF constant. / k=60 standart RRF sabitidir.

3. **AST-Aware Chunking / AST-Duyarli Parcalama** — Instead of fixed 60-line windows, chunk at function/class/interface boundaries. This preserves semantic units and improves retrieval precision. / Sabit 60 satirlik pencereler yerine fonksiyon/sinif/arayuz sinirlarinda parcalayin. Bu, anlamsal birimleri korur ve geri getirme hassasiyetini arttirir.

4. **Retrieval Metrics / Geri Getirme Metrikleri** — Measure your retrieval quality against gold labels. / Geri getirme kalitenizi altin etiketlere karsi olcun:
   - **Precision@k** — What fraction of retrieved items are relevant / Getirilen ogelerin ne kadari alakali
   - **MRR** — Mean Reciprocal Rank of the first relevant result / Ilk alakali sonucun Ortalama Ters Siralamasi
   - **NDCG@k** — Normalized Discounted Cumulative Gain / Normallesmis Iskontolu Kumlatif Kazanc

**Gold labels / Altin etiketler:** `eval/gold_labels.json` contains ground truth for each task. / Her gorev icin temel gercegi icerir.

### Phase 2: Prompt Engineering & Structured Output (20 pts)
### Faz 2: Prompt Muhendisligi ve Yapilandirilmis Cikti (20 puan)

**What you need to implement / Ne uygulamaniz gerekiyor:**

1. **Structured Output / Yapilandirilmis Cikti** — Pydantic models (`PatchResponse`, `AnalysisResponse`) for validating LLM outputs / LLM ciktilarini dogrulamak icin Pydantic modeller
2. **JSON Extraction / JSON Cikarimi** — Robust extraction from LLM responses (direct parse, code block, brace matching) / LLM yanitlarindan saglam cikarim (dogrudan ayristirma, kod blogu, suslu parantez eslestirme)
3. **Chain-of-Thought Templates / Dusunce Zinciri Sablonlari** — Step-by-step reasoning templates that guide the LLM through: / LLM'i asagidaki adimlarda yonlendiren adim adim akil yurutme sablonlari:
   - Identifying target files / Hedef dosyalari belirleme
   - Analyzing existing code / Mevcut kodu analiz etme
   - Planning changes / Degisiklikleri planlama
   - Considering edge cases / Uc durumlari dikkate alma
   - Generating the diff / Diff uretme

### Phase 3: Fine-Tuning & Training Data Curation (30 pts)
### Faz 3: Fine-Tuning ve Egitim Verisi Duzenleme (30 puan)

**What you need to implement / Ne uygulamaniz gerekiyor:**

1. **Training Data / Egitim Verisi** — `eval/training_data/examples.jsonl` contains 50 examples (5 per task). Format them for OpenAI fine-tuning. / 50 ornek icerir (gorev basina 5). OpenAI fine-tuning icin formatlarin.
2. **Data Quality / Veri Kalitesi** — Validate examples, filter by quality label (gold/bad/partial), compute statistics / Ornekleri dogrulayin, kalite etiketine gore filtreleyin, istatistikleri hesaplayin
3. **Train/Val Split / Egitim/Dogrulama Bolumu** — 80/20 stratified split maintaining task distribution / Gorev dagiliminI koruyan 80/20 katmanli bolum
4. **API Integration / API Entegrasyonu** — OpenAI fine-tuning API: upload file, create job, poll status / OpenAI fine-tuning API: dosya yukle, is olustur, durum sorgula
5. **Model Comparison / Model Karsilastirma** — Run evaluation with base model vs fine-tuned model / Temel model ile fine-tuned model karsilastirmali degerlendirme calistirin

### Phase 4: Analytics & Experiment Design (20 pts)
### Faz 4: Analitik ve Deney Tasarimi (20 puan)

**What you need to implement / Ne uygulamaniz gerekiyor:**

1. **Failure Analysis / Hata Analizi** — Classify each failed task into categories: / Her basarisiz gorevi kategorilere siniflandirin:
   - `RETRIEVAL_MISS` — Relevant code not retrieved / Alakali kod getirilmedi
   - `GENERATION_ERROR` — LLM produced invalid diff / LLM gecersiz diff uretti
   - `APPLY_FAILURE` — Patch couldn't be applied / Yama uygulanamadi
   - `BUILD_FAILURE` — Patched code doesn't compile / Yamali kod derlenmiyor
   - `CHECK_FAILURE` — Compiled but acceptance checks fail / Derlendi ama kabul kontrolleri basarisiz

2. **A/B Experiments / A/B Deneyleri** — Compare configurations (e.g., keyword vs hybrid retrieval): / Konfigurasyonlari karsilastirin (ornegin, anahtar kelime vs hibrit geri getirme):
   - Multiple runs per variant (n >= 5) / Varyant basina coklu calistirma (n >= 5)
   - Paired t-test for statistical significance / Istatistiksel anlamlilik icin eslestirilmis t-testi
   - Cohen's d for effect size / Etki boyutu icin Cohen's d
   - p < 0.05 threshold / p < 0.05 esik degeri

---

## 14. Fine-Tuning Guide / Fine-Tuning Rehberi

### Training Data / Egitim Verisi

Training examples are provided in `eval/training_data/examples.jsonl`.
You need to implement loading, validation, formatting, and splitting.
Refer to the OpenAI fine-tuning documentation for the expected format.

Egitim ornekleri `eval/training_data/examples.jsonl` dosyasinda saglanmaktadir.
Yukleme, dogrulama, formatlama ve bolme islemlerini uygulamaniz gerekmektedir.
Beklenen format icin OpenAI fine-tuning dokumantasyonuna bakin.

---

## 15. Retrieval Metrics / Geri Getirme Metrikleri

You must implement the following standard information retrieval metrics:
Asagidaki standart bilgi erisim metriklerini uygulamaniz gerekmektedir:

- **Precision@k** — Target / Hedef: >= 0.6
- **Recall@k**
- **MRR (Mean Reciprocal Rank)** — Target / Hedef: >= 0.7
- **NDCG@k (Normalized Discounted Cumulative Gain)**
- **Hit Rate**

Research these metrics and implement them in `metrics/retrieval_metrics.py`.
Bu metrikleri arastirin ve `metrics/retrieval_metrics.py` dosyasinda uygulayin.

---

## 16. Troubleshooting / Sorun Giderme

### Common Issues / Sik Karsilasilan Sorunlar

#### "npm run build fails" / "npm run build basarisiz"
```
Cause: TypeScript compilation error from bad patch
Fix: Check the patch output - it may have invalid syntax
```
Neden: Kotu yamadan TypeScript derleme hatasi
Cozum: Yama ciktisini kontrol edin - gecersiz sozdizimi olabilir

#### "togglePause must be exported as a function"
```
Cause: The patch didn't add/export the function correctly
Fix: Check if the function is in pause.ts AND exported from index.ts
```
Neden: Yama fonksiyonu dogru eklemedi/export etmedi
Cozum: Fonksiyonun pause.ts'de VE index.ts'den export edildigini kontrol edin

#### "OPENAI_API_KEY is not set"
```
Cause: .env file missing or key not set
Fix: Copy .env.example to .env and add your key
```
Neden: .env dosyasi eksik veya anahtar ayarlanmamis
Cozum: .env.example'i .env'ye kopyalayin ve anahtarinizi ekleyin

#### "git apply failed"
```
Cause: Patch format is invalid or context doesn't match
Fix: Ensure the LLM generates proper unified diff format
Fix: The manual fallback should handle simple cases
```
Neden: Yama formati gecersiz veya baglam uyusmuyor
Cozum: LLM'in duzgun unified diff formati urettiginden emin olun

#### "Build timed out"
```
Cause: npm run build takes too long
Fix: Check if node_modules is properly installed
Fix: Increase EVAL_TIMEOUT_SECONDS in .env
```

#### "Patch too large: N changed lines (max 250)"
```
Cause: LLM generated a patch that's too big
Fix: Improve prompt to request minimal changes
Fix: Use better retrieval to give focused context
Fix: Or increase DIFF_MAX_LINES (but that defeats the purpose)
```
Neden: LLM cok buyuk bir yama uretti
Cozum: Minimal degisiklik istemek icin prompt'u iyilestirin

---

## Appendix: File Tree / Ek: Dosya Agaci

```
ggf-llm-systems-case/
|-- .env.example
|-- .gitignore
|-- LICENSE
|-- README.md
|-- DETAILED_GUIDE.md          <-- This file / Bu dosya
|-- SCORING_GUIDE.md           (NEW in v2.0 - 100-point rubric)
|-- report.md
|-- docker-compose.yml
|
|-- ggf-mini-game/
|   |-- package.json
|   |-- tsconfig.json
|   |-- src/
|   |   |-- index.ts
|   |   |-- demo.ts
|   |   |-- core/
|   |   |   |-- gameState.ts
|   |   |-- systems/
|   |       |-- input.ts
|   |       |-- pause.ts
|   |       |-- score.ts
|   |       |-- enemyAI.ts
|   |       |-- save.ts
|   |-- dist/                  (compiled output)
|   |-- node_modules/          (dependencies)
|
|-- solution/
|   |-- pyproject.toml          (v2.0.0)
|   |-- src/
|       |-- ggf_case/
|           |-- __init__.py
|           |-- config.py       (extended for v2.0)
|           |-- cli.py          (extended with all phase commands)
|           |-- rag/
|           |   |-- __init__.py
|           |   |-- indexer.py   (AST-aware chunking)
|           |   |-- retriever.py (multi-strategy)
|           |   |-- bm25.py      (NEW - BM25 implementation)
|           |   |-- hybrid.py    (NEW - RRF combiner)
|           |   |-- reranker.py  (NEW - cross-encoder)
|           |-- llm/
|           |   |-- __init__.py
|           |   |-- openai_compat.py
|           |   |-- prompts.py
|           |   |-- structured_output.py  (NEW - Pydantic models, CoT)
|           |-- patch/
|           |   |-- __init__.py
|           |   |-- diff_guard.py
|           |   |-- apply_patch.py
|           |-- eval/
|           |   |-- __init__.py
|           |   |-- runner.py
|           |-- metrics/         (NEW in v2.0)
|           |   |-- __init__.py
|           |   |-- retrieval_metrics.py
|           |   |-- patch_metrics.py
|           |-- finetune/        (NEW in v2.0)
|           |   |-- __init__.py
|           |   |-- data_curator.py
|           |   |-- trainer.py
|           |   |-- evaluator.py
|           |-- analytics/       (NEW in v2.0)
|               |-- __init__.py
|               |-- failure_analyzer.py
|               |-- experiment.py
|
|-- eval/
|   |-- README.md
|   |-- tasks.json              (v2.0 with phase tags)
|   |-- gold_labels.json        (NEW - ground truth for retrieval)
|   |-- scoring_rubric.json     (NEW - 100-point rubric)
|   |-- training_data/          (NEW in v2.0)
|   |   |-- examples.jsonl       (50 training examples)
|   |   |-- hard_negatives.jsonl (30 hard negatives)
|   |-- run_eval.sh
|   |-- run_eval.ps1
|   |-- checks/
|   |   |-- run_check.mjs
|   |   |-- baseline_sanity.mjs
|   |-- phase_checks/           (NEW in v2.0)
|   |   |-- phase1_rag.mjs
|   |   |-- phase2_prompting.mjs
|   |   |-- phase3_finetune.mjs
|   |   |-- phase4_analytics.mjs
|   |-- outputs/                (gitignored)
|       |-- .gitkeep
```

---

**End of Detailed Guide / Detayli Rehber Sonu**
