#!/usr/bin/env node

/**
 * Phase 3: Fine-Tuning & Training Data Curation Checks (30 points)
 *
 * Verifies:
 * 1. Training data is valid JSONL with correct schema
 * 2. Data curator module exists with proper functions
 * 3. Trainer module with OpenAI API integration
 * 4. Evaluator module for model comparison
 * 5. Train/val split implementation
 * 6. Hyperparameter documentation
 *
 * Usage:
 *   node eval/phase_checks/phase3_finetune.mjs [--solution-dir ./solution]
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";

const args = process.argv.slice(2);
let solutionDir = null;
let repoRoot = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--solution-dir" && args[i + 1]) {
    solutionDir = args[i + 1];
    i++;
  } else if (args[i] === "--repo-root" && args[i + 1]) {
    repoRoot = args[i + 1];
    i++;
  }
}

const solDir = solutionDir ? resolve(solutionDir) : resolve("solution");
const rootDir = repoRoot ? resolve(repoRoot) : resolve(".");
const srcDir = join(solDir, "src", "ggf_case");

let passed = 0;
let failed = 0;
let total = 0;

function check(name, fn) {
  total++;
  try {
    fn();
    console.log(`  [PASS] ${name}`);
    passed++;
  } catch (e) {
    console.log(`  [FAIL] ${name}: ${e.message}`);
    failed++;
  }
}

function assertFileExists(path, msg) {
  if (!existsSync(path)) {
    throw new Error(msg || `File not found: ${path}`);
  }
}

function assertFileContains(path, pattern, msg) {
  const content = readFileSync(path, "utf-8");
  if (!content.includes(pattern)) {
    throw new Error(msg || `Pattern "${pattern}" not found in ${path}`);
  }
}

function assertFileContainsRegex(path, regex, msg) {
  const content = readFileSync(path, "utf-8");
  if (!regex.test(content)) {
    throw new Error(msg || `Regex ${regex} not found in ${path}`);
  }
}

console.log("=== Phase 3: Fine-Tuning & Training Data Curation ===\n");

// --- Check 1: Training Data Format (5 pts) ---
console.log("1. Training Data Format:");
const examplesPath = join(rootDir, "eval", "training_data", "examples.jsonl");
const hardNegsPath = join(rootDir, "eval", "training_data", "hard_negatives.jsonl");

check("examples.jsonl exists", () => {
  assertFileExists(examplesPath, "eval/training_data/examples.jsonl not found");
});

check("examples.jsonl has valid JSONL format", () => {
  const content = readFileSync(examplesPath, "utf-8").trim();
  const lines = content.split("\n").filter((l) => l.trim());
  if (lines.length < 10) {
    throw new Error(`Expected at least 10 examples, got ${lines.length}`);
  }
  // Validate each line is valid JSON
  for (let i = 0; i < Math.min(5, lines.length); i++) {
    try {
      JSON.parse(lines[i]);
    } catch {
      throw new Error(`Line ${i + 1} is not valid JSON`);
    }
  }
});

check("examples.jsonl has correct schema", () => {
  const content = readFileSync(examplesPath, "utf-8").trim();
  const lines = content.split("\n").filter((l) => l.trim());
  const first = JSON.parse(lines[0]);
  if (!first.task_id) throw new Error("Missing task_id field");
  if (!first.input_prompt) throw new Error("Missing input_prompt field");
  if (!first.expected_output) throw new Error("Missing expected_output field");
  if (!first.metadata) throw new Error("Missing metadata field");
});

check("examples cover multiple tasks", () => {
  const content = readFileSync(examplesPath, "utf-8").trim();
  const lines = content.split("\n").filter((l) => l.trim());
  const taskIds = new Set();
  for (const line of lines) {
    const obj = JSON.parse(line);
    taskIds.add(obj.task_id);
  }
  if (taskIds.size < 5) {
    throw new Error(`Expected examples for at least 5 tasks, got ${taskIds.size}`);
  }
});

check("examples include quality labels", () => {
  const content = readFileSync(examplesPath, "utf-8").trim();
  const lines = content.split("\n").filter((l) => l.trim());
  const qualities = new Set();
  for (const line of lines) {
    const obj = JSON.parse(line);
    if (obj.metadata && obj.metadata.quality) {
      qualities.add(obj.metadata.quality);
    }
  }
  if (qualities.size < 2) {
    throw new Error("Expected multiple quality labels (gold, bad, partial)");
  }
});

check("hard_negatives.jsonl exists", () => {
  assertFileExists(hardNegsPath, "eval/training_data/hard_negatives.jsonl not found");
});

check("hard_negatives has valid format", () => {
  const content = readFileSync(hardNegsPath, "utf-8").trim();
  const lines = content.split("\n").filter((l) => l.trim());
  if (lines.length < 10) {
    throw new Error(`Expected at least 10 hard negatives, got ${lines.length}`);
  }
  const first = JSON.parse(lines[0]);
  if (!first.task_id) throw new Error("Missing task_id");
  if (!first.negative_file) throw new Error("Missing negative_file");
  if (!first.reason) throw new Error("Missing reason");
});

// --- Check 2: Data Curator Module (5 pts) ---
console.log("\n2. Data Curator Module:");
const curatorPath = join(srcDir, "finetune", "data_curator.py");

check("data_curator.py exists", () => {
  assertFileExists(curatorPath, "finetune/data_curator.py not found");
});

check("DataCurator class exists", () => {
  assertFileContains(curatorPath, "class DataCurator", "DataCurator class missing");
});

check("load_examples method exists", () => {
  assertFileContains(curatorPath, "def load_examples(", "load_examples() missing");
});

check("validate_examples method exists", () => {
  assertFileContains(curatorPath, "def validate_examples(", "validate_examples() missing");
});

check("format_for_openai method exists", () => {
  assertFileContains(curatorPath, "def format_for_openai(", "format_for_openai() missing");
});

// --- Check 3: Train/Val Split (3 pts) ---
console.log("\n3. Train/Val Split:");

check("split_train_val method exists", () => {
  assertFileContains(curatorPath, "def split_train_val(", "split_train_val() missing");
});

check("Stratification support", () => {
  assertFileContainsRegex(curatorPath, /stratif|by_task/i,
    "Stratification not supported in split");
});

check("TrainValSplit dataclass exists", () => {
  assertFileContainsRegex(curatorPath, /TrainValSplit|train_size|val_size/,
    "TrainValSplit dataclass missing");
});

// --- Check 4: Trainer Module (7 pts) ---
console.log("\n4. Fine-Tune Trainer:");
const trainerPath = join(srcDir, "finetune", "trainer.py");

check("trainer.py exists", () => {
  assertFileExists(trainerPath, "finetune/trainer.py not found");
});

check("FineTuneTrainer class exists", () => {
  assertFileContains(trainerPath, "class FineTuneTrainer", "FineTuneTrainer class missing");
});

check("FineTuneConfig dataclass exists", () => {
  assertFileContains(trainerPath, "FineTuneConfig", "FineTuneConfig missing");
});

check("upload_training_file method exists", () => {
  assertFileContains(trainerPath, "def upload_training_file(",
    "upload_training_file() missing");
});

check("create_job method exists", () => {
  assertFileContains(trainerPath, "def create_job(", "create_job() missing");
});

check("get_job_status method exists", () => {
  assertFileContains(trainerPath, "def get_job_status(", "get_job_status() missing");
});

check("OpenAI API integration", () => {
  assertFileContainsRegex(trainerPath, /fine.?tun|\/v1\/fine_tuning/i,
    "OpenAI fine-tuning API integration missing");
});

// --- Check 5: Model Evaluator (5 pts) ---
console.log("\n5. Model Evaluator:");
const evaluatorPath = join(srcDir, "finetune", "evaluator.py");

check("evaluator.py exists", () => {
  assertFileExists(evaluatorPath, "finetune/evaluator.py not found");
});

check("ModelEvaluator class exists", () => {
  assertFileContains(evaluatorPath, "class ModelEvaluator", "ModelEvaluator class missing");
});

check("Comparison report generation", () => {
  assertFileContainsRegex(evaluatorPath, /generate_comparison|ComparisonReport/,
    "Comparison report generation missing");
});

check("Per-task breakdown", () => {
  assertFileContainsRegex(evaluatorPath, /per.*task|task.*result/i,
    "Per-task breakdown missing");
});

// --- Check 6: Hyperparameter Documentation (5 pts) ---
console.log("\n6. Hyperparameter Documentation:");

check("Trainer has hyperparameter settings", () => {
  assertFileContainsRegex(trainerPath, /n_epochs|batch_size|learning_rate/,
    "Hyperparameter fields missing from trainer config");
});

check("Hyperparameter documentation/comments", () => {
  assertFileContainsRegex(trainerPath, /epoch|batch|learning.*rate.*multiplier/i,
    "Hyperparameter documentation missing");
});

check("CLI finetune commands exist", () => {
  const cliPath = join(srcDir, "cli.py");
  assertFileContains(cliPath, "finetune", "Finetune CLI commands missing");
});

// --- Summary ---
console.log(`\n=== Phase 3 Results: ${passed}/${total} checks passed ===`);
const score = Math.round((passed / total) * 30);
console.log(`Estimated Phase 3 Score: ${score}/30 points`);

process.exit(failed > 0 ? 1 : 0);
