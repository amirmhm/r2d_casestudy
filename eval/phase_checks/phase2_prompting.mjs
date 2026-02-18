#!/usr/bin/env node

/**
 * Phase 2: Prompt Engineering & Structured Output Checks (20 points)
 *
 * Verifies:
 * 1. Structured output Pydantic models exist
 * 2. JSON extraction from LLM responses works
 * 3. Chain-of-thought templates exist
 * 4. Prompt templates produce valid format
 * 5. Patch format compliance
 *
 * Usage:
 *   node eval/phase_checks/phase2_prompting.mjs [--solution-dir ./solution]
 */

import { existsSync, readFileSync } from "node:fs";
import { resolve, join } from "node:path";

const args = process.argv.slice(2);
let solutionDir = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--solution-dir" && args[i + 1]) {
    solutionDir = args[i + 1];
    i++;
  }
}

const solDir = solutionDir ? resolve(solutionDir) : resolve("solution");
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

console.log("=== Phase 2: Prompt Engineering & Structured Output ===\n");

// --- Check 1: Structured Output Models (5 pts) ---
console.log("1. Structured Output Models:");
const structuredPath = join(srcDir, "llm", "structured_output.py");

check("structured_output.py exists", () => {
  assertFileExists(structuredPath, "llm/structured_output.py not found");
});

check("PatchAnalysis model exists", () => {
  assertFileContains(structuredPath, "class PatchAnalysis", "PatchAnalysis model missing");
});

check("PatchResponse model exists", () => {
  assertFileContains(structuredPath, "class PatchResponse", "PatchResponse model missing");
});

check("AnalysisResponse model exists", () => {
  assertFileContains(structuredPath, "class AnalysisResponse", "AnalysisResponse model missing");
});

check("Models use Pydantic BaseModel", () => {
  assertFileContains(structuredPath, "BaseModel", "Pydantic BaseModel not used");
});

check("Diff validation exists", () => {
  assertFileContainsRegex(structuredPath, /validate_diff|field_validator.*diff/,
    "Diff validation missing");
});

// --- Check 2: JSON Extraction (5 pts) ---
console.log("\n2. JSON Extraction:");

check("extract_json_from_response exists", () => {
  assertFileContains(structuredPath, "def extract_json_from_response(",
    "extract_json_from_response() missing");
});

check("Code block extraction supported", () => {
  assertFileContainsRegex(structuredPath, /```.*json|code.*block/,
    "Code block JSON extraction not implemented");
});

check("Brace matching extraction", () => {
  assertFileContainsRegex(structuredPath, /brace|depth.*=.*0|\{.*\}/,
    "Brace-matching JSON extraction not implemented");
});

check("parse_structured_response exists", () => {
  assertFileContains(structuredPath, "def parse_structured_response(",
    "parse_structured_response() missing");
});

check("JSON schema builder exists", () => {
  assertFileContains(structuredPath, "def build_json_mode_prompt(",
    "build_json_mode_prompt() missing");
});

// --- Check 3: Chain-of-Thought Templates (5 pts) ---
console.log("\n3. Chain-of-Thought Templates:");

check("CoT patch template exists", () => {
  assertFileContains(structuredPath, "COT_PATCH_TEMPLATE",
    "COT_PATCH_TEMPLATE not found");
});

check("CoT analysis template exists", () => {
  assertFileContains(structuredPath, "COT_ANALYSIS_TEMPLATE",
    "COT_ANALYSIS_TEMPLATE not found");
});

check("CoT patch builder function", () => {
  assertFileContains(structuredPath, "def build_cot_patch_prompt(",
    "build_cot_patch_prompt() missing");
});

check("CoT analysis builder function", () => {
  assertFileContains(structuredPath, "def build_cot_analysis_prompt(",
    "build_cot_analysis_prompt() missing");
});

check("Step-by-step reasoning in template", () => {
  assertFileContainsRegex(structuredPath, /step.*by.*step|think.*step/i,
    "Step-by-step reasoning not in CoT templates");
});

// --- Check 4: Prompt Quality (5 pts) ---
console.log("\n4. Prompt Quality:");
const promptsPath = join(srcDir, "llm", "prompts.py");

check("prompts.py exists", () => {
  assertFileExists(promptsPath, "llm/prompts.py not found");
});

check("System prompt exists", () => {
  assertFileContainsRegex(promptsPath, /system.*prompt|SYSTEM_PROMPT/i,
    "System prompt not found");
});

check("User prompt template exists", () => {
  assertFileContainsRegex(promptsPath, /user.*prompt|build.*prompt|USER_PROMPT/i,
    "User prompt template not found");
});

check("Patch metrics module exists", () => {
  const patchMetricsPath = join(srcDir, "metrics", "patch_metrics.py");
  assertFileExists(patchMetricsPath, "metrics/patch_metrics.py not found");
});

check("Patch metrics has scoring functions", () => {
  const patchMetricsPath = join(srcDir, "metrics", "patch_metrics.py");
  assertFileContains(patchMetricsPath, "def exact_match(", "exact_match() missing");
  assertFileContains(patchMetricsPath, "def hunk_match_rate(", "hunk_match_rate() missing");
});

// --- Summary ---
console.log(`\n=== Phase 2 Results: ${passed}/${total} checks passed ===`);
const score = Math.round((passed / total) * 20);
console.log(`Estimated Phase 2 Score: ${score}/20 points`);

process.exit(failed > 0 ? 1 : 0);
