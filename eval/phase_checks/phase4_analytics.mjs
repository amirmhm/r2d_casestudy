#!/usr/bin/env node

/**
 * Phase 4: Analytics, Experiment Design & Failure Analysis Checks (20 points)
 *
 * Verifies:
 * 1. Failure analyzer with classification and attribution
 * 2. Root cause analysis quality
 * 3. A/B experiment framework with statistical significance
 * 4. Report generation completeness
 *
 * Usage:
 *   node eval/phase_checks/phase4_analytics.mjs [--solution-dir ./solution]
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

console.log("=== Phase 4: Analytics, Experiment Design & Failure Analysis ===\n");

// --- Check 1: Failure Analyzer (5 pts) ---
console.log("1. Failure Attribution:");
const analyzerPath = join(srcDir, "analytics", "failure_analyzer.py");

check("failure_analyzer.py exists", () => {
  assertFileExists(analyzerPath, "analytics/failure_analyzer.py not found");
});

check("FailureAnalyzer class exists", () => {
  assertFileContains(analyzerPath, "class FailureAnalyzer", "FailureAnalyzer class missing");
});

check("Failure classification function", () => {
  assertFileContains(analyzerPath, "def classify_failure(", "classify_failure() missing");
});

check("Multiple failure categories defined", () => {
  // Check for at least 4 different failure categories
  const content = readFileSync(analyzerPath, "utf-8");
  const categories = ["RETRIEVAL_MISS", "GENERATION_ERROR", "APPLY_FAILURE", "CHECK_FAILURE"];
  let found = 0;
  for (const cat of categories) {
    if (content.includes(cat)) found++;
  }
  if (found < 4) {
    throw new Error(`Only ${found}/4 required failure categories found`);
  }
});

check("analyze_results method", () => {
  assertFileContains(analyzerPath, "def analyze_results(", "analyze_results() missing");
});

// --- Check 2: Root Cause Analysis (5 pts) ---
console.log("\n2. Root Cause Analysis:");

check("Pattern identification", () => {
  assertFileContainsRegex(analyzerPath, /pattern|_identify_patterns/i,
    "Pattern identification missing from analyzer");
});

check("Recommendation generation", () => {
  assertFileContainsRegex(analyzerPath, /recommend|_generate_recommendations/i,
    "Recommendation generation missing");
});

check("Report export function", () => {
  assertFileContains(analyzerPath, "def export_report(", "export_report() missing");
});

check("Print report function", () => {
  assertFileContains(analyzerPath, "def print_report(", "print_report() missing");
});

check("Correlation analysis", () => {
  assertFileContainsRegex(analyzerPath, /correlat|retrieval.*quality.*success/i,
    "Correlation analysis missing");
});

// --- Check 3: A/B Experiment Framework (5 pts) ---
console.log("\n3. A/B Experiment Framework:");
const experimentPath = join(srcDir, "analytics", "experiment.py");

check("experiment.py exists", () => {
  assertFileExists(experimentPath, "analytics/experiment.py not found");
});

check("ExperimentRunner class exists", () => {
  assertFileContains(experimentPath, "class ExperimentRunner", "ExperimentRunner class missing");
});

check("ExperimentConfig dataclass exists", () => {
  assertFileContains(experimentPath, "ExperimentConfig", "ExperimentConfig missing");
});

check("Statistical significance test (t-test)", () => {
  assertFileContainsRegex(experimentPath, /t.?test|paired.*t|t_stat/i,
    "Statistical significance testing (t-test) missing");
});

check("Effect size computation (Cohen's d)", () => {
  assertFileContainsRegex(experimentPath, /cohen|effect.*size|cohens_d/i,
    "Effect size computation missing");
});

check("P-value computation", () => {
  assertFileContains(experimentPath, "p_value", "P-value computation missing");
});

check("StatisticalResult dataclass", () => {
  assertFileContains(experimentPath, "StatisticalResult", "StatisticalResult missing");
});

// --- Check 4: Report Completeness (5 pts) ---
console.log("\n4. Report Generation:");

check("ExperimentReport dataclass", () => {
  assertFileContains(experimentPath, "ExperimentReport", "ExperimentReport missing");
});

check("Report generation function", () => {
  assertFileContains(experimentPath, "def generate_report(", "generate_report() missing");
});

check("Report export to JSON", () => {
  assertFileContains(experimentPath, "def export_report(", "export_report() missing");
});

check("CLI analyze command exists", () => {
  const cliPath = join(srcDir, "cli.py");
  assertFileContains(cliPath, "analyze", "CLI analyze command missing");
});

check("CLI report command exists", () => {
  const cliPath = join(srcDir, "cli.py");
  assertFileContains(cliPath, "report", "CLI report command missing");
});

check("Scoring rubric exists", () => {
  const rubricPath = join(rootDir, "eval", "scoring_rubric.json");
  assertFileExists(rubricPath, "eval/scoring_rubric.json not found");
  const content = readFileSync(rubricPath, "utf-8");
  const rubric = JSON.parse(content);
  if (!rubric.total_points || rubric.total_points !== 100) {
    throw new Error("Rubric should have total_points: 100");
  }
});

check("Gold labels exist", () => {
  const goldPath = join(rootDir, "eval", "gold_labels.json");
  assertFileExists(goldPath, "eval/gold_labels.json not found");
  const content = readFileSync(goldPath, "utf-8");
  const gold = JSON.parse(content);
  if (!gold.tasks) {
    throw new Error("Gold labels should have a 'tasks' object");
  }
});

// --- Summary ---
console.log(`\n=== Phase 4 Results: ${passed}/${total} checks passed ===`);
const score = Math.round((passed / total) * 20);
console.log(`Estimated Phase 4 Score: ${score}/20 points`);

process.exit(failed > 0 ? 1 : 0);
