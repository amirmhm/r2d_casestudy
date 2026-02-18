#!/usr/bin/env node

/**
 * Phase 1: RAG Pipeline & Retrieval Quality Checks (30 points)
 *
 * Verifies:
 * 1. BM25 implementation exists and has correct structure
 * 2. Hybrid retrieval combiner exists
 * 3. Reranker module exists
 * 4. Retrieval metrics module with correct functions
 * 5. AST-aware chunking in indexer
 * 6. Multi-strategy retrieval in retriever
 *
 * Usage:
 *   node eval/phase_checks/phase1_rag.mjs [--solution-dir ./solution]
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

console.log("=== Phase 1: RAG Pipeline & Retrieval Quality ===\n");

// --- Check 1: BM25 Implementation (5 pts) ---
console.log("1. BM25 Implementation:");
const bm25Path = join(srcDir, "rag", "bm25.py");

check("bm25.py exists", () => {
  assertFileExists(bm25Path, "rag/bm25.py not found");
});

check("BM25 has tokenize function", () => {
  assertFileContains(bm25Path, "def tokenize(", "tokenize() function missing");
});

check("BM25 has build_bm25_index function", () => {
  assertFileContains(bm25Path, "def build_bm25_index(", "build_bm25_index() missing");
});

check("BM25 has scoring function", () => {
  assertFileContains(bm25Path, "def bm25_score(", "bm25_score() missing");
});

check("BM25 has retrieve function", () => {
  assertFileContains(bm25Path, "def retrieve_bm25(", "retrieve_bm25() missing");
});

check("BM25 uses IDF computation", () => {
  assertFileContainsRegex(bm25Path, /idf|inverse.*document.*freq/i, "IDF computation missing");
});

// --- Check 2: Hybrid Retrieval (5 pts) ---
console.log("\n2. Hybrid Retrieval:");
const hybridPath = join(srcDir, "rag", "hybrid.py");

check("hybrid.py exists", () => {
  assertFileExists(hybridPath, "rag/hybrid.py not found");
});

check("Hybrid has RRF function", () => {
  assertFileContains(hybridPath, "def reciprocal_rank_fusion(", "RRF function missing");
});

check("Hybrid has weighted combination", () => {
  assertFileContains(hybridPath, "def weighted_combination(", "weighted_combination() missing");
});

check("Hybrid has main entry point", () => {
  assertFileContains(hybridPath, "def hybrid_retrieve(", "hybrid_retrieve() missing");
});

// --- Check 3: Reranker Module (5 pts) ---
console.log("\n3. Reranker Module:");
const rerankerPath = join(srcDir, "rag", "reranker.py");

check("reranker.py exists", () => {
  assertFileExists(rerankerPath, "rag/reranker.py not found");
});

check("Reranker class exists", () => {
  assertFileContains(rerankerPath, "class Reranker", "Reranker class missing");
});

check("NoOp reranker exists", () => {
  assertFileContains(rerankerPath, "class NoOpReranker", "NoOpReranker class missing");
});

check("Factory function exists", () => {
  assertFileContains(rerankerPath, "def create_reranker(", "create_reranker() missing");
});

// --- Check 4: Retrieval Metrics (5 pts) ---
console.log("\n4. Retrieval Metrics:");
const metricsDir = join(srcDir, "metrics");
const retrievalMetricsPath = join(metricsDir, "retrieval_metrics.py");

check("metrics/ directory exists", () => {
  assertFileExists(metricsDir, "metrics/ directory not found");
});

check("retrieval_metrics.py exists", () => {
  assertFileExists(retrievalMetricsPath, "retrieval_metrics.py not found");
});

check("precision_at_k implemented", () => {
  assertFileContains(retrievalMetricsPath, "def precision_at_k(", "precision_at_k() missing");
});

check("recall_at_k implemented", () => {
  assertFileContains(retrievalMetricsPath, "def recall_at_k(", "recall_at_k() missing");
});

check("mrr implemented", () => {
  assertFileContains(retrievalMetricsPath, "def mrr(", "mrr() missing");
});

check("ndcg_at_k implemented", () => {
  assertFileContains(retrievalMetricsPath, "def ndcg_at_k(", "ndcg_at_k() missing");
});

check("hit_rate implemented", () => {
  assertFileContains(retrievalMetricsPath, "def hit_rate(", "hit_rate() missing");
});

// --- Check 5: AST-Aware Chunking (5 pts) ---
console.log("\n5. AST-Aware Chunking in Indexer:");
const indexerPath = join(srcDir, "rag", "indexer.py");

check("indexer.py exists", () => {
  assertFileExists(indexerPath, "rag/indexer.py not found");
});

check("AST boundary detection", () => {
  assertFileContainsRegex(indexerPath, /ast.*boundar|_find_ast_boundaries|function.*class.*interface/i,
    "AST boundary detection missing from indexer");
});

check("Multiple chunk strategies supported", () => {
  assertFileContainsRegex(indexerPath, /strategy.*=.*["']?(fixed|ast|hybrid)/,
    "Chunk strategy parameter missing");
});

check("Imports extraction", () => {
  assertFileContains(indexerPath, "extract_imports", "extract_imports function missing");
});

// --- Check 6: Multi-Strategy Retrieval (5 pts) ---
console.log("\n6. Multi-Strategy Retrieval:");
const retrieverPath = join(srcDir, "rag", "retriever.py");

check("retriever.py exists", () => {
  assertFileExists(retrieverPath, "rag/retriever.py not found");
});

check("BM25 strategy supported", () => {
  assertFileContainsRegex(retrieverPath, /["']bm25["']|strategy.*bm25/,
    "BM25 retrieval strategy not supported");
});

check("Hybrid strategy supported", () => {
  assertFileContainsRegex(retrieverPath, /["']hybrid["']|strategy.*hybrid/,
    "Hybrid retrieval strategy not supported");
});

check("Embedding strategy supported", () => {
  assertFileContainsRegex(retrieverPath, /["']embedding["']|strategy.*embedding/,
    "Embedding retrieval strategy not supported");
});

// --- Summary ---
console.log(`\n=== Phase 1 Results: ${passed}/${total} checks passed ===`);
const score = Math.round((passed / total) * 30);
console.log(`Estimated Phase 1 Score: ${score}/30 points`);

process.exit(failed > 0 ? 1 : 0);
