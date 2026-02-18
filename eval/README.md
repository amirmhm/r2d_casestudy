# Evaluation System / Degerlendirme Sistemi

## Overview / Genel Bakis

This directory contains the evaluation suite for the GGF LLM Systems Case.

Bu dizin, GGF LLM Sistemleri Case'inin degerlendirme paketini icerir.

## Structure / Yapi

```
eval/
  tasks.json         - 10 evaluation tasks / 10 degerlendirme gorevi
  checks/
    run_check.mjs    - Generic check runner / Genel kontrol calistiricisi
    baseline_sanity.mjs - Baseline verification / Temel dogrulama
  outputs/           - Eval results (gitignored) / Sonuclar (git-ignore)
  run_eval.sh        - Linux/macOS runner
  run_eval.ps1       - Windows PowerShell runner
```

## How Checks Work / Kontroller Nasil Calisir

1. The check runner builds the mini-game (`npm run build`)
2. It imports the compiled `dist/index.js`
3. For each task, it runs specific assertions using `node:assert`
4. Exit code 0 = PASS, non-zero = FAIL

---

1. Kontrol calistiricisi mini-oyunu derler (`npm run build`)
2. Derlenmis `dist/index.js` dosyasini import eder
3. Her gorev icin `node:assert` ile belirli dogrulamalar yapar
4. Cikis kodu 0 = BASARILI, sifir olmayan = BASARISIZ

## Running Individual Checks / Tek Kontrol Calistirma

```bash
# Run a specific task check / Belirli bir gorev kontrolu calistir
node eval/checks/run_check.mjs --task task_01

# Run baseline sanity / Temel dogrulama calistir
node eval/checks/baseline_sanity.mjs

# Run with custom working directory / Ozel calisma dizini ile calistir
node eval/checks/run_check.mjs --task task_01 --workdir ./path/to/copy
```
