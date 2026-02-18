# GGF LLM Systems Case v2.0

> A comprehensive take-home engineering case for evaluating **RAG expertise**, **model fine-tuning ability**, **prompt engineering**, and **analytical thinking skills**.
> **RAG uzmanligi**, **model fine-tuning becerisi**, **prompt muhendisligi** ve **analitik dusunme yeteneklerini** degerlendirmek icin kapsamli bir teknik degerlendirme case'i.

---

## What This Case Tests / Bu Case Ne Test Eder

| Phase | Area | Weight | Description |
|-------|------|--------|-------------|
| **Phase 1** | RAG Pipeline & Retrieval Quality | 30% | BM25, hybrid retrieval, AST-aware chunking, retrieval metrics |
| **Phase 2** | Prompt Engineering & Structured Output | 20% | CoT prompting, structured LLM output, patch format compliance |
| **Phase 3** | Fine-Tuning & Training Data Curation | 30% | Training data format, OpenAI API integration, model comparison |
| **Phase 4** | Analytics, Experiment Design & Failure Analysis | 20% | Failure attribution, A/B experiments, statistical significance |

---

## Time Expectation / Sure Beklentisi

**12-16 hours** — You are expected to implement all four phases and achieve the highest score possible on the 100-point rubric.

**12-16 saat** — Dort fazin tumunu uygulamaniz ve 100 puanlik rubrik uzerinden en yuksek puani almaniz beklenir.

---

## Scoring / Puanlama

This case uses a **100-point rubric** across 4 phases. See `SCORING_GUIDE.md` for the detailed breakdown.
Bu case 4 faz boyunca **100 puanlik rubrik** kullanir. Detayli dokulm icin `SCORING_GUIDE.md`'ye bakin.

| Phase | Points | Key Criteria |
|-------|--------|--------------|
| Phase 1: RAG | 30 | BM25, embeddings, hybrid retrieval, AST chunking, precision@5, MRR |
| Phase 2: Prompting | 20 | Structured output, CoT templates, format compliance, effectiveness |
| Phase 3: Fine-Tuning | 30 | Data format, train/val split, API integration, comparison, improvement |
| Phase 4: Analytics | 20 | Failure attribution, root cause analysis, A/B experiments, reporting |

---

## Repository Structure / Depo Yapisi

```
ggf-llm-systems-case/
  README.md                <- You are here / Buradasiniz
  DETAILED_GUIDE.md        <- Comprehensive explanation of all components
  SCORING_GUIDE.md         <- 100-point rubric detailed explanation
  .env.example             <- Environment variables template
  docker-compose.yml       <- Optional Qdrant for vector RAG
  report.md                <- Fill this after evaluation
  LICENSE                  <- MIT License

  ggf-mini-game/           <- Target codebase (TypeScript game systems)
    src/
      core/gameState.ts    <- State shape, reducer, initial state
      systems/
        input.ts           <- Input mapping
        pause.ts           <- Pause management
        score.ts           <- Score system
        enemyAI.ts         <- Enemy AI behaviors
        save.ts            <- Save/load serialization
      index.ts             <- Public API exports
      demo.ts              <- 5-tick simulation demo
    package.json
    tsconfig.json

  solution/                <- Your solution (Python 3.11)
    src/ggf_case/
      config.py            <- Pydantic settings (.env)
      cli.py               <- Typer CLI (all commands)
      rag/
        indexer.py          <- Codebase indexer (fixed + AST chunking)
        retriever.py        <- Multi-strategy retrieval
        bm25.py             <- BM25 implementation
        hybrid.py           <- Hybrid retrieval (RRF)
        reranker.py         <- Cross-encoder reranking
      llm/
        openai_compat.py    <- OpenAI-compatible LLM client
        prompts.py          <- System & patch prompt templates
        structured_output.py <- Pydantic models, CoT templates
      patch/
        diff_guard.py       <- Patch size validator
        apply_patch.py      <- git apply wrapper
      eval/
        runner.py           <- Main evaluation loop
      metrics/
        retrieval_metrics.py <- precision@k, recall@k, MRR, NDCG
        patch_metrics.py    <- exact_match, hunk_match, diff scoring
      finetune/
        data_curator.py     <- Training data curation & formatting
        trainer.py          <- OpenAI fine-tuning API integration
        evaluator.py        <- Base vs fine-tuned model comparison
      analytics/
        failure_analyzer.py <- Failure classification & attribution
        experiment.py       <- A/B experiment framework
    pyproject.toml

  eval/
    tasks.json             <- 10 evaluation tasks (with phase tags)
    gold_labels.json       <- Ground truth for retrieval evaluation
    scoring_rubric.json    <- Machine-readable 100-point rubric
    training_data/
      examples.jsonl       <- 50 training examples (5 per task)
      hard_negatives.jsonl <- 30 hard-negative retrieval examples
    checks/
      run_check.mjs        <- Generic assertion runner
      baseline_sanity.mjs  <- Baseline verification
    phase_checks/
      phase1_rag.mjs       <- RAG pipeline checks
      phase2_prompting.mjs <- Prompting checks
      phase3_finetune.mjs  <- Fine-tuning checks
      phase4_analytics.mjs <- Analytics checks
    outputs/               <- Results go here (gitignored)
    run_eval.sh            <- Linux/macOS runner
    run_eval.ps1           <- Windows runner

  report.md                <- Fill this after evaluation
```

---

## Setup / Kurulum

### Prerequisites / On Kosullar

- **Node.js** >= 18.0 (`node --version`)
- **Python** >= 3.11 (`python --version`)
- **Git** (for patch application)
- **Docker** (optional, for Qdrant vector DB)

### Step 1: Node Setup / Adim 1: Node Kurulumu

```bash
cd ggf-mini-game
npm install
npm run build        # Verify baseline compiles / Temel derlemeyi dogrulayin
npm run demo         # Run 5-tick simulation / 5 tiklik simulasyonu calistirin
```

### Step 2: Python Setup / Adim 2: Python Kurulumu

```bash
cd solution

# Create virtual environment / Sanal ortam olusturun
python -m venv .venv

# Activate (choose your OS) / Aktive edin (isletim sisteminizi secin)
source .venv/bin/activate        # Linux/macOS
.\.venv\Scripts\Activate.ps1     # Windows PowerShell

# Install core / Cekirdek paketleri yukleyin
pip install -e .

# With embeddings (for Phase 1): / Embedding ile (Faz 1 icin):
pip install -e ".[embeddings]"

# With analytics (for Phase 4): / Analitik ile (Faz 4 icin):
pip install -e ".[analytics]"

# Install everything: / Her seyi yukleyin:
pip install -e ".[all]"
```

### Step 3: Configuration / Adim 3: Yapilandirma

```bash
# Copy and edit .env / .env dosyasini kopyalayin ve duzenleyin
cp .env.example .env
# Edit .env with your LLM API key and endpoint / .env'yi LLM API anahtariniz ve endpoint'iniz ile duzenleyin
```

### Step 4: Optional Qdrant / Adim 4: Opsiyonel Qdrant

```bash
docker compose up -d
```

---

## Running / Calistirma

### Quick Start

```bash
# Linux/macOS
chmod +x eval/run_eval.sh
./eval/run_eval.sh

# Windows PowerShell
.\eval\run_eval.ps1
```

### CLI Commands / CLI Komutlari

```bash
# Baseline verification
node eval/checks/baseline_sanity.mjs

# Phase checks (verify your implementation)
node eval/phase_checks/phase1_rag.mjs
node eval/phase_checks/phase2_prompting.mjs
node eval/phase_checks/phase3_finetune.mjs
node eval/phase_checks/phase4_analytics.mjs

# Core evaluation
ggf-case index                          # Index the codebase
ggf-case run-eval                       # Run all 10 tasks
ggf-case run-task task_01               # Run a single task
ggf-case check-health                   # Verify LLM endpoint

# Phase 1: RAG metrics
ggf-case metrics                        # Compute retrieval metrics vs gold labels

# Phase 3: Fine-tuning
ggf-case finetune prepare               # Prepare training data
ggf-case finetune run <file>            # Start fine-tuning job
ggf-case finetune eval --job-id <id>    # Check job status

# Phase 4: Analytics
ggf-case analyze <results_path>         # Run failure analysis
ggf-case report <results_dir>           # Generate full report
```

---

## The 10 Tasks / 10 Gorev

| # | Task | Phase | Difficulty | Description |
|---|------|-------|------------|-------------|
| 1 | Pause Toggle | Phase 2 | Easy | Add `togglePause()` function |
| 2 | Input Remapping | Phase 2 | Easy | Add `remapKey()` for input bindings |
| 3 | Score Combo | Phase 2 | Medium | Add `addComboScore()` with streak multiplier |
| 4 | Enemy Patrol | Phase 1 | Medium | Add patrol mode + chase threshold to AI |
| 5 | Save V2 | Phase 1 | Medium | Versioned save with backward compatibility |
| 6 | Difficulty Speed | Phase 3 | Medium | Difficulty-based enemy speed multiplier |
| 7 | Event Log | Phase 3 | Hard | New event logging system with FIFO buffer |
| 8 | Cooldown | Phase 3 | Medium | Ability cooldown mechanism |
| 9 | Deterministic RNG | Phase 4 | Hard | Seeded PRNG (mulberry32) |
| 10 | Settings Validation | Phase 4 | Medium | Input validation with safe defaults |

---

## The 4 Phases / 4 Faz

### Phase 1: RAG Pipeline (30 pts) / Faz 1: RAG Hatti (30 puan)

Implement a production-quality retrieval system:
Uretim kalitesinde bir geri getirme sistemi uygulayin:

- **BM25 retrieval / BM25 geri getirme** — Term frequency with IDF weighting / IDF agirliklama ile terim frekans
- **Embedding retrieval / Embedding geri getirme** — Semantic search with sentence-transformers / sentence-transformers ile anlamsal arama
- **Hybrid retrieval / Hibrit geri getirme** — Combine BM25 + embeddings with Reciprocal Rank Fusion / BM25 + embedding'i RRF ile birlestirme
- **AST-aware chunking / AST-duyarli parcalama** — Chunk by function/class boundaries, not fixed windows / Sabit pencere degil fonksiyon/sinif sinirlarinda parcalama
- **Retrieval metrics / Geri getirme metrikleri** — Achieve precision@5 >= 0.6 and MRR >= 0.7 on gold labels / Altin etiketlerde precision@5 >= 0.6 ve MRR >= 0.7 elde edin

### Phase 2: Prompt Engineering (20 pts) / Faz 2: Prompt Muhendisligi (20 puan)

Design effective prompts and structured output handling:
Etkili promptlar ve yapilandirilmis cikti yonetimi tasarlayin:

- **Structured output models / Yapilandirilmis cikti modelleri** — Pydantic models for LLM responses / LLM yanitlari icin Pydantic modeller
- **Chain-of-thought templates / Dusunce zinciri sablonlari** — Step-by-step reasoning before patch generation / Yama uretiminden once adim adim akil yurutme
- **JSON extraction / JSON cikarimi** — Robust extraction from LLM responses / LLM yanitlarindan saglam cikarim
- **Patch format compliance / Yama format uyumlulugu** — Generate valid unified diffs for all tasks / Tum gorevler icin gecerli unified diff uretme

### Phase 3: Fine-Tuning (30 pts) / Faz 3: Fine-Tuning (30 puan)

Build a fine-tuning pipeline:
Bir fine-tuning hatti olusturun:

- **Training data curation / Egitim verisi duzenleme** — Load, validate, and format 50 examples / 50 ornegi yukle, dogrula ve formatla
- **Train/val split / Egitim/dogrulama bolumu** — 80/20 stratified split / 80/20 katmanli bolum
- **OpenAI API integration / OpenAI API entegrasyonu** — Upload data, create jobs, monitor status / Veri yukle, is olustur, durum izle
- **Model comparison / Model karsilastirma** — Evaluate base vs fine-tuned model / Temel model ile fine-tuned model karsilastirmasi
- **Document hyperparameters / Hiperparametre dokumantasyonu** — Explain choices for lr, epochs, batch size / lr, epoch, batch boyutu secimlerini aciklayin

### Phase 4: Analytics (20 pts) / Faz 4: Analitik (20 puan)

Analyze and improve system performance:
Sistem performansini analiz edin ve iyilestirin:

- **Failure attribution / Hata atfi** — Classify failures into categories / Hatalari kategorilere siniflandirin
- **Root cause analysis / Kok neden analizi** — Identify patterns and suggest fixes / Oruntuleri belirleyin ve duzeltmeler onerin
- **A/B experiments / A/B deneyleri** — Compare configurations with statistical significance / Konfigurasyonlari istatistiksel anlamlilikla karsilastirin
- **Comprehensive reporting / Kapsamli raporlama** — Generate full evaluation reports / Tam degerlendirme raporlari uretin

---

## Rules / Kurallar

1. **Patch/diff mode only / Sadece yama/diff modu** — Do NOT rewrite entire files. Generate minimal unified diffs. / Tum dosyalari yeniden YAZMAYIN. Minimal unified diff uretin.
2. **Diff guard / Fark korumasi** — Patches exceeding 250 changed lines or 6 files will be rejected. / 250 degisen satiri veya 6 dosyayi asan yamalar reddedilir.
3. **No disabling checks / Kontrolleri devre disi birakma** — Do not modify the eval checks or tasks.json. / Eval kontrollerini veya tasks.json'i degistirmeyin.
4. **No manual edits / Manuel duzenleme yok** — Your solution must be fully automated (LLM generates all patches). / Cozumunuz tamamen otomatik olmalidir (LLM tum yamalari uretir).
5. **Reproducible / Tekrarlanabilir** — Running `run_eval.sh` / `run_eval.ps1` must work end-to-end. / `run_eval.sh` / `run_eval.ps1` calistirmak uctan uca calismalidir.

---

## What to Submit / Ne Teslim Edilmeli

1. **Your forked repository / Fork'lanmis deponuz** — with all improvements to the `solution/` directory / `solution/` dizinindeki tum iyilestirmelerle birlikte
2. **Filled `report.md` / Doldurulmus `report.md`** — documenting approach, results, metrics, and analysis / yaklasim, sonuclar, metrikler ve analizi belgeleyin
3. **A short demo / Kisa bir demo** — screen recording or output logs showing eval results / degerlendirme sonuclarini gosteren ekran kaydi veya cikti loglari
4. **Phase check results / Faz kontrol sonuclari** — output of all 4 phase check scripts / 4 faz kontrol betigi ciktisi

---

## Evaluation Criteria / Degerlendirme Kriterleri

See `SCORING_GUIDE.md` for the full 100-point rubric. Summary:
Tam 100 puanlik rubrik icin `SCORING_GUIDE.md`'ye bakin. Ozet:

| Phase | Points | Weight |
|-------|--------|--------|
| Phase 1: RAG Pipeline | 30 | 30% |
| Phase 2: Prompt Engineering | 20 | 20% |
| Phase 3: Fine-Tuning | 30 | 30% |
| Phase 4: Analytics | 20 | 20% |
| **Total** | **100** | **100%** |

---

## Tips / Ipuclari

- Start with the baseline keyword retrieval — it works for simple tasks / Temel anahtar kelime geri getirme ile baslayin — basit gorevler icin calisir
- Implement BM25 first, then hybrid for better retrieval scores / Once BM25'i uygulayin, daha iyi geri getirme puanlari icin hibrit yapin
- Use `ggf-case metrics` to measure retrieval quality against gold labels / Altin etiketlere karsi geri getirme kalitesini olcmek icin `ggf-case metrics` kullanin
- Run `node eval/phase_checks/phase1_rag.mjs` to verify your RAG implementation / RAG uygulamanizi dogrulamak icin `node eval/phase_checks/phase1_rag.mjs` calistirin
- The training data in `eval/training_data/examples.jsonl` has both good and bad examples / `eval/training_data/examples.jsonl`'deki egitim verisi hem iyi hem kotu ornekler icerir
- Use the hard negatives to test your retrieval precision / Geri getirme hassasiyetinizi test etmek icin zor negatifleri kullanin
- The diff guard is your friend — it prevents the LLM from overwriting files / Diff korumasi arkadasinizdir — LLM'in dosyalarin uzerine yazmasini engeller
- You can run individual tasks to iterate: `ggf-case run-task task_01` / Tek tek gorev calistirarak ilerleyebilirsiniz: `ggf-case run-task task_01`

---

## License

MIT — See [LICENSE](LICENSE)
