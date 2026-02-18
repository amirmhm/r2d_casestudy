# GGF LLM Systems Case v2.0 — SCORING GUIDE
# GGF LLM Sistemleri Case v2.0 — PUANLAMA REHBERI

> Detailed 100-point rubric for evaluators.
> Degerlendirmeciler icin detayli 100 puanlik rubrik.

---

## Total: 100 Points / Toplam: 100 Puan

| Phase / Faz | Points / Puan | Focus / Odak |
|-------------|---------------|--------------|
| Phase 1: RAG Pipeline | 30 | Retrieval quality and implementation / Geri getirme kalitesi ve implementasyon |
| Phase 2: Prompting | 20 | Structured output and prompt quality / Yapilandirilmis cikti ve prompt kalitesi |
| Phase 3: Fine-Tuning | 30 | Training data and model improvement / Egitim verisi ve model iyilestirme |
| Phase 4: Analytics | 20 | Failure analysis and experimentation / Hata analizi ve deneysel calisma |

---

## Phase 1: RAG Pipeline & Retrieval Quality (30 pts)
## Faz 1: RAG Hatti ve Geri Getirme Kalitesi (30 puan)

### 1.1 BM25 Implementation / BM25 Uygulamasi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Full BM25 with tokenizer, inverted index, IDF computation, and k1/b parameters / Tokenizer, ters indeks, IDF hesaplama ve k1/b parametreleri ile tam BM25 |
| **3** | Basic BM25 scoring works but missing customizable parameters or edge cases / Temel BM25 puanlama calisir ama ozellestirilmis parametreler veya uc durumlar eksik |
| **1** | BM25 file exists with stub/partial implementation / BM25 dosyasi var ama taslak/kismi uygulama |
| **0** | No BM25 implementation / BM25 uygulamasi yok |

**Check / Kontrol:** `phase1_rag.mjs` > BM25 checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/rag/bm25.py`

### 1.2 Embedding Retrieval Integration / Embedding Geri Getirme Entegrasyonu (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Embedding retrieval works with sentence-transformers, configurable model, cosine similarity / sentence-transformers ile embedding geri getirme, yapilandirilebilir model, kosinus benzerligi |
| **3** | Embedding retrieval implemented but hardcoded model or missing error handling / Embedding geri getirme uygulanmis ama model sabit kodlanmis veya hata yonetimi eksik |
| **1** | Embedding code exists but not functional or not integrated / Embedding kodu var ama islevsel degil veya entegre edilmemis |
| **0** | No embedding retrieval / Embedding geri getirme yok |

**Check / Kontrol:** `phase1_rag.mjs` > Embedding strategy check
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/rag/retriever.py`

### 1.3 Hybrid Retrieval with RRF / RRF ile Hibrit Geri Getirme (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Reciprocal Rank Fusion combining keyword + BM25 + embeddings, configurable weights / Anahtar kelime + BM25 + embedding birlestiren RRF, yapilandirilebilir agirliklar |
| **3** | Hybrid works with 2 strategies or RRF is correct but weights aren't tunable / Hibrit 2 strateji ile calisir veya RRF dogru ama agirliklar ayarlanamaz |
| **1** | Hybrid code exists but doesn't properly combine results / Hibrit kodu var ama sonuclari duzgun birlestirmiyor |
| **0** | No hybrid retrieval / Hibrit geri getirme yok |

**Check / Kontrol:** `phase1_rag.mjs` > Hybrid checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/rag/hybrid.py`

### 1.4 AST-Aware Chunking / AST-Duyarli Parcalama (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Chunks at function/class/interface boundaries, handles nested structures, imports extraction / Fonksiyon/sinif/arayuz sinirlarinda parcalama, ic ice yapilar, import cikarimi |
| **3** | Basic AST chunking works but misses some boundary types / Temel AST parcalama calisir ama bazi sinir turleri eksik |
| **1** | AST chunking code exists but falls back to fixed for most cases / AST parcalama kodu var ama cogu durumda sabit pencereye geri donuyor |
| **0** | Only fixed-window chunking / Sadece sabit pencere parcalama |

**Check / Kontrol:** `phase1_rag.mjs` > AST chunking checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/rag/indexer.py`

### 1.5 Precision@5 >= 0.6 on Gold Labels / Altin Etiketlerde Precision@5 >= 0.6 (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | precision@5 >= 0.6 across all tasks using gold labels / Altin etiketler kullanilarak tum gorevlerde precision@5 >= 0.6 |
| **3** | precision@5 >= 0.4 |
| **1** | precision@5 >= 0.2 |
| **0** | precision@5 < 0.2 or metrics not computed / precision@5 < 0.2 veya metrikler hesaplanmamis |

**Check / Kontrol:** Run `ggf-case metrics` and check output / `ggf-case metrics` calistirin ve ciktiyi kontrol edin
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/metrics/retrieval_metrics.py`, `eval/gold_labels.json`

### 1.6 MRR >= 0.7 on Gold Labels / Altin Etiketlerde MRR >= 0.7 (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | MRR >= 0.7 across all tasks / Tum gorevlerde MRR >= 0.7 |
| **3** | MRR >= 0.5 |
| **1** | MRR >= 0.3 |
| **0** | MRR < 0.3 or not computed / MRR < 0.3 veya hesaplanmamis |

**Check / Kontrol:** Run `ggf-case metrics` and check output / `ggf-case metrics` calistirin ve ciktiyi kontrol edin
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/metrics/retrieval_metrics.py`

---

## Phase 2: Prompt Engineering & Structured Output (20 pts)
## Faz 2: Prompt Muhendisligi ve Yapilandirilmis Cikti (20 puan)

### 2.1 Structured Output Parsing / Yapilandirilmis Cikti Ayristirma (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Pydantic models with validation, multi-strategy JSON extraction, error recovery / Dogrulama ile Pydantic modeller, coklu strateji JSON cikarimi, hata kurtarma |
| **3** | Pydantic models exist, basic JSON extraction works / Pydantic modeller var, temel JSON cikarimi calisir |
| **1** | Models exist but extraction is fragile / Modeller var ama cikarim kirilgan |
| **0** | No structured output handling / Yapilandirilmis cikti yonetimi yok |

**Check / Kontrol:** `phase2_prompting.mjs` > Structured output checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/llm/structured_output.py`

### 2.2 Chain-of-Thought Template Quality / Dusunce Zinciri Sablon Kalitesi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | CoT templates with step-by-step reasoning, task analysis, edge case consideration / Adim adim akil yurutme, gorev analizi, uc durum degerlendirmesi iceren CoT sablonlari |
| **3** | CoT templates exist but reasoning steps are generic / CoT sablonlari var ama akil yurutme adimlari genel |
| **1** | Templates exist but no structured reasoning / Sablonlar var ama yapilandirilmis akil yurutme yok |
| **0** | No chain-of-thought templates / Dusunce zinciri sablonlari yok |

**Check / Kontrol:** `phase2_prompting.mjs` > CoT checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/llm/structured_output.py`

### 2.3 Patch Format Compliance / Yama Format Uyumlulugu (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | All 10 tasks produce valid unified diffs that parse correctly / 10 gorevden hepsi dogru ayristirilan gecerli unified diff uretir |
| **3** | 7+ tasks produce valid diffs / 7+ gorev gecerli diff uretir |
| **1** | 4+ tasks produce valid diffs / 4+ gorev gecerli diff uretir |
| **0** | Most patches are malformed / Yamalarin cogu bozuk formatta |

**Check / Kontrol:** Run `ggf-case run-eval` and check diff guard pass rate / `ggf-case run-eval` calistirin ve diff guard gecis oranini kontrol edin
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/patch/diff_guard.py`

### 2.4 Prompt Effectiveness / Prompt Etkinligi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Prompts produce passing patches for >= 7/10 tasks / Promptlar >= 7/10 gorev icin gecen yamalar uretir |
| **3** | Prompts produce passing patches for >= 5/10 tasks / Promptlar >= 5/10 gorev icin gecen yamalar uretir |
| **1** | Prompts produce passing patches for >= 3/10 tasks / Promptlar >= 3/10 gorev icin gecen yamalar uretir |
| **0** | Less than 3 tasks pass / 3'ten az gorev gecer |

**Check / Kontrol:** Run `ggf-case run-eval` and check pass rate / `ggf-case run-eval` calistirin ve gecis oranini kontrol edin

---

## Phase 3: Fine-Tuning & Training Data Curation (30 pts)
## Faz 3: Fine-Tuning ve Egitim Verisi Duzenleme (30 puan)

### 3.1 Training Data Correctly Formatted / Egitim Verisi Dogru Formatlanmis (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | 50+ examples in valid JSONL, correct schema, quality labels, diverse task coverage / Gecerli JSONL'de 50+ ornek, dogru sema, kalite etiketleri, cesitli gorev kapsamasi |
| **3** | Valid JSONL with 30+ examples and correct schema / 30+ ornek ile gecerli JSONL ve dogru sema |
| **1** | JSONL exists but schema issues or few examples / JSONL var ama sema sorunlari veya az ornek |
| **0** | No training data or invalid format / Egitim verisi yok veya gecersiz format |

**Check / Kontrol:** `phase3_finetune.mjs` > Training data checks
**Key files / Anahtar dosyalar:** `eval/training_data/examples.jsonl`

### 3.2 Train/Val Split / Egitim/Dogrulama Bolumu (3 pts / 3 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **3** | 80/20 split with stratification by task, reproducible seed / Goreve gore katmanli 80/20 bolum, tekrarlanabilir seed |
| **2** | Split exists but not stratified / Bolum var ama katmanli degil |
| **1** | Random split without proper proportions / Uygun oranlar olmadan rastgele bolum |
| **0** | No split implementation / Bolum uygulamasi yok |

**Check / Kontrol:** `phase3_finetune.mjs` > Split checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/finetune/data_curator.py`

### 3.3 Fine-Tune API Integration / Fine-Tune API Entegrasyonu (7 pts / 7 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **7** | Full OpenAI fine-tuning API: upload, create job, poll, status, list jobs / Tam OpenAI fine-tuning API: yukleme, is olusturma, yoklama, durum, is listeleme |
| **5** | API integration works but missing polling or error handling / API entegrasyonu calisir ama yoklama veya hata yonetimi eksik |
| **3** | Basic API calls implemented but not fully functional / Temel API cagirilari uygulanmis ama tam islevsel degil |
| **1** | API stubs exist / API taslaklari var |
| **0** | No API integration / API entegrasyonu yok |

**Check / Kontrol:** `phase3_finetune.mjs` > Trainer checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/finetune/trainer.py`

### 3.4 Base vs Fine-Tuned Comparison Report / Temel vs Fine-Tuned Karsilastirma Raporu (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Detailed per-task comparison with pass rates, improvement flags, rich output / Gecis oranlari, iyilestirme bayraklari, zengin cikti ile detayli gorev bazinda karsilastirma |
| **3** | Basic comparison report with aggregate metrics / Toplu metriklerle temel karsilastirma raporu |
| **1** | Comparison code exists but incomplete / Karsilastirma kodu var ama eksik |
| **0** | No comparison capability / Karsilastirma yetkinligi yok |

**Check / Kontrol:** `phase3_finetune.mjs` > Evaluator checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/finetune/evaluator.py`

### 3.5 Pass Rate Improvement >= 20% / Gecis Orani Iyilestirmesi >= %20 (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Fine-tuned model shows >= 20% improvement over base model / Fine-tuned model temel modele gore >= %20 iyilesme gosterir |
| **3** | 10-19% improvement / %10-19 iyilesme |
| **1** | Some improvement but < 10% / Bir miktar iyilesme ama < %10 |
| **0** | No improvement or not tested / Iyilesme yok veya test edilmemis |

**Note / Not:** This requires actually running fine-tuning. Document results in report.md. / Bu, fine-tuning'in gercekten calistirilmasini gerektirir. Sonuclari report.md'de belgeleyin.

### 3.6 Hyperparameter Documentation / Hiperparametre Dokumantasyonu (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Documented learning rate, epochs, batch size with rationale for choices / Secim gerekceleri ile belgelenmis ogrenme orani, epoch, batch boyutu |
| **3** | Parameters listed but rationale is generic / Parametreler listelenmi ama gerekce genel |
| **1** | Some parameters mentioned / Bazi parametreler bahsedilmis |
| **0** | No hyperparameter documentation / Hiperparametre dokumantasyonu yok |

**Check / Kontrol:** Review `trainer.py` comments and `report.md` / `trainer.py` yorumlarini ve `report.md`'yi inceleyin

---

## Phase 4: Analytics, Experiment Design & Failure Analysis (20 pts)
## Faz 4: Analitik, Deney Tasarimi ve Hata Analizi (20 puan)

### 4.1 Failure Attribution / Hata Atif (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | All failures classified into correct categories with evidence-based attribution / Tum hatalar kanit tabanli atif ile dogru kategorilere siniflandirilmis |
| **3** | Most failures classified, some misattributions / Cogu hata siniflandirilmis, bazi yanlis atiflar |
| **1** | Basic classification exists but accuracy is low / Temel siniflandirma var ama dogruluk dusuk |
| **0** | No failure classification / Hata siniflandirmasi yok |

**Check / Kontrol:** `phase4_analytics.mjs` > Failure analyzer checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/analytics/failure_analyzer.py`

### 4.2 Root Cause Analysis Quality / Kok Neden Analizi Kalitesi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Patterns identified, correlations analyzed, actionable recommendations generated / Oruntular tanimlanmis, korelasyonlar analiz edilmis, uygulanabilir oneriler uretilmis |
| **3** | Patterns found but recommendations are generic / Oruntular bulunmus ama oneriler genel |
| **1** | Basic analysis without meaningful insights / Anlamli icgoruler olmadan temel analiz |
| **0** | No root cause analysis / Kok neden analizi yok |

**Check / Kontrol:** `phase4_analytics.mjs` > Root cause checks

### 4.3 A/B Experiment with Statistical Significance / Istatistiksel Anlamlilik ile A/B Deneyi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Paired t-test, Cohen's d, p-value, correct statistical methodology / Eslestirilmis t-testi, Cohen's d, p-degeri, dogru istatistiksel metodoloji |
| **3** | Statistical test implemented but methodology has issues / Istatistiksel test uygulanmis ama metodolojide sorunlar var |
| **1** | Experiment framework exists without proper statistics / Deney cercevesi var ama uygun istatistik yok |
| **0** | No experiment framework / Deney cercevesi yok |

**Check / Kontrol:** `phase4_analytics.mjs` > Experiment checks
**Key files / Anahtar dosyalar:** `solution/src/ggf_case/analytics/experiment.py`

### 4.4 Final Report Completeness / Son Rapor Tamligi (5 pts / 5 puan)

| Score / Puan | Criteria / Kriterler |
|-------|----------|
| **5** | Complete report with metrics tables, experiment results, failure analysis, recommendations / Metrik tablolari, deney sonuclari, hata analizi, oneriler iceren tam rapor |
| **3** | Report has most sections but some are thin / Rapor cogu bolume sahip ama bazilari yetersiz |
| **1** | Basic report template filled / Temel rapor sablonu doldurulmus |
| **0** | Report not submitted or mostly empty / Rapor teslim edilmemis veya buyuk olcude bos |

**Check / Kontrol:** Review `report.md` / `report.md`'yi inceleyin

---

## Quick Scoring Worksheet / Hizli Puanlama Calismasi

```
Phase 1 / Faz 1 (30 pts / 30 puan):
  [ ] 1.1 BM25:                    ___/5
  [ ] 1.2 Embeddings:              ___/5
  [ ] 1.3 Hybrid RRF / Hibrit RRF: ___/5
  [ ] 1.4 AST Chunking / Parcalama: ___/5
  [ ] 1.5 Precision@5:              ___/5
  [ ] 1.6 MRR:                      ___/5
  Phase 1 Total / Faz 1 Toplam:    ___/30

Phase 2 / Faz 2 (20 pts / 20 puan):
  [ ] 2.1 Structured Output / Yapilandirilmis Cikti: ___/5
  [ ] 2.2 CoT Templates / CoT Sablonlari:            ___/5
  [ ] 2.3 Format Compliance / Format Uyumlulugu:      ___/5
  [ ] 2.4 Effectiveness / Etkinlik:                   ___/5
  Phase 2 Total / Faz 2 Toplam:                       ___/20

Phase 3 / Faz 3 (30 pts / 30 puan):
  [ ] 3.1 Data Format / Veri Formati:                 ___/5
  [ ] 3.2 Train/Val Split / Egitim/Dogrulama Bolumu:  ___/3
  [ ] 3.3 API Integration / API Entegrasyonu:          ___/7
  [ ] 3.4 Comparison Report / Karsilastirma Raporu:    ___/5
  [ ] 3.5 Improvement / Iyilestirme:                   ___/5
  [ ] 3.6 Hyperparams Doc / Hiperparametre Dok:        ___/5
  Phase 3 Total / Faz 3 Toplam:                        ___/30

Phase 4 / Faz 4 (20 pts / 20 puan):
  [ ] 4.1 Failure Attribution / Hata Atfi:             ___/5
  [ ] 4.2 Root Cause Analysis / Kok Neden Analizi:     ___/5
  [ ] 4.3 A/B Experiment / A/B Deneyi:                 ___/5
  [ ] 4.4 Report Completeness / Rapor Tamligi:         ___/5
  Phase 4 Total / Faz 4 Toplam:                        ___/20

GRAND TOTAL / GENEL TOPLAM: ___/100
```

---

## Grade Scale / Not Skalasi

| Score / Puan | Grade / Not | Description (EN) | Aciklama (TR) |
|-------|-------|-------------------|---------------|
| 90-100 | A+ | Exceptional — production-quality implementation | Olaganustu — uretim kalitesinde uygulama |
| 80-89 | A | Excellent — all phases well-implemented | Mukemmel — tum fazlar iyi uygulanmis |
| 70-79 | B+ | Very Good — strong in most phases | Cok iyi — cogu fazda guclu |
| 60-69 | B | Good — solid baseline with some advanced features | Iyi — saglam temel, bazi gelismis ozellikler |
| 50-59 | C | Adequate — basic implementation works | Yeterli — temel uygulama calisir |
| 40-49 | D | Below expectations — significant gaps | Beklentinin altinda — onemli eksiklikler |
| <40 | F | Insufficient — major components missing | Yetersiz — buyuk bilesenler eksik |

---

**End of Scoring Guide / Puanlama Rehberi Sonu**
