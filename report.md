# GGF LLM Systems Case v2.0 — Report / Rapor

> Fill this document after completing all 4 phases of the evaluation.
> Degerlendirmenin 4 fazini tamamladiktan sonra bu dokumani doldurun.

---

## Candidate Information / Aday Bilgileri

- **Name / Isim:**
- **Date / Tarih:**
- **Time Spent / Harcanan Sure:**

---

## 1. Approach / Yaklasim

_Describe your overall approach to solving this case. / Bu case'i cozmek icin genel yaklasiminizi aciklayin._

### Phase 1: RAG Strategy / RAG Stratejisi

_How did you implement BM25, hybrid retrieval, and AST-aware chunking?_
_BM25, hibrit geri getirme ve AST-duyarli parcalamayi nasil uyguladiniz?_

```
[Your answer here / Cevabiniz buraya]
```

### Phase 2: Prompt Engineering / Prompt Muhendisligi

_How did you structure CoT templates and structured output? What model did you use?_
_CoT sablonlarini ve yapilandirilmis ciktiyi nasil yapilandirdiniz? Hangi modeli kullandiniz?_

```
[Your answer here / Cevabiniz buraya]
```

### Phase 3: Fine-Tuning Strategy / Fine-Tuning Stratejisi

_How did you curate training data? What hyperparameters did you choose and why?_
_Egitim verilerini nasil duzenlediniz? Hangi hiperparametreleri secdiniz ve neden?_

```
[Your answer here / Cevabiniz buraya]
```

### Phase 4: Analytics Approach / Analitik Yaklasim

_How did you design experiments and analyze failures?_
_Deneyleri nasil tasarladiniz ve hatalari nasil analiz ettiniz?_

```
[Your answer here / Cevabiniz buraya]
```

### Key Decisions / Temel Kararlar

_What were the most important technical decisions you made?_
_Aldiginiz en onemli teknik kararlar nelerdi?_

```
[Your answer here / Cevabiniz buraya]
```

---

## 2. Results / Sonuclar

### Overall Summary / Genel Ozet

| Metric | Value |
|--------|-------|
| Total Tasks | 10 |
| Passed | |
| Failed | |
| Pass Rate | |
| Total Time (seconds) | |

### Retrieval Metrics / Geri Getirme Metrikleri

_Results from `ggf-case metrics`:_

| Metric | Value |
|--------|-------|
| Precision@5 | |
| Recall@5 | |
| MRR | |
| NDCG@5 | |
| Hit Rate | |
| Retrieval Strategy | |

### Per-Task Results / Gorev Bazinda Sonuclar

| Task | Phase | Status | Duration | Notes |
|------|-------|--------|----------|-------|
| task_01 - Pause Toggle | Phase 2 | | | |
| task_02 - Input Remap | Phase 2 | | | |
| task_03 - Score Combo | Phase 2 | | | |
| task_04 - Enemy Patrol | Phase 1 | | | |
| task_05 - Save V2 | Phase 1 | | | |
| task_06 - Difficulty Speed | Phase 3 | | | |
| task_07 - Event Log | Phase 3 | | | |
| task_08 - Cooldown | Phase 3 | | | |
| task_09 - Deterministic RNG | Phase 4 | | | |
| task_10 - Settings Validation | Phase 4 | | | |

### Phase Check Results / Faz Kontrol Sonuclari

_Output from phase check scripts:_

| Phase | Passed | Total | Score |
|-------|--------|-------|-------|
| Phase 1: RAG | | | /30 |
| Phase 2: Prompting | | | /20 |
| Phase 3: Fine-Tuning | | | /30 |
| Phase 4: Analytics | | | /20 |
| **Total** | | | **/100** |

---

## 3. Failure Analysis / Hata Analizi

_For each failing task, classify the failure and describe root cause._
_Her basarisiz gorev icin hatayi siniflandirin ve kok nedeni aciklayin._

### Failure Summary / Hata Ozeti

| Task | Failure Category | Root Cause |
|------|-----------------|------------|
| | | |

### Detailed Analysis / Detayli Analiz

#### Task: [task_id]

**Failure Category:** (retrieval_miss / generation_error / apply_failure / build_failure / check_failure)

**Root Cause:**

**Retrieval Quality Assessment:**

**Suggested Fix:**

---

## 4. Fine-Tuning Results / Fine-Tuning Sonuclari

### Training Data Statistics / Egitim Verisi Istatistikleri

| Metric | Value |
|--------|-------|
| Total Examples | |
| Valid Examples | |
| Train Size | |
| Val Size | |
| Avg Input Tokens | |
| Avg Output Tokens | |

### Hyperparameters / Hiperparametreler

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | | |
| Epochs | | |
| Batch Size | | |
| Learning Rate Multiplier | | |
| Suffix | | |

### Base vs Fine-Tuned Comparison / Temel vs Fine-Tuned Karsilastirma

| Task | Base Model | Fine-Tuned | Change |
|------|-----------|------------|--------|
| task_01 | | | |
| task_02 | | | |
| task_03 | | | |
| task_04 | | | |
| task_05 | | | |
| task_06 | | | |
| task_07 | | | |
| task_08 | | | |
| task_09 | | | |
| task_10 | | | |
| **Pass Rate** | | | |

---

## 5. Experiment Results / Deney Sonuclari

### Experiment Design / Deney Tasarimi

_What configurations did you compare? (e.g., keyword vs hybrid retrieval)_
_Hangi konfigurasyonlari karsilastirdiniz?_

| Variant | Description | Config |
|---------|-------------|--------|
| A | | |
| B | | |

### Statistical Results / Istatistiksel Sonuclar

| Metric | Variant A | Variant B | t-stat | p-value | Significant? |
|--------|-----------|-----------|--------|---------|-------------|
| Pass Rate | | | | | |

### Conclusion / Sonuc

```
[Which variant won and why? / Hangi varyant kazandi ve neden?]
```

---

## 6. Improvements Made / Yapilan Iyilestirmeler

_List the improvements you made to the baseline solution._
_Baseline cozume yaptiginiz iyilestirmeleri listeleyin._

### Phase 1: RAG
1.
2.
3.

### Phase 2: Prompting
1.
2.

### Phase 3: Fine-Tuning
1.
2.
3.

### Phase 4: Analytics
1.
2.

---

## 7. What I Would Do with More Time / Daha Fazla Zamanla Ne Yapardim

_If you had another 8-12 hours, what would you improve?_
_8-12 saat daha olsaydi neyi iyilestirirdiniz?_

```
[Your answer here / Cevabiniz buraya]
```

---

## 8. LLM / Model Information / LLM / Model Bilgisi

| Parameter / Parametre | Value / Deger |
|-----------|-------|
| Model / Model | |
| Base URL / Temel URL | |
| Temperature / Sicaklik | |
| Max Tokens / Maks Token | |
| Embedding Model (if used) / Embedding Modeli (kullanildiysa) | |
| Vector DB (if used) / Vektor DB (kullanildiysa) | |
| Fine-tuned Model ID (if created) / Fine-tuned Model ID (olusturulduysa) | |

---

## 9. Environment / Ortam

| Component / Bilesen | Version / Surum |
|-----------|---------|
| OS / Isletim Sistemi | |
| Node.js | |
| Python | |
| Docker (if used / kullanildiysa) | |
