# Business Entity Resolution — Strategy & Design Document
> Last updated: Round 9 — Match cardinality distribution confirmed (Q32); test_source1 non-Latin confirmed as French accented-Latin (risk closed); K=20 and 1:3 pair ratio validated | Status: **ALL DECISIONS LOCKED — READY FOR IMPLEMENTATION**

---

## 0. Problem Recap (Locked)

| Item | Detail |
|---|---|
| Task | Match S2/S3 → S1 (deduplicated reference source) |
| Metric | **F_0.5** — precision weighted 2× over recall |
| Countries (train) | US, India |
| Countries (test) | US, India, France (unseen in training) |
| Model constraint | MIT/Apache 2.0, ≤ 8B parameters |
| Local GPU | RTX 4050 Windows 11 — 6 GB VRAM |
| AWS Notebook | ml.t3.medium (FREE — 250 hrs) |
| AWS Training Job | ml.g4dn.xlarge (T4 16GB VRAM, paid via $200 credits) |
| AWS Credits | **$200 for all participants** + extra $100 for Top 500 at 48-hr mark |
| External lookups | STRICTLY PROHIBITED |

---

## 1. ALL LOCKED DECISIONS (Rounds 1–5)

| # | Topic | Decision |
|---|---|---|
| Q1 | Name abbreviation dict | Hybrid (hardcoded + learned from training pairs) |
| Q2 | Landmark phrases | Downweight (keep, penalise contribution) |
| Q3 | Transliteration | Country-specific normaliser plugin system |
| Q4 | Country blocking | Hard filter + soft feature, country normalised first |
| Q5 | Blocking keys | Iterative — add until recall ≥ 95% |
| Q6 | TF-IDF top-K | K = 20 |
| Q7 | Embedding features | Yes — BGE-m3 cosine is primary feature |
| Q8 | Model | Two-stage: LightGBM → BGE-reranker-v2-m3 |
| Q10 | French legal suffixes | SAS, SARL, EURL, SASU, SNC, GIE, SCI |
| Q11 | Country alias fuzzy | Dict + Levenshtein ≤ 2 fallback |
| Q12 | Bi-encoder | BAAI/bge-m3 (MIT, 570M, 100+ languages) |
| Q13 | Cross-encoder | BAAI/bge-reranker-v2-m3 (MIT, multilingual) |
| Q14 | Two-stage thresholds | < 0.60 REJECT · 0.60–0.90 → reranker · > 0.90 ACCEPT |
| Q15 | Validation | Triple eval (stratified split + cross-country proxy + final 100%) |
| Q16 | ANN library | FAISS |
| Q18 | Blocking primary | BGE-m3 FAISS primary + TF-IDF supplement |
| Q19 | NULL/NaN address | Empty string + `addr_is_null` flag; exhaustive 37-pattern handler |
| Q20 | Domain names | `base_domain` (TLD stripped) + `tld` as separate feature |
| Q22 | Duplicate tokens | Deduplicate repeated adjacent tokens in names |
| Q23 | Bracket notation | Strip brackets, keep inner content |
| Q24 | AWS credits | **$200 confirmed** — GPU instances affordable with credits |
| Q25 | AWS interface | **SageMaker Notebook Instance** `ml.t3.medium` (free tier, 250 hrs) |
| Q26 | Dataset size | train S1=2,206,821, S2=5,034,616, S3=5,285,603 · test S1=1,732,544, S2=4,887,273, S3=5,082,316 · **~24.2M total** · singleton ratio 5.58% · France zero-shot (test only) |
| Q27 | Hard negative ratio | **80% same-FAISS-bucket hard + 20% random within-country easy** |
| Q29 | Indic transliteration | **YES — add `indic-transliteration` before TF-IDF** |
| Q30 | Blocking at 24M scale | **Keep BGE-m3 primary blocking** on `ml.g4dn.4xlarge` (see Section 2F) |
| Q31 | GPU instance | **`ml.g4dn.4xlarge`** — 1×T4 16GB VRAM, 16 vCPUs, 64GB RAM, $1.505/hr |

---

## 2. ✅ AWS STRATEGY — FULLY LOCKED (Scale-Revised)

### 2A — What the Blog Confirms

From the Amazon ML Challenge 2026 AWS prep blog:

| Resource | Detail | Cost |
|---|---|---|
| **Notebook Instance** | `ml.t3.medium` (2 vCPU, 4 GB RAM) | FREE — 250 hrs/2 months |
| **Training Job** | `ml.m5.xlarge` (4 vCPU, 16 GB RAM, CPU) | FREE — 50 hrs/2 months |
| **Training Job GPU** | `ml.g4dn.xlarge` (T4 GPU, 16 GB VRAM) | ~$0.74/hr — paid from credits |
| **S3 Storage** | Object store for data, embeddings, models | 5 GB always free |
| **AWS Credits** | All participants get $200 | Covers GPU Training Jobs easily |
| **Extra Credits** | Top 500 teams get +$100 at 48-hr mark | Bonus if ranked well |
| **Region** | `us-east-1` | Best compatibility (as per blog) |

### 2B — Critical Insight: AWS T4 > Local RTX 4050

| Spec | Local RTX 4050 | AWS g4dn.xlarge T4 |
|---|---|---|
| VRAM | **6 GB** | **16 GB** ← 2.7× more |
| RAM | 16 GB (system) | 16 GB (instance) |
| Cost | Free (already have it) | ~$0.74/hr from $200 credits |
| Setup | Local Python env on Windows | SageMaker managed |
| Fine-tune batch size | 4 (tight) | 16–32 (comfortable) |

**Conclusion**: For GPU tasks, **AWS Training Job on g4dn.xlarge is better than local RTX 4050** because of 2.7× more VRAM. Fine-tuning bge-reranker-v2-m3 (570M) at batch_size=16 in fp16 needs ~4GB VRAM — fits comfortably on T4. 

Budget math: 72-hour hackathon. Realistically 5-10 hours of GPU Training Jobs total → $4–7.50. Well within $200.

### 2C — Final Compute Allocation

```
TASK                              WHERE           WHY
─────────────────────────────────────────────────────────────────────
Data exploration                  Notebook (free) Lightweight
Normaliser / pre-cleaner dev      Notebook (free) CPU only
TF-IDF blocking                   Notebook (free) CPU / sklearn
LightGBM training                 Notebook (free) CPU-friendly, fast
validate_submission.py            Notebook (free) stdlib only
─────────────────────────────────────────────────────────────────────
BGE-m3 embedding (all records)    Training Job    T4 GPU, fp16, fast
                                  ml.g4dn.xlarge  16 GB VRAM
BGE-reranker fine-tuning          Training Job    T4 GPU, batch_size=16
                                  ml.g4dn.xlarge  More VRAM than local
─────────────────────────────────────────────────────────────────────
FAISS index build                 Notebook        CPU, fast once emb ready
Feature engineering               Notebook        CPU, pandas
Final inference (test set)        Notebook        Loads saved embeddings
─────────────────────────────────────────────────────────────────────
All data + embeddings             S3 (us-east-1)  5 GB free + credits
```

### 2D — SageMaker Setup Steps (Beginner-Friendly)

```
1. Go to SageMaker console → Notebook → Notebook instances
2. Create notebook instance:
   Name: er-challenge-notebook
   Instance type: ml.t3.medium
   IAM Role: Create new role (leave defaults)
3. Open JupyterLab → conda_python3 kernel
4. Set region: us-east-1

For GPU Training Jobs:
5. Write training script (train_embeddings.py / train_reranker.py)
6. Upload to S3: s3://sagemaker-<account>-us-east-1/er-challenge/
7. Create Training Job → instance ml.g4dn.xlarge → point to S3 script
8. Job downloads data, trains, saves model/embeddings back to S3
9. Download .npy files from S3 back to notebook
```

### 2E — Important Billing Warnings (from blog)

- **Stop, don't delete** notebook when done for the day → files preserved, no charge
- **Delete endpoints** if any deployed (charge $0.12/hr even idle)
- **Set billing alert** immediately after account creation
- S3 5GB free → BGE-m3 embeddings for large dataset may exceed this → covered by credits

### 2F — Scale Analysis (Q26 Confirmed — Data Exploration Complete)

#### Dataset Row Counts (Confirmed by Exploration)

| File | Rows | US | India | France |
|---|---|---|---|---|
| train_source1.tsv | 2,206,821 | 1,323,633 | 883,188 | — |
| train_source2.tsv | 5,034,616 | 3,016,817 | 2,017,799 | — |
| train_source3.tsv | 5,285,603 | 3,170,056 | 2,115,547 | — |
| train_ground_truth.tsv | 2,206,821 | — | — | — |
| **Train total** | **12,527,040** | **7,510,506 (60%)** | **5,016,534 (40%)** | **—** |
| test_source1.tsv | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test_source2.tsv | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test_source3.tsv | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |
| **Test total** | **11,702,133** | **4,480,137 (38%)** | **5,527,551 (47%)** | **1,694,445 (14%)** |
| **GRAND TOTAL** | **~24.2M** | | | |

**Key distribution facts:**
- France = **zero training data** → ~1.69M test records the pipeline has never seen a labelled example for. Eval 2 cross-country proxy (US-train → India-validate) is the only pre-submission signal.
- India is heavier in test (47%) than train (40%). The pipeline must handle India well.
- US/India ratio shifts ~60/40 train → ~44/54 test — calibrate threshold sweeps on the test-realistic split.

#### Singleton Analysis (Confirmed)

| Metric | Value |
|---|---|
| Total S1 entities (train) | 2,206,821 |
| Singletons (no GT match) | 123,247 |
| Singleton ratio | **5.58%** |
| Matchable S1 entities | **2,083,574 (94.4%)** |

**Pipeline implication — CRITICAL for evaluation harness:** the 5.58% singleton ceiling is structural, not a tuning problem. Blocking recall ≥ 95% must be measured against **2,083,574 matchable S1 entities only**, not 2,206,821 total. At F0.5 (precision-weighted 2×), forcing singletons into matches hurts precision far more than the recall gain. Do not tune thresholds to recover singletons.

#### Non-Latin Script Distribution (Confirmed)

| File | Non-Latin records | % of file |
|---|---|---|
| train_source1.tsv | **0** | 0.0% |
| **test_source1.tsv** | **40,789** | **2.4%** |
| train_source2.tsv | 764,608 | 15.2% |
| test_source2.tsv | 928,158 | 19.0% |
| train_source3.tsv | 606,737 | 11.5% |
| test_source3.tsv | 737,515 | 14.5% |

**S1-clean finding — critical:** S1 train is 100% Latin-script. Every cross-script match is a non-Latin S2/S3 query → Latin S1 reference. TF-IDF cosine on those pairs is exactly **0** — BGE-m3 FAISS is the *only* blocker that can retrieve them. This validates Q7/Q12/Q18 decisively and means blocking recall on Indian-script records has no lexical fallback.

**New risk — test_source1 non-Latin (40,789 records):** train_source1 is 100% Latin but test_source1 has 40,789 non-Latin S1 reference records. These are the FAISS index entries in test — an S2/S3 query hitting one of these must match via BGE-m3 semantic similarity alone, with no Latin-side fallback either. Flag for post-submission analysis if France/India F0.5 is unexpectedly low. The script of these 40,789 records is unconfirmed — if they are French accented-Latin (é, ç, à) the counter fired on accents and the impact is minor; if they are Indian-script S1 references the situation is harder (no training labels exist for non-Latin S1).

#### NULL / NaN Address Distribution (Confirmed)

| File | NULL addresses | % of file |
|---|---|---|
| train_source1.tsv | 0 | 0.0% |
| test_source1.tsv | 0 | 0.0% |
| train_source2.tsv | 168,967 | 3.4% |
| test_source2.tsv | 129,408 | 2.6% |
| train_source3.tsv | 175,916 | 3.3% |
| test_source3.tsv | 136,098 | 2.7% |

S1 has zero null addresses in both splits — consistent with S1 being the cleaned reference. All null-address noise originates in S2/S3 (~3.3% of those files). The `addr_is_null_either` feature (§7) directly flags these pairs; the LightGBM model must handle name-only feature vectors for ~3.3% of candidates. Consider oversampling null-address pairs in reranker training data.

#### Domain-Name Distribution (Confirmed)

| File | Domain-name records | % of file |
|---|---|---|
| train_source1.tsv | 4 | <0.01% |
| test_source1.tsv | 13 | <0.01% |
| train_source2.tsv | 201,286 | 4.0% |
| test_source2.tsv | 156,069 | 3.2% |
| train_source3.tsv | 210,942 | 4.0% |
| test_source3.tsv | 164,040 | 3.2% |

Domain names are almost exclusively a S2/S3 noise pattern. The canonical case (noise audit pair #5): `cardiologymetrocare.com` → `Cardiology Metro Care Associates Inc` — the address matches precisely, so `bgem3_full_cosine` rescues low name cosine via the address component. The `name_is_domain_both` and `domain_base_jaro` features handle the rarer both-domains case.

#### Match Cardinality Distribution (Q32 — Confirmed)

How many S2/S3 matches a single S1 entity has, across the full ground truth:

| Statistic | Value |
|---|---|
| Median | 3 matches per S1 |
| Mean | ~3.46 matches per S1 |
| 95th percentile | 6 matches per S1 |
| 99th percentile | 8 matches per S1 |
| Maximum | **11 matches per S1** |

**K=20 confirmed safe:** even the maximum-cardinality entity (11 matches) leaves 9 spare FAISS retrieval slots for near-misses and hard negatives. No entity can exhaust K=20. No change to the blocking top-K.

**Reranker pair construction confirmed tractable:** at the median of 3 positives, the 1:3 positive-to-negative ratio yields ~9 hard negatives per entity. No S1 entity has enough positives to dominate a training batch. The 80/20 hard/easy negative split holds as designed.

**Implication for candidate_pairs.tsv size:**
```
Upper bound: 10,233,574 (matchable S1) × 20 (top-K) = ~204M candidate pairs
Realistic: many S1 entities share top-K candidates → actual ~150–180M pairs
After deduplication + ground-truth join → training set is manageable for LightGBM
```

#### Visual Noise Audit (30-pair sample — Confirmed)

Five representative pairs from the 30-pair ground truth audit:

| # | Noise class | S1 (reference) | S2/S3 (noisy) | Pipeline handler |
|---|---|---|---|---|
| 1 | Word transposition + punctuation | Sarasva India Limited · 9/1/3 Kasundia 2Nd Bye Lane | limited sarasva india · 9/1/3. KASUNDIA 2ND BYE LANE | Token-sort name; addr token-sort; ordinal normalisation |
| 2 | OCR typo + null address | Lambodar Softwares · B-702 Shreeji Arcade | Lambodar Sfocwares · (null) | Jaro-Winkler on name; `addr_is_null` flag; BGE-m3 bridges typo |
| 3 | Abbreviation reorder | Prairie Capital Partners PLLC · 120 Autumn Woods Blvd | PRAIRIE CAPITAL PLLC-PARTNERS · 120. Autumn Woods Boulevard | Suffix strip (PLLC); token sort; street abbrev expansion |
| 4 | Non-Latin translation (Telugu) | My Construction Private Limited · H.No 1-1-742/3, Gandhi Nagar | మై కన్‌స్ట్రక్షన్ · H. NO 1-1-742/3, GANDHI NAGAR | BGE-m3 only for name; hyphenated numeric token `1-1-742/3` anchors match |
| 5 | Domain-as-name | Cardiology Metro Care Associates Inc · 1138 Trimble Creek Dr | cardiologymetrocare.com · 1138 TRIMBLE CREEK DRIVE | `base_domain` extraction; address similarity rescues low name score |

**Critical finding from pair #4 — A34 emphasis:** the hyphenated Indian house number `1-1-742/3` is identical on both sides and is the *primary match signal* when the name is entirely in Telugu script. Normaliser rule A34 (mixed separators) **must not destroy these hyphenated sequences**. The `addr_numeric_exact_match` feature (§7) is the numerical anchor that catches these pairs; it must preserve the full token including its hyphen-separated components.

#### Memory Math

| Item | Calculation | Result |
|---|---|---|
| BGE-m3 embedding dim | 1024 floats × 4 bytes | 4 KB/record |
| S1 train FAISS index (fp32) | 2,206,821 × 4KB | ~8.5 GB |
| S1 train FAISS index (fp16) | 2,206,821 × 2KB | **~4.25 GB** |
| S1 test FAISS index (fp16) | 1,732,544 × 2KB | **~3.33 GB** |
| All train embeddings (fp16) | 12,527,040 × 2KB | ~24.1 GB |
| All test embeddings (fp16) | 11,702,133 × 2KB | ~22.5 GB |
| ml.g4dn.4xlarge RAM | — | **64 GB** ✅ |
| ml.g4dn.4xlarge VRAM | — | **16 GB** |
| ml.t3.medium RAM | — | **4 GB** ❌ for FAISS |

**Key insight**: ml.g4dn.4xlarge has 64GB RAM — FAISS index builds and searches run in CPU RAM (not VRAM). The 4.25GB fp16 S1 train index fits comfortably; per-country sharding reduces it further (US shard ~2.5GB, India shard ~1.7GB). FAISS IVF search is CPU-side anyway.

#### Embedding Time Math

| Stage | Records | BGE-m3 @ batch=256 on T4 | Time |
|---|---|---|---|
| Train S1+S2+S3 embed | 12,527,040 | ~600 rec/sec | ~5.8 hrs |
| Test S1+S2+S3 embed | 11,702,133 | ~600 rec/sec | ~5.4 hrs |
| **Total** | **24,229,173** | | **~11.2 hrs** |
| **Cost @ $1.505/hr** | | | **~$16.87** |

**Mitigation — Batch embedding strategy**:
- Split into 2 Training Jobs: Job 1 = train files, Job 2 = test files
- Each job runs in parallel if credits allow, or sequentially (~11 hrs total GPU time)
- Use `batch_size=256` and `torch.no_grad()` + fp16 throughout
- Save as `.npy` shards of 500K records each → S3
- Download to g4dn for FAISS build (64GB RAM fits all S1 fp16 vectors)

#### FAISS Strategy at Scale

```
Index type: IndexIVFFlat (approximate) with nlist=4096
  → Faster search than IndexFlatIP
  → Recall@20 ≈ 95%+ with nprobe=64
  → Build on g4dn.4xlarge CPU RAM (64GB — fits fp16 S1 index)

Per-country sharding:
  → Build separate FAISS index per canonical country
  → S1_US index:     1,323,633 train vectors  (~2.5 GB fp16) |  663,106 test vectors (~1.3 GB fp16)
  → S1_India index:    883,188 train vectors  (~1.7 GB fp16) |  809,986 test vectors (~1.6 GB fp16)
  → S1_France index:   0 train vectors        (ZERO-SHOT)    |  259,452 test vectors (~0.5 GB fp16)
  → S1_Unknown:        any remainder
  → Per-country sharding keeps each index well under RAM limits
  NOTE: France index is built from test S1 embeddings only — no training labels exist for it.
        The pipeline must generalise from US/India training signal alone.
```

#### Revised Compute Allocation

```
TASK                              WHERE                   COST
──────────────────────────────────────────────────────────────
Data exploration + EDA            Notebook ml.t3.medium   FREE
Normaliser dev + unit tests       Notebook ml.t3.medium   FREE
TF-IDF blocking build             Notebook ml.t3.medium   FREE (sparse, CPU)
indic-transliteration step        Notebook ml.t3.medium   FREE
LightGBM training                 Notebook ml.t3.medium   FREE (CPU-friendly)
validate_submission.py            Notebook ml.t3.medium   FREE
──────────────────────────────────────────────────────────────
BGE-m3 embed (train 12.5M)        Training Job g4dn.4x    ~$8.73 (5.8hr)
BGE-m3 embed (test 11.7M)         Training Job g4dn.4x    ~$8.14 (5.4hr)
FAISS index build + search        Training Job g4dn.4x    ~$1.50 (1hr, CPU RAM)
Reranker fine-tuning              Training Job g4dn.4x    ~$4.52 (3hr, GPU)
──────────────────────────────────────────────────────────────
TOTAL GPU COST ESTIMATE                                    ~$22.89
REMAINING FROM $200 CREDITS                                ~$177
```

**Important**: FAISS index build happens on the g4dn.4xlarge Training Job (not Notebook) because it needs 64GB RAM. The Notebook (ml.t3.medium, 4GB) cannot hold even the fp16 S1 index.

---

## 3. BLOCKING ARCHITECTURE (Locked — Scale-Revised)

### Primary: BGE-m3 Dense Retrieval via FAISS

```
All records S1, S2, S3 (~26.4M total)
    → pre_clean() + country normaliser (Notebook, CPU)
    → BGE-m3 encode on g4dn.4xlarge Training Job (fp16, batch_size=256)
        Job 1: train_S1 + train_S2 + train_S3  → S1_train_emb.npy shards
        Job 2: test_S1  + test_S2  + test_S3   → S1_test_emb.npy shards
        All saved to S3 as 500K-record .npy shards
    → FAISS build on g4dn.4xlarge Training Job (64GB RAM — fits fp16 index)
        Index type: IndexIVFFlat, nlist=4096, nprobe=64
        Per-country shards: S1_US / S1_India / S1_France / S1_Unknown
        For each S2/S3 record: top-20 S1 candidates within same country
    → TF-IDF supplement on Notebook (sparse, CPU, with indic-transliteration)
    → Union of BGE-m3 top-20 + TF-IDF top-20 per record
    → Measure blocking recall on train validation split
    → Add token/soundex keys only if recall < 95%
    → Output: candidate_pairs.tsv
```

### Supplementary Blocking Keys (Iterative)

| Round | Key | Purpose |
|---|---|---|
| 1 | BGE-m3 FAISS top-20 | Cross-script + semantic (MANDATORY for Indian scripts) |
| 1 | TF-IDF cosine top-20 | Same-script lexical similarity |
| 2 | First 2 sorted name tokens | Word-order robustness |
| 2 | All numeric address tokens | Address number anchoring |
| 3 | Soundex of first name token | Phonetic / typo |
| 3 | Character trigrams of core name | Sub-word similarity |

**Add rounds 2 and 3 iteratively only if blocking recall < 95% on training validation split.**

---

## 4. EXHAUSTIVE DIRTY-DATA CATALOGUE

### 4A — Name Field (26 patterns)

| # | Pattern | Example | Handler |
|---|---|---|---|
| N1 | ALL CAPS | `FOUNDATION EXCEL AGENCY` | `.lower()` |
| N2 | Cross-script Indian | `குளோபல் பிசினஸ்` (Tamil) | BGE-m3 only |
| N3 | Mixed script | `Sun पावर Provision` | BGE-m3 + normalise each |
| N4 | Duplicate tokens | `Crestline Crestline Clean` | Deduplicate adjacent identical |
| N5 | Bracket legal suffix | `Clm Agro [Limited]`, `[INCORPORATED]` | Strip `[]`, treat content as suffix |
| N6 | Leading non-alpha | `*** Yamuica Venus`, `<< Team Ecole`, `#central` | Strip `^[^a-zA-Z0-9\u0900-\uFFFF]+` |
| N7 | OCR digit-for-letter | `TRADIN6` (6→G), `5uperior` (5→S) | BGE-m3 semantic; Jaro-Winkler |
| N8 | Domain-as-name | `heassociates.com`, `prprivate.com` | Extract base_domain + tld feature |
| N9 | DBA format | `Ectolumdrex dba X+ Madison Inc` | Split on `\bdba\b|d/b/a|t/a` |
| N10 | French diacritics present/absent | `Àmicale` vs `Amicale` | NFC; keep for BGE-m3, strip for TF-IDF |
| N11 | Extra spaces | `Fresh  Truist` | `re.sub(r'\s+', ' ')` |
| N12 | Trailing punctuation | `Best Software Innovations,` | `rstrip('.,;:!')` |
| N13 | Trademark symbols | `Company® Name™` | Strip `[®™©]` |
| N14 | Parenthetical former name | `(Formerly ABC Ltd)` | Extract as alt_name, try matching |
| N15 | Repeated legal suffix | `Ltd Ltd`, `Pvt Pvt Ltd` | Deduplicate suffix list |
| N16 | HTML entities | `A &amp; B` | `html.unescape()` |
| N17 | Zero-width / invisible chars | U+200B zero-width space | Strip Unicode category `Cf` |
| N18 | Ordinal casing | `1St Cross`, `2Nd Floor` | `re.sub(r'(\d+)(st\|nd\|rd\|th)\b', r'\1', IGNORECASE)` |
| N19 | Number words | `One Vision` vs `1 Vision` | Normalise 1–10 number words |
| N20 | Common word abbreviations | `Mktg`, `Mgmt`, `Intl`, `Nat'l`, `Assoc` | Add to hybrid abbreviation dict |
| N21 | URL-style domain name | `erwinsignaturekrakacquisition.com` | Domain handler (N8) |
| N22 | Phone / fax in name | `ABC Co. Tel: 555-1234` | Strip `(Tel\|Fax\|Ph)\s*:?\s*[\d\-\+]{7,}` |
| N23 | Emoji / symbols | `★ ABC Company` | Strip non-printable non-script |
| N24 | Devanagari digits | `४` (4), `७` (7) | Translate to ASCII via `str.translate` |
| N25 | Registered domain prefix | `www.company.com` | Strip `www.`; then N8 |
| N26 | Industry noise suffixes | `& Sons`, `& Co`, `& Partners` | Add to strippable suffix set |

### 4B — Address Field (37 patterns)

| # | Pattern | Example | Handler |
|---|---|---|---|
| A1 | NaN / None | `NaN` | → `""`, set `addr_is_null=1` |
| A2 | Literal "NULL" | `067 CT, NULL, KY` | Remove `\bNULL\b` token |
| A3 | ALL CAPS (S2) | `8219 17RD STREET` | `.lower()` |
| A4 | Component reorder | `OH, Columbus, 5559 Ave` | Token sort representation |
| A5 | Double-hash | `##2326 341ND PLACE` | `re.sub(r'#{2,}', '')` |
| A6 | PO Box | `PO Box 6009, Cincinnati` | Strip `P\.?O\.?\s*Box\s*\d+` |
| A7 | C/O prefix | `C/O Manoj Kumar` | Strip `^C/O\s+` |
| A8 | Landmark downweight | `Near Amani`, `OPP. MUKHI HOSPITAL` | `has_landmark` flag + 40% penalty |
| A9 | Fraction address | `19 1/2 STARDUST TRAIL` | `(\d+)\s*/\s*(\d+)` → keep first |
| A10 | Ordinal floors | `2Nd Floor`, `3Rd Floor` | Normalise ordinal suffix → number |
| A11 | Unit variations | `Unit BUILDING 3030`, `Unit APARTMENT G` | `unit\s+(building\|apartment\|apt\|ste)?\s*` → `unit ` |
| A12 | Floor notations | `F-0`, `Fl 0`, `Fl.`, `2F`, `2nd Flr` | Normalise all → `floor N` |
| A13 | Slash house numbers (India) | `12/A`, `63/2275/7` | `(\d+)/([A-Za-z])` → `\1\2` |
| A14 | PIN code (India 6-digit) | `110015` | Strip `\b\d{6}\b` in IndiaNormalizer |
| A15 | ZIP code (US 5/9-digit) | `28570`, `28570-1234` | Strip `\b\d{5}(-\d{4})?\b` in USNormalizer |
| A16 | Postal code (France 5-digit) | `75001` | Strip `\b\d{5}\b` in FranceNormalizer |
| A17 | State embedded (US) | `IL, Rockton, 211 Dewey Rd` | Expand state abbrevs; token sort |
| A18 | Cedex (France) | `TOURCOING Cedex`, `Paris Cedex 13` | Strip `\bCedex\s*\d*\b` |
| A19 | Arrondissement (France) | `Paris 1er`, `Lyon 13ème` | `(\w+)\s+(\d+)(e[rm]e?\|er)?\b` → `(\w+) \2` |
| A20 | Devanagari in address | `New Delhi, हरयाना` | BGE-m3 handles; script detect |
| A21 | State in Indian script | `महाराष्ट्र`, `তামিল নাড়ু` | IndiaNormalizer; BGE-m3 |
| A22 | Empty comma segments | `,,`, `NULL,NULL,KY` | Strip empty/null segments |
| A23 | URL in address | `www.biz.com, 123 Main St` | Strip URL pattern from address |
| A24 | Phone in address | `123 Main St, Ph: 9876543210` | Strip `(Ph\|Tel\|Mobile)\s*:?\s*[\d]{7,}` |
| A25 | Landmark: ATM/Bank | `Near SBI ATM`, `Opp HDFC Bank` | `has_bank_atm_landmark=1` |
| A26 | Road number format | `Road No.10`, `Road #5` | Normalise → `road 10` |
| A27 | Survey/Gat/Plot (India) | `Gat No 34`, `S No. 42/2/3` | Keep numeric part as key token |
| A28 | Door No (South India) | `Door No 183`, `D No. 20-6-3` | Normalise → `door 183` |
| A29 | House No variations | `H.No.16-11-23`, `H NO-B3`, `HN 247` | Extract first number → `hno N` |
| A30 | KH No (land record) | `KH NO. -570/13`, `KH. NO. 48/19` | Strip `KH\s*NO\s*[-.]?\s*` → keep number |
| A31 | Sector/Phase/Block | `Sector 42`, `Phase-10`, `Block A` | Keep; important address component |
| A32 | PMB (Private Mailbox) | `PMB 2592`, `PMB 9302` | Strip `PMB\s*\d+` |
| A33 | Parenthetical old address | `(Old: 123 Main St)` | Strip parenthetical pattern |
| A34 | Mixed separators | `1-2-3/A Street` vs `1 2 3 A Street` | Normalise `-`, `/` between components — **CRITICAL: do NOT destroy hyphenated Indian house numbers like `1-1-742/3`; these are the primary match signal for cross-script pairs (noise audit pair #4)** |
| A35 | Nagar/Colony/Vihar | `Rajendra Nagar`, `Satnam Nagar` | Keep; valid area component |
| A36 | Building-first vs number-first | `ABC Tower, Floor 3, 123 St` vs `123 St, ABC Tower, Fl3` | Token sort handles |
| A37 | C-Road / B-Road style (India) | `C-road`, `B/H NATRAJ CINEMA` | Keep; normalise separator |

### 4C — Country Field (7 patterns)

| # | Pattern | Handler |
|---|---|---|
| C1 | US / USA / United States | `canonical_country()` dict |
| C2 | India / IN / Bharat / Inde | `canonical_country()` dict |
| C3 | France / FR / République française | `canonical_country()` dict |
| C4 | Typo ("Frnace") | Levenshtein ≤ 2 fuzzy fallback |
| C5 | Case variation ("INDIA") | `.strip().lower()` |
| C6 | Trailing period ("U.S.") | Strip punctuation before lookup |
| C7 | Unseen country | → `GenericNormalizer()` |

---

## 5. DOMAIN NAME FEATURE ENGINEERING (Q20 — Locked)

```python
KNOWN_TLDS = {
    '.com': 'generic', '.net': 'generic', '.org': 'generic',
    '.edu': 'us',      '.gov': 'us',      '.mil': 'us',
    '.in': 'india',    '.co.in': 'india',
    '.fr': 'france',
    '.us': 'us',       '.io': 'generic',  '.co': 'generic',
}

def process_name_domain(name: str) -> dict:
    clean = name.strip().lower()
    is_domain = False
    base_domain = clean
    tld = ""
    tld_country_signal = "unknown"

    if ' ' not in clean:
        for known_tld in sorted(KNOWN_TLDS.keys(), key=len, reverse=True):
            if clean.endswith(known_tld):
                is_domain = True
                base_domain = re.sub(r'^www\.', '', clean[: -len(known_tld)])
                tld = known_tld
                tld_country_signal = KNOWN_TLDS[known_tld]
                break

    return {
        "is_domain":           int(is_domain),   # binary flag feature
        "base_domain":         base_domain,        # used as name_normalized
        "tld":                 tld,                # ".in" / ".fr" etc.
        "tld_country_signal":  tld_country_signal, # weak country hint
    }
```

**Features generated**:
- `name_is_domain` — LightGBM learns to upweight BGE-m3 for domain-name records
- `domain_tld_match` — 1 if TLD-implied country = canonical_country (consistency signal)
- `domain_base_jaro` — Jaro-Winkler on base_domain pair (when both are domains)

---

## 6. COMPLETE NORMALISATION FLOW

```
raw_name
  → html.unescape()
  → strip Unicode Cf (zero-width) chars
  → strip leading non-alphanumeric noise (***, <<, ##, --)
  → rstrip trailing punctuation  
  → process_name_domain()          → is_domain, base_domain, tld
  → split_dba()                    → legal_name, trade_name
  → deduplicate_adjacent_tokens()  → "Crestline Crestline" → "Crestline"
  → bracket_handler()              → [INCORPORATED] → INCORPORATED
  → normalise_ordinals()           → 1St → 1, 2Nd → 2
  → normalise_devanagari_digits()  → ४ → 4
  → html_entity_decode()           → &amp; → &
  → collapse_whitespace()
  → country_normaliser.normalize_name()
        → expand abbreviations (hybrid dict)
        → strip legal suffixes   → name_core
        → sorted token set       → name_token_set
  OUTPUTS: name_normalized, name_core, name_token_set, name_trade, is_domain, tld

raw_address
  → handle NaN/None                → "" + addr_is_null=1
  → html.unescape() + Cf strip
  → remove NULL literal
  → remove PO Box
  → remove C/O prefix
  → remove PMB
  → remove phone in address
  → remove URL in address
  → normalise ## → strip
  → normalise ordinals (2Nd → 2)
  → normalise unit variants        → Unit APARTMENT G → unit g
  → normalise floor variants       → 2nd Flr → floor 2
  → strip_postal_codes()           → country-specific (ZIP/PIN/FR postal)
  → normalise_fraction_numbers()   → 19 1/2 → 19
  → normalise_slash_house()        → 12/A → 12a
  → country_normaliser.normalize_address()
        → expand street abbrevs
        → city transliteration  (India)
        → cedex strip           (France)
        → landmark detection    → has_landmark flag
        → sorted token set      → addr_tokens
  OUTPUTS: addr_normalized, addr_tokens, addr_numeric_tokens,
           has_landmark, addr_is_null, has_bank_atm_landmark
```

---

## 7. FEATURE SET (30 features per candidate pair)

### Name Features

| Feature | Method | Cross-script? |
|---|---|---|
| `bgem3_name_cosine` | BGE-m3 cosine on name only | ✅ |
| `bgem3_full_cosine` | BGE-m3 cosine on name+addr | ✅ PRIMARY |
| `name_jaro_winkler` | Jaro-Winkler on name_normalized | ❌ (0 if cross-script) |
| `name_levenshtein_ratio` | Normalised edit distance | ❌ |
| `name_token_jaccard` | Jaccard on word tokens | ❌ |
| `name_token_sort_ratio` | Sorted token ratio | ❌ |
| `name_char_trigram_jaccard` | Char-3gram Jaccard | ❌ |
| `core_name_jaro_winkler` | Jaro-Winkler on name_core | ❌ |
| `core_name_token_jaccard` | Jaccard on name_core tokens | ❌ |
| `name_tfidf_cosine` | Country-level TF-IDF cosine | ❌ |
| `name_is_domain_both` | Both names are domains | ✅ |
| `domain_base_jaro` | Jaro on base_domain (when domains) | ✅ |
| `tld_match` | Same TLD | ✅ |
| `max_name_jaro` | max(legal_name_jaro, trade_name_jaro) | ❌ |
| `name_dedup_ratio` | Unique/total tokens (noise signal) | ✅ |

### Address Features

| Feature | Method |
|---|---|
| `addr_jaro_winkler` | Jaro-Winkler on addr_normalized |
| `addr_token_jaccard` | Jaccard on addr_tokens |
| `addr_token_sort_ratio` | Token sort ratio |
| `addr_numeric_overlap` | Fraction of shared numeric tokens |
| `addr_tfidf_cosine` | Country TF-IDF cosine on address |
| `addr_landmark_penalised_jaccard` | Jaccard × (1 - 0.4 × landmark_ratio) |
| `addr_is_null_either` | 1 if either record has null address |
| `addr_is_null_both` | 1 if both null (rely on name only) |
| `addr_numeric_exact_match` | 1 if ALL numeric tokens identical |
| `addr_city_token_match` | Exact match on largest non-numeric token |

### Global Features

| Feature | Source |
|---|---|
| `bgem3_full_cosine` | FAISS score (free from blocking) |
| `faiss_rank` | Position in FAISS top-20 |
| `tfidf_rank` | Position in TF-IDF top-20 |
| `script_mismatch` | 1 if dominant Unicode scripts differ |
| `country_canonical_match` | 1 if same canonical country |
| `source_pair` | 0=S1→S2, 1=S1→S3 |
| `tld_country_signal_match` | TLD-implied country = canonical country |

---

## 8. TWO-STAGE MODEL

```
Candidates
    ↓
[STAGE 1 — LightGBM] (run in Notebook, CPU)
  30 features, anchor = bgem3_full_cosine
  P_lgbm per pair
  P < 0.60  → AUTO-REJECT
  P > 0.90  → AUTO-ACCEPT
  0.60–0.90 → STAGE 2

    ↓ (uncertain pairs only)

[STAGE 2 — bge-reranker-v2-m3] (fine-tuned, loaded in Notebook)
  Input: "Business: {name1} Address: {addr1} [SEP]
          Business: {name2} Address: {addr2}"
  Fine-tuned on: positives (ground truth) + hard negatives
  Negative ratio: 80% same-FAISS-bucket hard (cosine > 0.7, not in GT)
                  20% random within-country easy
  Pair ratio: 1 positive : 3 negatives (mix per above)
  Training: AWS g4dn.xlarge, fp16, batch_size=16
  → P_ce per pair
  P_ce > T_ce → MATCH  (T_ce tuned on validation F_0.5)
```

---

## 9. VALIDATION STRATEGY (Triple — Locked)

```python
# Eval 1: Stratified 80/20 by country → overall F_0.5
train_ids, val_ids = train_test_split(s1_ids, test_size=0.20,
                                       stratify=country_labels, random_state=42)

# Eval 2: US-train → India-validate → France generalisation proxy
# If F_0.5(India | US-trained) ≥ 0.80 → confident France will work

# Eval 3: Full 100% training data → test set inference
```

**Critical eval harness note — singleton denominator:**
Blocking recall must be computed against **matchable S1 entities only (2,083,574)**, not total S1 (2,206,821). The 123,247 singletons (5.58%) have no ground-truth match and can never contribute to recall. Using the wrong denominator will make the pipeline appear ~5.6% below the true recall ceiling and mask genuine threshold issues. This applies to Eval 1, 2, and 3.

```python
# Correct blocking recall denominator
matchable_s1_ids = set(ground_truth["s1_entity_id"].unique())  # 2,083,574
blocking_recall  = len(gt_pairs_recovered) / len(matchable_s1_ids)
```

---

## 10. ALL DECISIONS LOCKED ✅

All questions from Rounds 1–9 are now answered. No open questions remain.

| Q | Decision |
|---|---|
| Q26 | Dataset **24.2M records total** (confirmed by exploration): train 12,527,040 · test 11,702,133 · singleton ratio 5.58% (123,247 unmatchable S1) · France zero-shot (1,694,445 test-only records) · S1 train 100% Latin, S1 test 40,789 accented-Latin French (risk closed) · S2/S3 ~15% non-Latin · ~4% domain names · ~3.3% null addresses (all in S2/S3) |
| Q27 | Hard negatives: 80% same-FAISS-bucket (cosine>0.7, not GT) + 20% random within-country |
| Q29 | Add `indic-transliteration` before TF-IDF — converts Indian scripts → Roman |
| Q30 | Keep BGE-m3 as primary blocker on ml.g4dn.4xlarge (64GB RAM handles FAISS) |
| Q31 | Instance: **ml.g4dn.4xlarge** — 1×T4 16GB VRAM, 16 vCPUs, 64GB RAM, **$1.505/hr** |
| Q32 | Match cardinality: median=3, mean=3.46, p95=6, p99=8, max=11 per S1 entity · **K=20 confirmed safe** (9 spare slots above max) · 1:3 positive-to-negative ratio confirmed tractable at median (~9 hard negatives per entity) |

**Total estimated GPU cost: ~$22.89 from $200 credits. Remaining: ~$177.**

---

## 11. DEPENDENCY LIST

```txt
# Core ML
lightgbm>=4.0
faiss-cpu>=1.7.4
sentence-transformers>=2.7
transformers>=4.40
torch>=2.1
optimum[onnxruntime]>=1.18

# String similarity
rapidfuzz>=3.0

# Data
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3

# Text cleaning
ftfy>=6.1
indic-transliteration>=2.3   # LOCKED Q29

# AWS (SageMaker notebook pre-installed)
boto3>=1.34
sagemaker>=2.200
```

---

## 12. IMPLEMENTATION CHECKLIST

### Phase 0 — Setup ⬜
- [ ] Create SageMaker Notebook Instance `ml.t3.medium` in `us-east-1`
- [ ] Create S3 bucket `er-challenge-<name>` in `us-east-1`
- [ ] Upload all 6 TSV files to S3
- [ ] Install dependencies in notebook: `!pip install lightgbm rapidfuzz faiss-cpu sentence-transformers ftfy`

### Phase 1 — Data Exploration ✅ COMPLETE
- [x] Load all 6 TSVs, count records per file
- [x] Count: country distribution, singleton ratio, non-Latin script records
- [x] Count: NULL/NaN address records, domain-name records
- [x] Sample 30 matched pairs from ground truth, visual noise audit
- [x] **Q26 answered** — full findings in ER_Pipeline___Data_Exploration_Findings_(Q26).md
- [x] **NEW** Investigate script of the 40,789 non-Latin test_source1 records → **CONFIRMED French accented-Latin; risk closed**

### Phase 2 — Normaliser + Pre-cleaner ⬜
- [ ] Implement all handlers: N1–N26 (names), A1–A37 (addresses), C1–C7 (country)
- [ ] Implement domain feature engineering
- [ ] Implement CountryNormalizer hierarchy (US/India/France/Generic)
- [ ] Implement canonical_country() + fuzzy fallback
- [ ] Unit tests on all sample records from data exploration
- [ ] **NEW — A34 critical test**: confirm rule preserves `1-1-742/3`, `12/A`, `63/2275/7` intact (noise audit pair #4 — sole cross-script match signal)

### Phase 3 — BGE-m3 Embeddings (AWS ml.g4dn.4xlarge Training Job) ⬜
- [ ] Write `train_embeddings.py` script — SageMaker Training Job entry point
      - batch_size=256, fp16, torch.no_grad()
      - Save as 500K-record .npy shards to /opt/ml/output → S3 auto-upload
- [ ] Job 1: Embed train_S1 + train_S2 + train_S3 (12,527,040 records, ~5.8 hrs, ~$8.73)
- [ ] Job 2: Embed test_S1 + test_S2 + test_S3 (11,702,133 records, ~5.4 hrs, ~$8.14)
- [ ] Verify all shards uploaded to S3

### Phase 4 — FAISS Build + Blocking (AWS ml.g4dn.4xlarge Training Job) ⬜
- [ ] Write `build_faiss_and_block.py` — runs on g4dn.4xlarge (64GB RAM for FAISS)
      - Load all S1 embeddings → build IndexIVFFlat (nlist=4096, fp16)
      - Per-country shards: US / India / France / Unknown
      - For each S2/S3 record: query same-country index → top-20 candidates
      - Save candidate pairs → S3
- [ ] Add TF-IDF supplement on Notebook:
      - Apply indic-transliteration on Indian-script tokens before TF-IDF
      - TF-IDF top-20 per S2/S3 record
      - Union with BGE-m3 candidates
- [ ] Measure blocking recall on train validation split (20% holdout) — **denominator = 2,083,574 matchable S1 only (not 2,206,821 total)**
- [ ] Add token/Soundex keys only if recall < 95% (on matchable S1 denominator)
- [ ] Generate candidate_pairs.tsv

### Phase 5 — Feature Engineering ⬜
- [ ] Compute all 30 features per candidate pair
- [ ] Label pairs (1=match, 0=non-match) from ground truth
- [ ] Audit feature distributions (NaN check, scale check)

### Phase 6 — LightGBM Stage 1 ⬜
- [ ] Train on 80% split
- [ ] Sweep threshold 0.30–0.95 → max validation F_0.5
- [ ] Confirm or adjust T_low=0.60, T_high=0.90
- [ ] Collect uncertain pairs (0.60–0.90) for reranker

### Phase 7 — Reranker Fine-tuning (AWS g4dn.xlarge Training Job) ⬜
- [ ] Write `train_reranker.py` (positive + hard negative pairs, binary cross-entropy)
- [ ] Submit Training Job → ml.g4dn.xlarge (~$1–3, takes 1–3 hr)
- [ ] Download fine-tuned weights to notebook
- [ ] Evaluate standalone reranker F_0.5
- [ ] Evaluate full pipeline F_0.5

### Phase 8 — Test Inference ⬜
- [ ] Run pipeline on test set (US + India + France)
- [ ] Generate matching_results.tsv, candidate_pairs.tsv
- [ ] Run `python utils/validate_submission.py` → PASS
- [ ] Upload matching_results.tsv to leaderboard

### Phase 9 — Triple Validation ⬜
- [ ] Eval 1: 80/20 stratified split F_0.5
- [ ] Eval 2: US-train → India-validate F_0.5 (France proxy)
- [ ] If Eval 2 < 0.70 → fix pipeline; if ≥ 0.80 → France likely OK

---

## 13. RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cross-script recall = 0 without BGE-m3 | CERTAIN | CRITICAL | BGE-m3 FAISS is primary — addressed |
| S3 free tier (5GB) exceeded by embeddings | MEDIUM | LOW | $200 credits cover storage overage |
| g4dn.xlarge Training Job OOM | LOW | MEDIUM | fp16 + batch_size=64; 570M × fp16 = ~1.1GB weights |
| Reranker fine-tuning time > budget | LOW | MEDIUM | g4dn.xlarge T4 fast; estimated 1–3 hr = $0.74–$2.22 |
| France recall low (no training data) | MEDIUM | HIGH | BGE-m3 multilingual; Eval 2 proxy measures this |
| NULL addresses causing false no-match | MEDIUM | MEDIUM | addr_is_null feature; both-null → name-only mode |
| DBA trade names not caught | HIGH | HIGH | Dual name matching; max() across legal/trade Jaro |
| Hard negatives too easy → weak reranker | MEDIUM | HIGH | Same-FAISS-bucket hard negatives + random easy mix |
| Ordinal casing (2Nd Floor) in address | LOW | LOW | Normaliser handles — addressed |
| Bills exceeding $200 credits | VERY LOW | HIGH | Total GPU budget ~$23; well within $200; monitor billing |
| Embedding jobs take >12hrs total | MEDIUM | HIGH | Run Job1+Job2 sequentially; 72hr window has room; start immediately |
| FAISS build OOM on g4dn.4xlarge | LOW | HIGH | 64GB RAM; fp16 S1 index = ~4.25GB; per-country sharding reduces further |
| S3 shard download bottleneck | LOW | MEDIUM | Keep FAISS build on same Training Job; avoid large S3→Notebook transfers |
| indic-transliteration coverage gap | LOW | LOW | BGE-m3 is primary; TF-IDF is supplement; gap has low impact |
| **NEW** Singleton ceiling misread as tuning problem | MEDIUM | MEDIUM | Always measure blocking recall on 2,083,574 matchable S1 only; see §9 note |
| **NEW** A34 destroying hyphenated Indian house numbers | MEDIUM | HIGH | Rule must preserve `1-1-742/3` style tokens — these are sole match signal for cross-script pairs; add unit test |
| ~~test_source1 non-Latin (40,789) — unknown script~~ | ~~LOW~~ | ~~MEDIUM~~ | **CLOSED** — confirmed French accented-Latin (é, â, è, ô) from sample inspection; not Indian script; no pipeline change needed |
| **NEW** Test US/India ratio shift (60/40 train → 44/54 test) | LOW | LOW | India heavier in test; calibrate threshold on test-realistic held-out split rather than train distribution |
| **NEW** candidate_pairs.tsv size (~150–180M rows) | LOW | MEDIUM | LightGBM handles this on Notebook CPU; feature engineering must be vectorised (no Python loops); consider chunked processing |

---

*Document updated: Round 9. All questions locked. Q32 (match cardinality) confirmed — K=20 safe, 1:3 pair ratio tractable, max 11 matches per S1. test_source1 non-Latin confirmed French accented-Latin (risk register closed). No open investigative items remain. Pipeline is fully specified and ready for implementation.*
