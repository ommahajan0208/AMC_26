# ER Pipeline — Data Exploration Findings (Q26)

Sep 25, 2026 · @Sam

All six TSV files analysed. S1 is exceptionally clean (0 non-Latin, 0 NULL addresses); S2/S3 carry nearly all noise. France appears only in test.

## Country Distribution

Train contains only US and India; France is a zero-shot generalisation target present only in test.

| File | US | India | France | Total |
| --- | --- | --- | --- | --- |
| train\_source1.tsv | 1,323,633 | 883,188 | — | 2,206,821 |
| train\_source2.tsv | 3,016,817 | 2,017,799 | — | 5,034,616 |
| train\_source3.tsv | 3,170,056 | 2,115,547 | — | 5,285,603 |
| **Train total** | **7,510,506 (63%)** | **5,016,534 (40%)** | **—** | **12,527,040** |
| test\_source1.tsv | 663,106 | 809,986 | 259,452 | 1,732,544 |
| test\_source2.tsv | 1,871,330 | 2,312,565 | 703,378 | 4,887,273 |
| test\_source3.tsv | 1,945,701 | 2,405,000 | 731,615 | 5,082,316 |
| **Test total** | **4,480,137 (40%)** | **5,527,551 (50%)** | **1,694,445 (15%)** | **11,702,133** |

US/India ratio is \~60/40 in train and \~44/54 in test (India is slightly heavier in test). France accounts for \~15% of test records — roughly 1.7M entities that the model has never seen a labelled example for.

## Singleton Analysis

5.58% of S1 entities (123,247 of 2,206,821) have no match in S2 or S3. These are true singletons — the pipeline can never recover them, so they act as a precision floor rather than a recall problem. At F0.5 (precision-weighted 2×), leaving singletons unmatched is correct behaviour; attempting to force-match them risks false positives that hurt the score more than the misses.

| Metric | Value |
| --- | --- |
| Total S1 entities (train) | 2,206,821 |
| Singletons (no GT match) | 123,247 |
| Singleton ratio | 5.58% |
| Matchable S1 entities | 2,083,574 (94.4%) |

**Pipeline implication:** the 5.58% ceiling is not a bug — do not tune thresholds trying to recover these. The blocking recall target of ≥ 95% should be measured against matchable S1 entities only.

## Non-Latin Script Records

S1 (train) is completely clean — zero non-Latin characters. S1 (test) has a small fraction. S2 and S3 carry the vast majority of script noise.

| File | Non-Latin records | % of file |
| --- | --- | --- |
| train\_source1.tsv | 0 | 0.0% |
| test\_source1.tsv | 40,789 | 2.4% |
| train\_source2.tsv | 764,608 | 15.2% |
| test\_source2.tsv | 928,158 | 19.0% |
| train\_source3.tsv | 606,737 | 11.5% |
| test\_source3.tsv | 737,515 | 14.5% |

**S1-clean finding — critical implication:** ground-truth labels are anchored on Latin-script S1 names. Every cross-script match (e.g. Telugu → English, pair #4 in the noise audit) requires the model to bridge from a non-Latin S2/S3 query to a clean Latin S1 reference. TF-IDF will score 0 on these pairs; BGE-m3 FAISS is the *only* mechanism that can retrieve them. This validates Q7/Q12/Q18 and means blocking recall on Indian-script records depends entirely on embedding quality — there is no lexical fallback.

## NULL / NaN Address Records

S1 has zero null addresses in both train and test — consistent with S1 being the cleaned reference source. All null-address noise originates in S2 and S3.

| File | NULL addresses | % of file |
| --- | --- | --- |
| train\_source1.tsv | 0 | 0.0% |
| test\_source1.tsv | 0 | 0.0% |
| train\_source2.tsv | 168,967 | 3.4% |
| test\_source2.tsv | 129,408 | 2.6% |
| train\_source3.tsv | 175,916 | 3.3% |
| test\_source3.tsv | 136,098 | 2.7% |

\~3–3.4% of S2/S3 records have no address. When an S2/S3 query has a null address and its S1 target has a valid one, all address features score 0 — the LightGBM model must learn to rely on name-only features for these pairs. The `addr_is_null_either` feature (strategy §7) directly flags this condition. Null-address pairs are also the hardest cases for the reranker, since its input text loses the "Address:" segment entirely.

## Domain-Name Records

S1 is nearly domain-free (4 train, 13 test). Domain names are almost exclusively a S2/S3 noise pattern — around 4% of those files.

| File | Domain-name records | % of file |
| --- | --- | --- |
| train\_source1.tsv | 4 | <0.01% |
| test\_source1.tsv | 13 | <0.01% |
| train\_source2.tsv | 201,286 | 4.0% |
| test\_source2.tsv | 156,069 | 3.2% |
| train\_source3.tsv | 210,942 | 4.0% |
| test\_source3.tsv | 164,040 | 3.2% |

Noise audit pair #5 (cardiologymetrocare.com → Cardiology Metro Care Associates Inc) confirms the canonical case: the address matches precisely, which means `bgem3_full_cosine` will rank it high via the address component even if the name component scores low. The `name_is_domain_both` and `domain_base_jaro` features (strategy §7) are needed to handle the rarer case where *both* sides are domains.

## Visual Noise Audit (30-pair sample)

Five representative pairs from the audit, each demonstrating a distinct noise class:

| # | Pattern class | S1 (reference) | S2/S3 (noisy) | Handler |
| --- | --- | --- | --- | --- |
| 1 | Word transposition + punctuation | Sarasva India Limited · 9/1/3 Kasundia 2Nd Bye Lane… | limited sarasva india · 9/1/3. KASUNDIA 2ND BYE LANE… | Token-sort name; addr token-sort; ordinal normalisation |
| 2 | OCR / typo + null address | Lambodar Softwares · B-702, Shreeji Arcade… | Lambodar Sfocwares · (null) | Jaro-Winkler on name; addr\_is\_null flag; BGE-m3 bridges typo |
| 3 | Abbreviation reorder | Prairie Capital Partners PLLC · 120 Autumn Woods Blvd, Mount Holly NC | PRAIRIE CAPITAL PLLC-PARTNERS · 120. Autumn Woods Boulevard, MOUNT HOLLY NC | Suffix strip (PLLC); token sort; street abbrev expansion |
| 4 | Non-Latin translation (Telugu) | My Construction Private Limited · H. No 1-1-742/3, Gandhi Nagar, Hyderabad | మై కన్‌స్ట్రక్షన్ ప్రైవేట్ లిమిటెడ్ · H. NO 1-1-742/3… తెలంగాణ | BGE-m3 only for name; address numeric tokens anchor match |
| 5 | Domain-as-name | Cardiology Metro Care Associates Inc · 1138 Trimble Creek Dr, West Jordan City UT | cardiologymetrocare.com · 1138 TRIMBLE CREEK DRIVE, WEST JORDAN UT | base\_domain extraction; address match rescues low name score |

**Key observation from pair #4:** the address carries the match — the numeric token `1-1-742/3` is identical on both sides even though the name is entirely in Telugu script. This is exactly why `addr_numeric_exact_match` is in the feature set and why address normalisation must preserve these hyphenated sequences.

## Pipeline Implications

What these numbers confirm, change, or flag as new risks vs ER\_Strategy.md:

| Item | Status | Detail |
| --- | --- | --- |
| BGE-m3 as sole cross-script blocker | **Confirmed critical** | S1 is 100% Latin in train; 15% of S2/S3 train records are non-Latin. TF-IDF recall on those pairs is 0 — there is no fallback. |
| France as zero-shot target | **Confirmed** | 0 French records in train; \~1.69M in test. Eval 2 cross-country proxy (US→India) is the only signal before submission. |
| Singleton floor | **Confirmed** | 5.58% of S1 can never be matched. Do not tune recall on these. Measure blocking recall on matchable S1 only (2,083,574 entities). |
| Null-address pairs | **Confirmed, quantified** | \~3.3% of S2/S3 records. LightGBM must handle name-only feature vectors for these. Consider oversampling null-address pairs in reranker training data. |
| Domain-name skew | **Confirmed, quantified** | 4% of S2/S3 vs <0.01% of S1. The domain→canonical match relies on address similarity as the rescue signal (pair #5 audit). |
| Numeric token preservation in addresses | **New emphasis** | Pair #4 shows that hyphenated Indian house numbers (1-1-742/3) are the primary match signal when the name is in a non-Latin script. Normaliser A34 (mixed separators) must NOT destroy these sequences. |
| Hard negative ratio Q27 (80% FAISS-bucket) | **Confirmed appropriate** | High non-Latin rate in S2/S3 means many hard negatives will be cross-script near-misses — exactly the distribution the ratio was designed for. |
| S1 test non-Latin (40,789 records) | **New — not in strategy doc** | train\_source1 has 0 non-Latin but test\_source1 has 40,789. These are S1 reference records in test with non-Latin names. The FAISS index built from test S1 embeddings must cover these; they will not have a Latin-script fallback on the S1 side either. Flag for post-submission analysis if France F0.5 is low. |
