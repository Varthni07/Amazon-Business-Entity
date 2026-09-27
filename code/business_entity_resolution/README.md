# Business Entity Resolution — `Code Orbit_submission`

**Authors:** Jayasamvarthni V, Madhu Parameswari Ganesan, Kavinila N

Matches business records from **Source 2** and **Source 3** against the deduplicated
**Source 1** reference set, across noisy, multi country business names and addresses
(US, India, and an unseen France test split).

**Pipeline:** Normalize → Block on 3 keys → Generate candidates → Engineer 8 pairwise
features → Logistic Regression → Threshold-tune on F0.5 → Score test → Emit
`matching_results.tsv` + `candidate_pairs.tsv`.

See `Documentation_template.md` (in the parent submission zip) for the full methodology
write up and pipeline diagram.

## Results at a glance

| | Train | Test |
|---|---|---|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| Candidate pairs generated | 4,862,994 | 3,807,242 |

- **Model:** Logistic Regression (`class_weight="balanced"`, `max_iter=500`) — 8 features, well within the MIT/Apache-2.0, ≤8B-parameter constraint.
- **Best threshold:** 0.55 (selected by sweeping 0.10–0.95 on a held-out validation split)
- **Validation F0.5:** 0.8229
- **Predicted matching pairs (test):** 1,010,833
- **Test Source 1 entities with no predicted match (singletons):** 1,068,894 / 1,732,544

## Repository layout

```
business_entity_resolution/
├── src/
│   └── pipeline.py          # end-to-end: load → normalize → block → feature → train → score → write outputs
├── README.md                 # this file
├── Documentation_template.md  # filled-in methodology write-up + pipeline diagram
├── pipeline_diagram.png       # referenced by the methodology doc
└── requirements.txt
```

> The reference implementation lives in a single notebook-derived script
> (`pipeline.py`, place it under `src/` in the final submission zip) rather than a
> multi-module package, matching how the solution was actually developed and run.
> Data paths are configurable via the `--data-dir` and `--output-dir` flags below.

## Requirements

```
python >= 3.10
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

## Data layout expected

```
<data-dir>/
├── train_source1.tsv
├── train_source2.tsv
├── train_source3.tsv
├── train_ground_truth.tsv
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv
```

All files are tab-separated; read with `sep="\t"`.

## How to reproduce end-to-end

```bash
python src/pipeline.py \
  --data-dir /path/to/dataset \
  --output-dir output
```

This will, in order:

1. Load all six train/test TSVs (`dtype="string"`, explicit `sep="\t"`).
2. Normalize `business_name` → `name_norm` (lowercased, punctuation stripped,
   `&` → `and`, trailing legal suffixes such as *Inc/LLC/Ltd/Pvt/Corp* removed),
   `business_address` → `address_norm`, and `country` → `country_norm`.
3. Concatenate Source 2 + Source 3 into one candidate pool per split.
4. Build three blocking keys — `country + full name`, `country + name prefix (4 chars)`,
   and `country + full address` — capped at block size 5, and inner-join Source 1
   against the candidate pool on each key to form `candidate_pairs.tsv`.
5. On the training split only, label every candidate pair 1/0 against
   `train_ground_truth.tsv`.
6. Hold out ~2% of Source 1 entities (capped at 30,000) as a validation split;
   draw a class-balanced training sample (up to 100k positives / 200k negatives)
   from the remainder.
7. Engineer 8 pairwise features (see below) and fit the Logistic Regression model.
8. Sweep decision thresholds from 0.10 to 0.95 on the validation split, scoring each
   with the macro-averaged F0.5 metric, and keep the best (0.55 in the reference run).
9. Repeat normalization + blocking + candidate generation on the test split, score
   every test candidate pair with the trained model, and keep pairs at/above the
   chosen threshold.
10. Write `output/matching_results.tsv` and `output/candidate_pairs.tsv`, one row per
    Source 1 test entity in both files (empty match list = predicted singleton).

Validate the outputs before submitting:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir /path/to/dataset/test
```

## Feature set (8 features)

| Feature | Description |
|---|---|
| `country_match` | Exact match on normalized country |
| `name_exact` | Exact match on normalized business name |
| `address_exact` | Exact match on normalized address |
| `name_prefix_match` | First 4 characters of normalized name match |
| `name_suffix_match` | Last 4 characters of normalized name match |
| `address_prefix_match` | First 8 characters of normalized address match |
| `name_length_ratio` | `min(len)/max(len)` of the two normalized names |
| `address_length_ratio` | `min(len)/max(len)` of the two normalized addresses |

## Notes 

- Blocking keys cap block size at 5, trading some recall for tractable candidate volume — `candidate_pairs.tsv` is provided so blocking recall can be audited independently of the final model.
- The France country label appears only in the test set; the pipeline treats
`country` as an open string label throughout (no hard-coded country list), so no retraining or special-casing is needed for unseen countries.
- No external data, APIs, or lookups are used anywhere in the pipeline, per the challenge's fair play rules.
