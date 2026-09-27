"""
Business Entity Resolution — end-to-end pipeline.

Reproduces: normalization -> blocking -> candidate generation ->
feature engineering -> Logistic Regression training -> threshold
selection (macro F0.5) -> test scoring -> output files.

Usage:
    python pipeline.py --data-dir /path/to/dataset --output-dir output
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
MAX_BLOCK_SIZE = 5
VAL_FRACTION = 0.02
VAL_CAP = 30_000
N_POS_SAMPLE = 100_000
N_NEG_SAMPLE = 200_000
THRESHOLD_GRID = np.arange(0.10, 0.96, 0.05)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

def normalize_basic(series: pd.Series) -> pd.Series:
    series = (
        series.fillna("")
        .astype("string")
        .str.lower()
        .str.replace("&", " and ", regex=False)
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.strip()
    )
    return series


def normalize_name(series: pd.Series) -> pd.Series:
    s = normalize_basic(series)
    # Strip common legal suffixes only when they appear at the end of the name
    s = s.str.replace(
        r"\b(incorporated|inc|llc|ltd|limited|corp|corporation|company|co|pvt|plc)\s*$",
        "",
        regex=True,
    )
    return s.str.strip()


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["name_norm"] = normalize_name(df["business_name"])
    df["address_norm"] = normalize_basic(df["business_address"])
    df["country_norm"] = normalize_basic(df["country"])
    return df


# --------------------------------------------------------------------------
# Blocking + candidate generation
# --------------------------------------------------------------------------

def add_block_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["name_key"] = df["country_norm"].fillna("") + "|" + df["name_norm"].fillna("")
    df["name_prefix_key"] = (
        df["country_norm"].fillna("") + "|" + df["name_norm"].fillna("").str[:4]
    )
    df["address_key"] = df["country_norm"].fillna("") + "|" + df["address_norm"].fillna("")
    return df


def make_candidates(left: pd.DataFrame, right: pd.DataFrame, key: str,
                     max_block_size: int = MAX_BLOCK_SIZE) -> pd.DataFrame:
    """Inner-join left (Source 1) to right (Source 2+3) on `key`, only through
    blocks whose size is within [1, max_block_size] on the right side."""
    right_counts = right[key].value_counts()
    allowed = right_counts[(right_counts > 0) & (right_counts <= max_block_size)].index

    left_small = left[left[key].isin(allowed) & left[key].ne("")][["entity_id", key]]
    right_small = right[right[key].isin(allowed) & right[key].ne("")][["entity_id", key]]

    pairs = left_small.merge(right_small, on=key, how="inner")
    pairs = pairs.rename(
        columns={"entity_id_x": "source1_entity_id", "entity_id_y": "candidate_entity_id"}
    )
    return pairs[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()


def generate_candidates(s1: pd.DataFrame, s23: pd.DataFrame) -> pd.DataFrame:
    name_candidates = make_candidates(s1, s23, "name_key")
    prefix_candidates = make_candidates(s1, s23, "name_prefix_key")
    address_candidates = make_candidates(s1, s23, "address_key")
    return pd.concat(
        [name_candidates, prefix_candidates, address_candidates], ignore_index=True
    ).drop_duplicates()


# --------------------------------------------------------------------------
# Feature engineering (8 features)
# --------------------------------------------------------------------------

def make_features(candidates: pd.DataFrame, left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    cols = ["entity_id", "name_norm", "address_norm", "country_norm"]

    x = candidates.merge(
        left[cols], left_on="source1_entity_id", right_on="entity_id", how="left"
    ).drop(columns=["entity_id"])

    x = x.merge(
        right[cols], left_on="candidate_entity_id", right_on="entity_id",
        how="left", suffixes=("_1", "_2"),
    ).drop(columns=["entity_id"])

    features = pd.DataFrame(index=x.index)

    name1, name2 = x["name_norm_1"].fillna(""), x["name_norm_2"].fillna("")
    addr1, addr2 = x["address_norm_1"].fillna(""), x["address_norm_2"].fillna("")
    country1, country2 = x["country_norm_1"].fillna(""), x["country_norm_2"].fillna("")

    features["country_match"] = (country1 == country2).astype("int8")
    features["name_exact"] = (name1 == name2).astype("int8")
    features["address_exact"] = (addr1 == addr2).astype("int8")
    features["name_prefix_match"] = (name1.str[:4] == name2.str[:4]).astype("int8")
    features["name_suffix_match"] = (name1.str[-4:] == name2.str[-4:]).astype("int8")
    features["address_prefix_match"] = (addr1.str[:8] == addr2.str[:8]).astype("int8")

    n1, n2 = name1.str.len(), name2.str.len()
    denom = np.maximum(n1.to_numpy(), n2.to_numpy())
    numer = np.minimum(n1.to_numpy(), n2.to_numpy())
    features["name_length_ratio"] = np.where(denom > 0, numer / denom, 0)

    a1, a2 = addr1.str.len(), addr2.str.len()
    denom = np.maximum(a1.to_numpy(), a2.to_numpy())
    numer = np.minimum(a1.to_numpy(), a2.to_numpy())
    features["address_length_ratio"] = np.where(denom > 0, numer / denom, 0)

    return features


# --------------------------------------------------------------------------
# Ground truth expansion + F0.5
# --------------------------------------------------------------------------

def explode_ground_truth(ground_truth: pd.DataFrame) -> pd.DataFrame:
    gt = ground_truth.copy()
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("").str.split(",")
    gt_pairs = gt.explode("matched_entity_ids", ignore_index=True)
    gt_pairs["matched_entity_ids"] = gt_pairs["matched_entity_ids"].fillna("").str.strip()
    gt_pairs = gt_pairs[gt_pairs["matched_entity_ids"].ne("")]
    gt_pairs = gt_pairs.rename(columns={"matched_entity_ids": "candidate_entity_id"})
    return gt_pairs[["source1_entity_id", "candidate_entity_id"]].drop_duplicates()


def f05(precision: float, recall: float) -> float:
    beta2 = 0.25
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


def select_best_threshold(val_scored: pd.DataFrame, val_gt: pd.DataFrame, val_ids: set):
    true_map = {sid: set(g["candidate_entity_id"]) for sid, g in val_gt.groupby("source1_entity_id")}

    best_threshold, best_score = 0.50, -1.0
    for threshold in THRESHOLD_GRID:
        selected = val_scored[val_scored["probability"] >= threshold]
        pred_map = {sid: set(g["candidate_entity_id"]) for sid, g in selected.groupby("source1_entity_id")}

        scores = []
        for sid in val_ids:
            true_set = true_map.get(sid, set())
            pred_set = pred_map.get(sid, set())
            if not true_set and not pred_set:
                scores.append(1.0)
                continue
            if not true_set or not pred_set:
                scores.append(0.0)
                continue
            tp = len(true_set & pred_set)
            precision = tp / len(pred_set)
            recall = tp / len(true_set)
            scores.append(f05(precision, recall))

        score = float(np.mean(scores))
        print(f"Threshold: {round(threshold, 2)}  F0.5: {round(score, 4)}")
        if score > best_score:
            best_score, best_threshold = score, threshold

    return best_threshold, best_score


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def load_source(data_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, name), sep="\t", dtype="string")


def group_to_output(df: pd.DataFrame, s1: pd.DataFrame, out_col: str) -> pd.DataFrame:
    grouped = (
        df.groupby("source1_entity_id")["candidate_entity_id"]
        .agg(lambda ids: ",".join(sorted(set(ids))))
        .reset_index()
        .rename(columns={"candidate_entity_id": out_col})
    )
    out = s1[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
    out = out.merge(grouped, on="source1_entity_id", how="left")
    out[out_col] = out[out_col].fillna("")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Directory with train/ and test/ TSVs")
    parser.add_argument("--output-dir", default="output", help="Where to write output TSVs")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")
    train_dir = train_dir if os.path.isdir(train_dir) else args.data_dir
    test_dir = test_dir if os.path.isdir(test_dir) else args.data_dir

    # --- Load + normalize training data ---
    train_s1 = add_normalized_columns(load_source(train_dir, "train_source1.tsv"))
    train_s2 = add_normalized_columns(load_source(train_dir, "train_source2.tsv"))
    train_s3 = add_normalized_columns(load_source(train_dir, "train_source3.tsv"))
    train_s23 = pd.concat([train_s2, train_s3], ignore_index=True)

    train_s1 = add_block_keys(train_s1)
    train_s23 = add_block_keys(train_s23)

    # --- Candidate generation + labeling ---
    train_candidates = generate_candidates(train_s1, train_s23)
    ground_truth = load_source(train_dir, "train_ground_truth.tsv")
    gt_pairs = explode_ground_truth(ground_truth)

    gt_small = gt_pairs[gt_pairs["source1_entity_id"].isin(train_candidates["source1_entity_id"].unique())]
    train_labeled = train_candidates.merge(
        gt_small.assign(label=1), on=["source1_entity_id", "candidate_entity_id"], how="left"
    )
    train_labeled["label"] = train_labeled["label"].fillna(0).astype("int8")

    # --- Train / validation split ---
    rng = np.random.default_rng(RANDOM_STATE)
    all_ids = train_s1["entity_id"].dropna().unique()
    val_size = min(VAL_CAP, int(len(all_ids) * VAL_FRACTION))
    val_ids = set(rng.choice(all_ids, size=val_size, replace=False))

    train_part = train_labeled[~train_labeled["source1_entity_id"].isin(val_ids)]
    val_part = train_labeled[train_labeled["source1_entity_id"].isin(val_ids)]

    # --- Balanced training sample ---
    positive = train_part[train_part["label"] == 1]
    negative = train_part[train_part["label"] == 0]
    n_pos, n_neg = min(N_POS_SAMPLE, len(positive)), min(N_NEG_SAMPLE, len(negative))
    ml_train = pd.concat(
        [positive.sample(n=n_pos, random_state=RANDOM_STATE),
         negative.sample(n=n_neg, random_state=RANDOM_STATE)],
        ignore_index=True,
    ).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    # --- Train Logistic Regression ---
    X_train = make_features(ml_train[["source1_entity_id", "candidate_entity_id"]], train_s1, train_s23)
    y_train = ml_train["label"]
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("Model trained.")

    # --- Validation scoring + threshold selection ---
    X_val = make_features(val_part[["source1_entity_id", "candidate_entity_id"]], train_s1, train_s23)
    val_scored = val_part[["source1_entity_id", "candidate_entity_id"]].copy()
    val_scored["probability"] = model.predict_proba(X_val)[:, 1]

    val_gt = gt_pairs[gt_pairs["source1_entity_id"].isin(val_ids)]
    best_threshold, best_score = select_best_threshold(val_scored, val_gt, val_ids)
    print(f"\nBest threshold: {best_threshold}  Best validation F0.5: {best_score}")

    # --- Load + normalize test data ---
    test_s1 = add_normalized_columns(load_source(test_dir, "test_source1.tsv"))
    test_s2 = add_normalized_columns(load_source(test_dir, "test_source2.tsv"))
    test_s3 = add_normalized_columns(load_source(test_dir, "test_source3.tsv"))
    test_s23 = pd.concat([test_s2, test_s3], ignore_index=True)

    test_s1 = add_block_keys(test_s1)
    test_s23 = add_block_keys(test_s23)

    # --- Test candidate generation + scoring ---
    test_candidates = generate_candidates(test_s1, test_s23)
    X_test = make_features(test_candidates[["source1_entity_id", "candidate_entity_id"]], test_s1, test_s23)
    test_candidates["probability"] = model.predict_proba(X_test)[:, 1]

    test_matches = test_candidates[test_candidates["probability"] >= best_threshold].copy()

    # --- Write outputs ---
    os.makedirs(args.output_dir, exist_ok=True)
    candidate_output = group_to_output(test_candidates, test_s1, "candidate_entity_ids")
    matching_output = group_to_output(test_matches, test_s1, "matched_entity_ids")

    matching_output.to_csv(os.path.join(args.output_dir, "matching_results.tsv"), sep="\t", index=False)
    candidate_output.to_csv(os.path.join(args.output_dir, "candidate_pairs.tsv"), sep="\t", index=False)

    print(f"\nWrote {len(matching_output)} rows to matching_results.tsv")
    print(f"Wrote {len(candidate_output)} rows to candidate_pairs.tsv")
    print(f"Duplicate Source1 IDs in matching output: {matching_output['source1_entity_id'].duplicated().sum()}")


if __name__ == "__main__":
    main()
