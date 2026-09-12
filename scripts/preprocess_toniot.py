"""
preprocess_toniot.py — convert the raw ToN_IoT Train_Test_Network sample
into the fully-numeric format the Paper-3 pipeline expects.

Key decisions (documented for the paper's reproducibility):
  * Target  : the multiclass ``type`` column, renamed to ``Label``.
  * LEAKAGE : the binary ``label`` column (0/1 attack flag) is DROPPED —
              it is a direct function of the target and would trivialise
              the task.
  * IDENTIFIERS / free-text columns are dropped (they encode the
              simulation topology or are high-cardinality strings that act
              as near-identifiers): src_ip, dst_ip, dns_query, ssl_subject,
              ssl_issuer, http_uri, http_user_agent, http_orig_mime_types,
              http_resp_mime_types, weird_name, weird_addl, weird_notice.
  * Remaining object columns (proto, service, conn_state, http_method,
              ssl_*, dns flags, ...) are label-encoded (factorised).
  * Numeric columns are kept as-is; ``-`` placeholders coerce to 0.

Output: merged_TONIoT_clean.csv with numeric features + string ``Label``.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "train_test_network.csv")
OUT = os.path.join(HERE, "merged_TONIoT_clean.csv")

DROP_LEAKAGE = ["label"]                       # binary flag == target
DROP_IDENT = [
    "src_ip", "dst_ip",
    "dns_query", "ssl_subject", "ssl_issuer",
    "http_uri", "http_user_agent",
    "http_orig_mime_types", "http_resp_mime_types",
    "weird_name", "weird_addl", "weird_notice",
]


def main():
    df = pd.read_csv(SRC)
    print(f"raw shape: {df.shape}")

    # target
    df = df.rename(columns={"type": "Label"})
    y = df["Label"].astype(str)

    # drop leakage + identifiers + the target from the feature frame
    drop = [c for c in (DROP_LEAKAGE + DROP_IDENT + ["Label"]) if c in df.columns]
    X = df.drop(columns=drop)

    # label-encode every remaining object column; coerce numeric placeholders
    n_cat = 0
    for c in X.columns:
        if X[c].dtype == object:
            # treat '-' and NaN as a distinct category
            X[c] = pd.factorize(X[c].fillna("NA").replace("-", "NA"))[0]
            n_cat += 1
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    print(f"label-encoded {n_cat} categorical columns; "
          f"{X.shape[1]} numeric feature columns total")

    out = X.copy()
    out["Label"] = y.values
    # shuffle so chunked balanced-subset builder sees all classes early
    out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {OUT}  shape={out.shape}")
    print("Label distribution:")
    print(out["Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
