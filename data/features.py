"""Loading, cleaning and feature engineering for the IEEE-CIS Fraud Detection data.

Three feature groups are produced so that each teammate can own one and test it
against the same partitions:

    txn  transaction-level features (amount, product, card type, C/D/M columns, time of day)
    beh  account-takeover style behavioural features, computed causally from each
         account's own past transactions (velocity, new address/email/device, ...)
    idn  identity / device features (id_xx, DeviceType, DeviceInfo, e-mail domains)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
CARD_COLS = [f"card{i}" for i in range(1, 7)]
ID_NUM = [f"id_{i:02d}" for i in range(1, 12)]
ID_CAT = ["id_12", "id_15", "id_16", "id_28", "id_29", "id_31",
          "id_35", "id_36", "id_37", "id_38", "DeviceType", "DeviceInfo"]
EMAILS = ["P_emaildomain", "R_emaildomain"]

REQUIRED = ["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "card1"]
TXN_BASE = (["TransactionID", "isFraud", "TransactionDT", "TransactionAmt", "ProductCD"]
            + CARD_COLS + ["addr1", "addr2", "dist1", "dist2"] + EMAILS + C_COLS + D_COLS + M_COLS)

# columns that are heavy-tailed and non-negative -> signed log1p before scaling
LOG_COLS = {"TransactionAmt", "dist1", "dist2", *C_COLS, *D_COLS,
            "beh_prior_count", "beh_secs_since_prev", "beh_cnt_1h", "beh_cnt_24h", "beh_cnt_7d"}

BEH_NUM = ["beh_prior_count", "beh_secs_since_prev", "beh_amt_logratio",
           "beh_cnt_1h", "beh_cnt_24h", "beh_cnt_7d"]
BEH_FLAG = ["beh_is_first", "beh_new_addr", "beh_new_email", "beh_new_device"]
WINDOWS = {"beh_cnt_1h": 3600.0, "beh_cnt_24h": 86400.0, "beh_cnt_7d": 7 * 86400.0}


def make_groups(v_cols: list[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Feature-group specification: which columns are numeric / 0-1 flags / categorical."""
    txn_num = (["TransactionAmt", "dist1", "dist2"] + C_COLS + D_COLS
               + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"] + list(v_cols or []))
    return {
        "txn": {"num": txn_num, "flag": [], "cat": ["ProductCD", "card4", "card6"] + M_COLS},
        "beh": {"num": BEH_NUM, "flag": BEH_FLAG, "cat": []},
        "idn": {"num": ID_NUM, "flag": ["has_identity", "email_mismatch"], "cat": ID_CAT + EMAILS},
    }


# --------------------------------------------------------------------------- loading
def load_ieee(data_dir: str | Path, include_v: bool = False) -> pd.DataFrame:
    """Read train_transaction.csv (+ train_identity.csv) and left-join them.

    Only the labelled *train* files are used: the Kaggle test files have no labels.
    """
    d = Path(data_dir)
    tx_path, id_path = d / "train_transaction.csv", d / "train_identity.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"{tx_path} not found. Download the IEEE-CIS competition data "
                                "(train_transaction.csv, train_identity.csv) into this folder.")
    header = pd.read_csv(tx_path, nrows=0).columns
    want = [c for c in TXN_BASE if c in header]
    v_cols: list[str] = []
    if include_v:
        v_cols = [c for c in header if c.startswith("V") and c[1:].isdigit()]
        want += v_cols
    absent = [c for c in REQUIRED if c not in want]
    if absent:
        raise ValueError(f"train_transaction.csv is missing required columns: {absent}")
    tx = pd.read_csv(tx_path, usecols=want)

    idn = None
    if id_path.exists():
        idn = pd.read_csv(id_path)
        idn.columns = [c.replace("-", "_") for c in idn.columns]  # test files use id-01
        keep = ["TransactionID"] + [c for c in ID_NUM + ID_CAT if c in idn.columns]
        idn = idn[keep].drop_duplicates("TransactionID")

    df = tx if idn is None else tx.merge(idn, on="TransactionID", how="left")
    df["has_identity"] = (tx["TransactionID"].isin(idn["TransactionID"]).to_numpy()
                          if idn is not None else False)
    for c in TXN_BASE + ID_NUM + ID_CAT:  # tolerate absent optional columns
        if c not in df.columns:
            df[c] = np.nan
    df.attrs["v_cols"] = v_cols
    return df


def load_ulb(csv_path: str | Path) -> pd.DataFrame:
    """ULB creditcard.csv -> the same column names the rest of the pipeline expects."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"Time": "TransactionDT", "Amount": "TransactionAmt", "Class": "isFraud"})
    df.insert(0, "TransactionID", np.arange(len(df)))
    return df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- cleaning
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = (df.drop_duplicates("TransactionID")
            .sort_values(["TransactionDT", "TransactionID"], kind="stable")
            .reset_index(drop=True))
    df["isFraud"] = df["isFraud"].astype("int8")
    hour = (df["TransactionDT"] // 3600) % 24
    dow = (df["TransactionDT"] // 86400) % 7
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    if "P_emaildomain" in df:
        p, r = df["P_emaildomain"], df["R_emaildomain"]
        df["email_mismatch"] = (p.notna() & r.notna() & (p != r)).astype("int8")
    return df


# --------------------------------------------------------------------------- behavioural
def _window_counts(ts: np.ndarray, codes: np.ndarray) -> dict[str, np.ndarray]:
    """Number of *earlier* transactions of the same account inside each time window."""
    out = {k: np.zeros(len(ts)) for k in WINDOWS}
    order = np.argsort(codes, kind="stable")  # stable => time order kept inside an account
    sc = codes[order]
    bounds = np.flatnonzero(np.diff(sc)) + 1
    for s, e in zip(np.r_[0, bounds], np.r_[bounds, len(order)]):
        idx = order[s:e]
        t = ts[idx]
        pos = np.arange(len(t))
        for name, w in WINDOWS.items():
            out[name][idx] = pos - np.searchsorted(t, t - w, side="left")
    return out


def _first_seen(df: pd.DataFrame, acct: np.ndarray, col: str, prior: np.ndarray) -> np.ndarray:
    """1 if this value of `col` has never been seen on this account before (and the account has history)."""
    # Compared only against earlier *observed* values: identity columns are mostly empty, so a
    # card's first recorded device is not "new" - only a different one later is.
    key = df[col].astype("string")
    valid = key.notna().to_numpy()
    tmp = pd.DataFrame({"a": acct, "k": key.fillna("<NA>").to_numpy(), "v": valid.astype(int)})
    first = ~tmp.duplicated(["a", "k"], keep="first").to_numpy()
    prior_valid = tmp.groupby("a")["v"].cumsum().to_numpy() - valid.astype(int)
    return (first & valid & (prior_valid > 0)).astype("int8")


def add_behavioral(df: pd.DataFrame) -> pd.DataFrame:
    """Causal per-account features. `df` MUST be sorted by TransactionDT (clean() does this).

    An account is the full tuple card1..card6. Every feature at row t only uses rows before t,
    so nothing from the future leaks in. Because the account key contains card1 and banks are
    assigned per card1, an account's whole history always lives inside a single bank.
    """
    ts = df["TransactionDT"].to_numpy(float)
    amt = df["TransactionAmt"].to_numpy(float)
    acct = df.groupby(CARD_COLS, dropna=False, sort=False).ngroup().to_numpy()
    s_ts, s_amt = pd.Series(ts), pd.Series(amt)

    prior = s_ts.groupby(acct).cumcount().to_numpy()
    prev_ts = s_ts.groupby(acct).shift(1).to_numpy()
    cum = s_amt.groupby(acct).cumsum().to_numpy()
    prev_mean = np.where(prior > 0, (cum - amt) / np.maximum(prior, 1), np.nan)
    ratio = np.where(prior > 0, amt / np.maximum(prev_mean, 1e-6), 1.0)

    out = df.copy()
    out["beh_prior_count"] = prior.astype(float)
    out["beh_secs_since_prev"] = np.where(prior > 0, ts - prev_ts, 0.0)
    out["beh_amt_logratio"] = np.log(np.clip(ratio, 1e-3, 1e3))
    for name, vals in _window_counts(ts, acct).items():
        out[name] = vals
    out["beh_is_first"] = (prior == 0).astype("int8")
    out["beh_new_addr"] = _first_seen(df, acct, "addr1", prior)
    out["beh_new_email"] = _first_seen(df, acct, "P_emaildomain", prior)
    out["beh_new_device"] = _first_seen(df, acct, "DeviceInfo", prior)
    out["meta_account"] = acct
    return out


# --------------------------------------------------------------------------- encoding
def _numeric(df: pd.DataFrame, c: str) -> pd.Series:
    x = pd.to_numeric(df[c], errors="coerce").astype(float)
    if c in LOG_COLS:
        x = np.sign(x) * np.log1p(np.abs(x))
    return x


class _F32(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, np.asarray(value, dtype=np.float32))


class FeatureEncoder:
    """Fit on the global TRAIN split only, then apply to every split and every bank.

    Output is a fixed-width float32 matrix: numeric columns are median-imputed and standardised
    (with a missing-indicator when 5-95% of train values are missing); categoricals are one-hot
    over the top-k train values plus `__other` / `=<missing>`. Column order is stored in
    `self.columns`, so every bank and every teammate sees exactly the same feature space.
    """

    def __init__(self, groups: dict, top_k: int = 12, clip: float = 10.0,
                 miss_range: tuple[float, float] = (0.05, 0.95)):
        self.groups, self.top_k, self.clip, self.miss_range = groups, top_k, clip, miss_range
        self.num: dict[str, dict] = {}
        self.cat: dict[str, dict] = {}
        self.columns: list[str] = []
        self.column_group: dict[str, str] = {}

    def _add(self, g: str, name: str):
        self.columns.append(name)
        self.column_group[name] = g

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        lo, hi = self.miss_range
        for g, spec in self.groups.items():
            for c in spec["num"]:
                x = _numeric(df, c)
                miss = float(x.isna().mean())
                med = float(x.median()) if x.notna().any() else 0.0
                f = x.fillna(med)
                sd = float(f.std(ddof=0))
                ind = lo < miss < hi
                self.num[f"{g}|{c}"] = dict(median=med, mean=float(f.mean()),
                                            std=sd if sd > 0 else 1.0, indicator=ind, missing_rate=miss)
                self._add(g, f"{g}__{c}")
                if ind:
                    self._add(g, f"{g}__{c}__missing")
            for c in spec["flag"]:
                self._add(g, f"{g}__{c}")
            for c in spec["cat"]:
                s = df[c].astype("string")
                vc = s.value_counts()
                vocab = [str(v) for v in vc.index[: self.top_k]]
                has_other, has_missing = len(vc) > self.top_k, bool(s.isna().any())
                self.cat[f"{g}|{c}"] = dict(vocab=vocab, other=has_other, missing=has_missing)
                for v in vocab:
                    self._add(g, f"{g}__{c}={v}")
                if has_other:
                    self._add(g, f"{g}__{c}=<other>")
                if has_missing:
                    self._add(g, f"{g}__{c}=<missing>")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = _F32()  # casts every column to float32 on assignment (keeps peak memory down)
        for g, spec in self.groups.items():
            for c in spec["num"]:
                p = self.num[f"{g}|{c}"]
                x = _numeric(df, c)
                z = ((x.fillna(p["median"]) - p["mean"]) / p["std"]).clip(-self.clip, self.clip)
                out[f"{g}__{c}"] = z.to_numpy()
                if p["indicator"]:
                    out[f"{g}__{c}__missing"] = x.isna().to_numpy(float)
            for c in spec["flag"]:
                out[f"{g}__{c}"] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(float)
            for c in spec["cat"]:
                p = self.cat[f"{g}|{c}"]
                s = df[c].astype("string")
                known = np.zeros(len(df), dtype=bool)
                for v in p["vocab"]:
                    m = (s == v).fillna(False).to_numpy(dtype=bool)
                    out[f"{g}__{c}={v}"] = m.astype(float)
                    known |= m
                if p["other"]:
                    out[f"{g}__{c}=<other>"] = ((~known) & s.notna().to_numpy(dtype=bool)).astype(float)
                if p["missing"]:
                    out[f"{g}__{c}=<missing>"] = s.isna().to_numpy(float)
        return pd.DataFrame(out, index=df.index)[self.columns].astype("float32")

    def to_dict(self) -> dict:
        return dict(top_k=self.top_k, clip=self.clip, miss_range=list(self.miss_range),
                    groups=self.groups, numeric=self.num, categorical=self.cat, columns=self.columns)
