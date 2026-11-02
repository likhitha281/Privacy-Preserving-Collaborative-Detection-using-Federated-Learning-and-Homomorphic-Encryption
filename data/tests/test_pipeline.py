import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from data import features as F
from data import loader
from data.prepare import run
from data.synthetic import generate

SPLITS = ("train", "val", "test")


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    generate(d, n_txn=30000, n_cards=1500, seed=0)
    return d


def build(raw_dir, tmp_path_factory, name, **kw):
    out = tmp_path_factory.mktemp(name)
    args = dict(dataset="ieee", data_dir=raw_dir, out_dir=out, n_banks=3, partition_mode="label_skew",
                alpha=0.5, seed=7, min_fraud_train=20, top_k=8)
    args.update(kw)
    run(**args)
    return out


@pytest.fixture(scope="session")
def fed(raw_dir, tmp_path_factory):
    return build(raw_dir, tmp_path_factory, "fed")


def all_parts(root):
    for b in range(loader.n_banks(root)):
        for s in SPLITS:
            yield b, s, pd.read_parquet(root / f"bank_{b}" / f"{s}.parquet")


def test_hyphenated_identity_columns_are_normalised(raw_dir):
    df = F.load_ieee(raw_dir)
    assert "id_01" in df.columns and not any("-" in c for c in df.columns)


def test_every_bank_and_split_has_identical_columns(fed):
    schema = json.loads((fed / "schema.json").read_text())
    expected = ["label", *schema["meta_columns"], *[c for g in schema["feature_groups"].values() for c in g]]
    for _, _, p in all_parts(fed):
        assert sorted(p.columns) == sorted(expected)


def test_no_card_is_split_across_banks(fed):
    seen = {}
    for b, _, p in all_parts(fed):
        for g in p["meta__group"].unique():
            assert seen.setdefault(g, b) == b, f"card {g} appears in banks {seen[g]} and {b}"


def test_time_split_is_ordered(fed):
    dts = {s: [] for s in SPLITS}
    for _, s, p in all_parts(fed):
        dts[s].append(p["meta__TransactionDT"].to_numpy())
    dts = {s: np.concatenate(v) for s, v in dts.items()}
    assert dts["train"].max() <= dts["val"].min() and dts["val"].max() <= dts["test"].min()


def test_encoder_fit_on_train_only(fed):
    m = json.loads((fed / "manifest.json").read_text())
    assert m["encoder_fit_rows"] == m["split_rows"]["train"]
    X, _ = loader.load_global(fed, "train", ["txn"], as_numpy=False)
    col = X["txn__TransactionAmt"]
    assert abs(col.mean()) < 0.05 and 0.8 < col.std() < 1.05  # standardised on train


def test_features_are_finite(fed):
    cols = loader.feature_columns(fed)
    for _, _, p in all_parts(fed):
        assert np.isfinite(p[cols].to_numpy()).all()


def test_behavioural_features_are_causal(raw_dir):
    df = F.clean(F.load_ieee(raw_dir))
    a = F.add_behavioral(df)
    k = int(len(df) * 0.8)
    df2 = df.copy()
    tail = df2.index[k:]
    df2.loc[tail, "TransactionAmt"] = df2.loc[tail, "TransactionAmt"] * 7 + 3
    df2.loc[tail, "addr1"] = 999.0
    df2.loc[tail, "DeviceInfo"] = "zzz"
    df2.loc[tail, "P_emaildomain"] = "evil.com"
    b = F.add_behavioral(df2)
    cols = [c for c in a.columns if c.startswith("beh_")]
    pd.testing.assert_frame_equal(a.loc[: k - 1, cols], b.loc[: k - 1, cols])  # past unaffected by future
    assert not a.loc[k:, cols].equals(b.loc[k:, cols])  # sanity: the perturbation did matter


def test_first_seen_flags_ignore_unobserved_history():
    df = pd.DataFrame({"TransactionDT": [1, 2, 3, 4], "TransactionAmt": [1.0] * 4,
                       "card1": [1] * 4, "card2": [1] * 4, "card3": [1] * 4, "card4": ["v"] * 4,
                       "card5": [1] * 4, "card6": ["d"] * 4,
                       "addr1": [1.0] * 4, "P_emaildomain": ["a.com"] * 4,
                       "DeviceInfo": [None, "d1", "d1", "d2"]})
    out = F.add_behavioral(df)
    assert out["beh_new_device"].tolist() == [0, 0, 0, 1]  # first observed device is not "new"


def test_label_skew_is_more_skewed_than_iid(raw_dir, tmp_path_factory):
    def spread(mode):
        root = build(raw_dir, tmp_path_factory, mode, partition_mode=mode, alpha=0.3, min_fraud_train=5)
        s = pd.read_csv(root / "partition_summary.csv")
        return s[s.split == "train"]["fraud_rate"].std()
    assert spread("label_skew") > spread("iid")


def test_logistic_regression_sanity_and_group_ablation(fed):
    base = {}
    for name, groups in (("txn", ["txn"]), ("all", None)):
        Xtr, ytr = loader.load_global(fed, "train", groups)
        Xte, yte = loader.load_global(fed, "test", groups)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
        base[name] = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])
    rate = loader.load_global(fed, "test")[1].mean()
    assert base["all"] > 3 * rate
    assert base["all"] > base["txn"]  # the planted behavioural signal is picked up


def test_same_seed_gives_identical_output(raw_dir, tmp_path_factory):
    a = build(raw_dir, tmp_path_factory, "r1")
    b = build(raw_dir, tmp_path_factory, "r2")
    pd.testing.assert_frame_equal(pd.read_csv(a / "partition_summary.csv"), pd.read_csv(b / "partition_summary.csv"))
    pd.testing.assert_frame_equal(pd.read_parquet(a / "bank_0" / "train.parquet"),
                                  pd.read_parquet(b / "bank_0" / "train.parquet"))


def test_loader_group_selection(fed):
    X_txn, y = loader.load_bank(fed, 0, "train", ["txn"])
    X_all, y2 = loader.load_bank(fed, 0, "train")
    assert X_txn.shape[0] == X_all.shape[0] == len(y) == len(y2)
    assert X_txn.shape[1] == len(loader.feature_columns(fed, ["txn"])) < X_all.shape[1]
    with pytest.raises(ValueError):
        loader.feature_columns(fed, ["nope"])


def test_ulb_path(tmp_path):
    rng = np.random.default_rng(1)
    n = 6000
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, n)))
    df["Amount"] = rng.lognormal(3, 1, n)
    df["Class"] = (rng.random(n) < 0.02).astype(int)
    raw = tmp_path / "raw"
    raw.mkdir()
    df.to_csv(raw / "creditcard.csv", index=False)
    m = run("ulb", raw, tmp_path / "out", n_banks=2, partition_mode="iid", min_fraud_train=10)
    assert m["rows"] == n and m["n_features"] == {"txn": 33}


# ----------------------------------------------------------------------------- plan interface
@pytest.fixture
def plan(raw_dir, tmp_path):
    loader.configure(raw_dir=raw_dir, processed_dir=tmp_path)
    yield loader
    loader.configure(raw_dir=loader.HERE / "raw" / "ieee", processed_dir=loader.HERE / "processed")


def test_load_client_data_matches_the_plan_contract(plan):
    for ft in ("credit_card", "account_takeover", "identity_fraud"):
        Xtr, ytr, Xv, yv = plan.load_client_data(0, ft, n_banks=3, alpha=1.0)
        assert Xtr.dtype == np.float32 and Xv.dtype == np.float32
        assert Xtr.shape[0] == len(ytr) > 0 and Xv.shape[0] == len(yv) > 0
        assert Xtr.shape[1] == Xv.shape[1] == plan.input_dim(ft, n_banks=3, alpha=1.0)
        assert set(np.unique(ytr)) <= {0, 1}
    dims = {ft: plan.input_dim(ft, n_banks=3, alpha=1.0) for ft in plan.VIEWS}
    assert len(set(dims.values())) == 3  # the three views have different widths


def test_views_map_onto_existing_feature_groups(plan):
    root = plan.ensure_federation(n_banks=3, alpha=1.0)
    for groups in plan.VIEWS.values():
        assert set(groups) <= set(plan.feature_groups(root))


def test_global_test_set_is_latest_20_percent_and_independent_of_bank_count(plan):
    sizes, pos = [], []
    for nb in (2, 3):
        Xte, yte = plan.load_test("credit_card", n_banks=nb, alpha=1.0)
        sizes.append(len(yte)); pos.append(int(yte.sum()))
        root = plan.ensure_federation(n_banks=nb, alpha=1.0)
        dts = {s: np.concatenate([pd.read_parquet(root / f"bank_{b}" / f"{s}.parquet",
                                                  columns=["meta__TransactionDT"]).to_numpy().ravel()
                                  for b in range(nb)]) for s in SPLITS}
        assert max(dts["train"].max(), dts["val"].max()) <= dts["test"].min()   # test is strictly the latest
        total = sum(len(v) for v in dts.values())
        assert abs(len(dts["test"]) / total - 0.20) < 0.01
    assert sizes[0] == sizes[1] and pos[0] == pos[1]


def test_pos_weight_and_argument_checks(plan):
    w = plan.get_pos_weight(0, "credit_card", n_banks=3, alpha=1.0)
    _, y, _, _ = plan.load_client_data(0, "credit_card", n_banks=3, alpha=1.0)
    assert w == pytest.approx((len(y) - y.sum()) / y.sum())
    with pytest.raises(ValueError):
        plan.load_client_data(0, "not_a_fraud_type", n_banks=3, alpha=1.0)
    with pytest.raises(ValueError):
        plan.load_client_data(3, "credit_card", n_banks=3, alpha=1.0)


def test_split_report_is_written(fed):
    text = (fed / "split_report.md").read_text()
    assert "Class balance per bank" in text and "global test 20%" in text
