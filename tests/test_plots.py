"""make_plots and reproduce on a tiny hand-made results tree (no IEEE-CIS data, no training)."""
import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from experiments import reproduce  # noqa: E402
from experiments.plots import make_plots as mp  # noqa: E402
from fl.metrics import ROW_KEYS  # noqa: E402

PR = {"local": 0.30, "fedavg": 0.40, "fedavg_he": 0.402}  # final-round PR-AUC before the seed offset
SEED_OFF = {42: 0.0, 43: 0.02}
ROUNDS = 3


def run_rows(ft, mode, n, alpha, seed):
    rows = []
    for r in range(1, ROUNDS + 1):
        pr = PR[mode] + SEED_OFF[seed] - 0.01 * (ROUNDS - r) - (0.05 if alpha == 0.1 else 0.0)
        row = {k: None for k in ROW_KEYS}
        row.update(mode=mode, fraud_type=ft, n_clients=n, alpha=alpha, seed=seed, round=r, test_pr_auc=pr,
                   test_f1_tuned=pr + 0.05, test_f1=pr - 0.1, test_roc_auc=0.8, test_precision=0.3,
                   test_recall=0.5, test_accuracy=0.95, test_loss=0.4, train_s=2.0,
                   round_wall_s=10.0 if mode == "fedavg" else 10.2,
                   encrypt_s=0.05 if mode == "fedavg_he" else (0.0 if mode == "fedavg" else None),
                   decrypt_s=0.02 if mode == "fedavg_he" else (0.0 if mode == "fedavg" else None),
                   aggregate_s=0.03 if mode == "fedavg_he" else (0.001 if mode == "fedavg" else None),
                   uplink_bytes_per_client={"local": 0, "fedavg": 1000, "fedavg_he": 20000}[mode],
                   downlink_bytes={"local": 0, "fedavg": 1000, "fedavg_he": 20000}[mode],
                   max_weight_err=1e-8 if mode == "fedavg_he" else None)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(ROW_KEYS))


def write_fixture(root: Path, attacks: bool = True) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43):
        for n in (3, 5):
            for ft in ("credit_card", "account_takeover"):
                for mode in ("local", "fedavg", "fedavg_he"):
                    run_rows(ft, mode, n, 0.5, seed).to_csv(
                        root / f"{ft}_{mode}_b{n}_a0.5_s{seed}.csv", index=False)
            for alpha in (0.1, 5.0):
                for mode in ("local", "fedavg", "fedavg_he"):
                    run_rows("credit_card", mode, n, alpha, seed).to_csv(
                        root / f"credit_card_{mode}_b{n}_a{alpha}_s{seed}.csv", index=False)
    (root / "summary_main.csv").write_text("not,a,run\n1,2,3\n")  # must be ignored
    (root / "garbage.csv").write_bytes(b"\x00\x01 partial,\n\"unterminated")  # e.g. a file being written
    if not attacks:
        return
    a = root / "attacks"
    a.mkdir()
    flip = []
    for mode in ("fedavg", "fedavg_he"):
        for f, (pr, tuned, f05) in {0.0: (0.40, 0.42, 0.23), 0.25: (0.38, 0.41, 0.07),
                                    1.0: (0.05, 0.07, 0.07)}.items():
            for r in (1, 2):
                flip.append(dict(mode=mode, fraud_type="credit_card", n_clients=3, alpha=0.5, seed=42,
                                 flip_frac=f, round=r, test_pr_auc=pr, test_f1_tuned=tuned, test_f1=f05,
                                 test_roc_auc=0.8))
    pd.DataFrame(flip).to_csv(a / "label_flip.csv", index=False)
    pd.DataFrame([{"x": 1}]).to_csv(a / "label_flip_summary.csv", index=False)  # must be ignored
    leak = []
    for mode, corr in (("fedavg", {1: (0.6, 0.8), 16: (0.5, 0.5)}), ("fedavg_he", {1: (0.4, 0.4), 16: (0.3, 0.3)})):
        for b, rs in corr.items():
            for t, rc in enumerate(rs):
                leak.append(dict(mode=mode, batch_size=b, local_steps=1, recon_nmse=0.5, recon_corr=rc,
                                 baseline_nmse=0.7, baseline_corr={1: 0.45, 16: 0.55}[b], attack_iters=10, seed=t))
        leak.append(dict(mode=mode, batch_size=512, local_steps=400, recon_nmse=0.5, recon_corr=0.6,
                         baseline_nmse=0.3, baseline_corr=0.7, attack_iters=10, seed=42))
    pd.DataFrame(leak).to_csv(a / "gradient_leakage.csv", index=False)


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        out = fn(*a, **kw)
    return out, buf.getvalue()


class TestMakePlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name) / "results"
        write_fixture(cls.root)
        cls.out = cls.root / "figures"
        cls.figs, cls.log = quiet(mp.make_plots, cls.root, cls.out)
        cls.index = (cls.out / "index.md").read_text()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_all_figures_rendered_as_png_and_pdf(self):
        names = ["main_comparison", "convergence_b3", "convergence_b5", "he_cost", "noniid", "label_flip",
                 "gradient_leakage"]
        self.assertEqual([f.name for f in self.figs], names)
        for f in self.figs:
            self.assertIsNone(f.skipped, f.name)
        for n in names:
            for ext in ("png", "pdf"):
                p = self.out / f"{n}.{ext}"
                self.assertTrue(p.exists() and p.stat().st_size > 1000, p)
        self.assertIn("skipping unreadable garbage.csv", self.log)

    def test_index_numbers_match_fixture(self):
        # main comparison: fedavg beats local on both fraud types; |he - fedavg| = 0.002
        self.assertIn("3 banks: FedAvg beats local-only PR-AUC on 2/2 fraud types; max |FedAvg+CKKS - FedAvg| "
                      "PR-AUC gap 0.002", self.index)
        # convergence: per-round seed means differ by 0.002 in every round
        self.assertIn("max PR-AUC difference 0.0020", self.index)
        # cost: crypto = 0.05 + 0.02 + 0.03 = 0.10 s; ratio 20000/1000; overhead 0.10 / 10.0 s = 1.0%
        self.assertIn("encrypt+decrypt+aggregate takes 0.100-0.100 s per round", self.index)
        self.assertIn("ciphertext uploads are 20-20x the plaintext size", self.index)
        self.assertIn("crypto adds 1.0-1.0% of a FedAvg round", self.index)
        # non-IID: local final PR-AUC over seeds (0.30, 0.32) -> 0.31, alpha 0.1 is 0.05 lower -> 0.26
        self.assertIn("Local only PR-AUC 0.260-0.310", self.index)
        # label flip: f=0.25 drops 0.16 / 0.01 / 0.02; f=1 PR-AUC drop 0.35 in both modes
        self.assertIn("F1@0.5 drops 0.160, tuned F1 drops 0.010, PR-AUC drops 0.020", self.index)
        self.assertIn("At f=1 PR-AUC drops by 0.350-0.350", self.index)
        self.assertIn("threshold shift", self.index)
        # gradient leakage: batch 1 plaintext mean(0.6, 0.8) = 0.7, ciphertext 0.4, baseline 0.45
        self.assertIn("Batch size 1: FedAvg r=0.700, FedAvg + CKKS r=0.400 vs mean-guess baseline r=0.450",
                      self.index)
        self.assertIn("at batch size 16 plaintext r=0.500 vs baseline 0.550", self.index)

    def test_index_lists_sources(self):
        self.assertIn("`attacks/label_flip.csv`", self.index)
        self.assertNotIn("label_flip_summary.csv", self.index)
        section = self.index.split("(`noniid`)")[1].split("##")[0]
        self.assertIn("credit_card_fedavg_b3_a0.1_s42.csv", section)
        self.assertNotIn("account_takeover", section)


class TestMakePlotsPartial(unittest.TestCase):
    def test_skips_missing_attacks_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "results"
            write_fixture(root, attacks=False)
            figs, log = quiet(mp.make_plots, root, root / "figures")
            skipped = {f.name: f.skipped for f in figs if f.skipped}
            self.assertEqual(set(skipped), {"label_flip", "gradient_leakage"})
            self.assertIn("no attacks/label_flip", skipped["label_flip"])
            self.assertTrue((root / "figures" / "main_comparison.png").exists())
            self.assertFalse((root / "figures" / "label_flip.png").exists())
            self.assertIn("Skipped: no attacks/gradient_leakage", (root / "figures" / "index.md").read_text())

    def test_empty_results_dir_does_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            figs, _ = quiet(mp.make_plots, d, Path(d) / "figures")
            self.assertTrue(all(f.skipped for f in figs))
            self.assertTrue((Path(d) / "figures" / "index.md").exists())

    def test_single_seed_single_alpha(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run_rows("credit_card", "fedavg", 3, 0.5, 42).to_csv(root / "credit_card_fedavg_b3_a0.5_s42.csv",
                                                                 index=False)
            figs, _ = quiet(mp.make_plots, root, root / "figures")
            state = {f.name: f.skipped for f in figs}
            self.assertIsNone(state["main_comparison"])
            self.assertIsNone(state["convergence_b3"])
            self.assertIn("no 5-bank runs", state["convergence_b5"])
            self.assertIn("no fedavg_he", state["he_cost"])
            self.assertIn("1 alpha", state["noniid"])


class TestReproduce(unittest.TestCase):
    def test_quick_dry_run_lists_stages_without_running(self):
        calls = []
        stages = tuple((name, lambda p, a, name=name: calls.append(name)) for name, _ in reproduce.STAGES)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(reproduce, "STAGES", stages):
            _, out = quiet(reproduce.main, ["--quick", "--dry-run", "--results-dir", d])
        self.assertEqual(calls, [])
        for i, name in enumerate(("grid", "attacks", "summarize", "plots"), 1):
            self.assertRegex(out, rf"{i}\. {name}")
        self.assertIn("--quick", out)
        self.assertIn("Dry run: nothing executed.", out)

    def test_runs_stages_in_order(self):
        calls = []
        stages = tuple((name, lambda p, a, name=name: calls.append(name)) for name, _ in reproduce.STAGES)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(reproduce, "STAGES", stages):
            _, out = quiet(reproduce.main, ["--quick", "--results-dir", d])
        self.assertEqual(calls, ["grid", "attacks", "summarize", "plots"])
        self.assertEqual(len(re.findall(r"stage \w+ done in", out)), 4)

    def test_full_profile_settings(self):
        p = reproduce.profile(False, None)
        self.assertEqual(p.grid_argv[:2], ["--rounds", "15"])
        self.assertEqual(p.flip["seeds"], [42, 43, 44])
        self.assertEqual(reproduce.profile(True, None).results, Path("results/quick"))


if __name__ == "__main__":
    unittest.main()
