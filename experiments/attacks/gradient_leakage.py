"""Gradient leakage: an honest-but-curious server tries to reconstruct a bank's training rows from its upload.

    python -m experiments.attacks.gradient_leakage [--quick] [--seeds 42 ...] [--trials 5] [--iters 600]

Attack (DLG / "inverting gradients" style, Zhu et al. 2019; Geiping et al. 2020, with iDLG label inference):
the server knows the model architecture, the weights it broadcast (w0) and the client protocol (one Adam run with
lr, pos_weight and local step count; attacker-favourable assumption). It observes the bank's upload w1, forms
u_obs = (w0 - w1) / lr and optimises dummy inputs x (and soft labels) so that the update the client would have
produced matches u_obs (per-layer cosine distance, Adam, fixed seed, bounded iterations).
The client optimiser is Adam: its first step is lr * g / (|g| + eps), i.e. ~lr * sign(g), not lr * g. The attack
therefore matches a smooth sign surrogate g / sqrt(g^2 + tau^2) (tau = TAU * mean |g| per layer) instead of the
raw gradient. Labels: the sign of the last-layer bias update gives the dominant label (iDLG); soft labels start
there (logit +-8, hard enough that pos_weight cannot invert the gradient sign) and are optimised when n_rec > 1.
Dummy rows are parameterised in standardised units around the population mean, clamped to the population
feature range, with a small L2 prior (LAM). Dropout: the client trains with a random dropout mask the server does
not know; the attacker uses the model in eval mode (no dropout). TAU, LAM, STEP and the +-8 label init were set
in a small pilot on bank 0 (sample seeds 0 and 1, batch 1 and 4); reported trials use other sample seeds.

Settings (credit_card, 3 banks, bank 0 is the target):
  * controlled: every bank trains on B rows of its real partition with batch_size B and one epoch, i.e. exactly
    one local step, for B in BATCH_SIZES. The update goes through the real FraudClient, Flower serialisation and
    strategy; the attack runs on the RoundObservation the server_observer received.
  * realistic: the bank's full partition, cfg.local_epochs epochs at cfg.batch_size (hundreds of Adam steps).
    The attacker reconstructs N_REC_REALISTIC rows and each is matched to its closest real training row
    (attacker-favourable); reconstruction is expected to fail.
Both modes: "fedavg" (server sees float32 weights) and "fedavg_he" (server sees uint8 CKKS ciphertext chunks).
In fedavg_he the identical pipeline is run on what the server actually has: the ciphertext bytes, numerically cast
to float32 and reshaped into the model's parameter shapes (the only way they type-check as weights).

Metrics (same matching for attack and baseline): rows matched by the Hungarian algorithm on per-row NMSE, where
NMSE = mean over features of (x_hat - x)^2 / max(var_j, 0.01) with var_j the pooled training variance (the floor
keeps near-constant binary indicators from dominating). recon_corr = Pearson correlation between matched
reconstructed and true feature vectors (across features, averaged over rows). Baseline: guess the population
mean feature vector for every row.

What the fedavg_he result does and does not show: it shows that this server, holding only ciphertext, gets no
usable signal from it with this attack, and that the bytes are uncorrelated with the plaintext update. It is not a
proof of security: confidentiality against the server rests on CKKS being IND-CPA secure (RLWE) with honestly
generated keys that the server never holds. The byte-level checks (ciphertext_checks.csv) are sanity checks only;
the bytes are a compressed SEAL serialisation with headers, so they are not expected to be uniform bytes.

Outputs (in --results-dir, default results/attacks, or results/attacks/quick with --quick):
  gradient_leakage.csv                   mode, batch_size, local_steps, recon_nmse, recon_corr, baseline_nmse,
                                         baseline_corr, attack_iters, seed (one row per trial)
  gradient_leakage_ciphertext_checks.csv fedavg_he only: byte statistics of the observed ciphertext
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
from flwr.common import ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from scipy.optimize import linear_sum_assignment
from scipy.stats import chisquare
from torch import nn
from torch.func import functional_call

from common import interfaces as I
from common.config import ExperimentConfig
from common.utils import get_weights, set_seed
from fl.client import FraudClient
from fl.logging import write_rows_csv
from fl.server import InProcessClientProxy, _call_all, _trusted_setup
from fl.strategy import EncryptedFedAvg, FraudFedAvg, RoundObservation

from .common import Recorder, flat, out_dir

FRAUD_TYPE = "credit_card"
N_CLIENTS = 3
TARGET = 0
BATCH_SIZES = (1, 4, 16)
N_REC_REALISTIC = 16
TAU, LAM, STEP, LABEL_LOGIT = 0.1, 0.1, 0.05, 8.0
VAR_FLOOR = 0.01
COLUMNS = ("mode", "batch_size", "local_steps", "recon_nmse", "recon_corr", "baseline_nmse", "baseline_corr",
           "attack_iters", "seed")


# ----------------------------------------------------------------------------- what the server observes
def observe_one_round(cfg: ExperimentConfig, data: dict, pos_weights: dict) -> tuple[RoundObservation, dict]:
    """Run one fit round of cfg.mode on the given per-bank data through the real client, Flower serialisation and
    strategy (same objects as fl.server.simulate), and return what the server_observer saw plus the clients
    (the clients are for the trusted harness's checks only; the attack never touches them). Clients are built
    as fl.client.create_client does, with each bank's loader pos_weight passed in to avoid reloading data."""
    he = cfg.mode == "fedavg_he"
    set_seed(cfg.seed)
    counts = {str(b): len(d[1]) for b, d in data.items()}
    N = sum(counts.values())
    model0 = I.build_model(cfg.fraud_type, next(iter(data.values()))[0].shape[1], seed=cfg.seed)
    he_clients, aggregator, _ = _trusted_setup(cfg, model0, counts) if he else ({}, None, None)
    manager, cid_to_bank, clients = SimpleClientManager(), {}, {}
    for b in data:
        model = I.build_model(cfg.fraud_type, data[b][0].shape[1], seed=cfg.seed)
        clients[b] = FraudClient(b, cfg, model, data[b], pos_weights[b], he_clients.get(b), N if he else None)
        manager.register(InProcessClientProxy(f"client-{b}", clients[b]))
        cid_to_bank[f"client-{b}"] = b
    rec = Recorder()
    initial = ndarrays_to_parameters(get_weights(model0))
    strategy = (EncryptedFedAvg(len(data), initial, aggregator, cid_to_bank, server_observer=rec) if he
                else FraudFedAvg(len(data), initial, server_observer=rec))
    params = strategy.initialize_parameters(manager)
    res, fail = _call_all(strategy.configure_fit(1, params, manager), "fit")
    strategy.aggregate_fit(1, res, fail)
    return rec.rounds[0], clients


def as_model_arrays(arrays, shapes) -> list[np.ndarray]:
    """Interpret received arrays as weights of the given shapes: numeric cast to float32 of the concatenated
    values, first sum(sizes) values reshaped. For a plaintext fedavg upload this is the identity; for uint8
    ciphertext chunks it is the only way the bytes type-check as model weights (it gives them no meaning)."""
    values = np.concatenate([np.asarray(a).ravel() for a in arrays]).astype(np.float32)
    sizes = [math.prod(s) for s in shapes]
    if values.size < sum(sizes):
        raise ValueError(f"received {values.size} values, model needs {sum(sizes)}")
    out, off = [], 0
    for s, n in zip(shapes, sizes):
        out.append(values[off:off + n].reshape(s))
        off += n
    return out


# ----------------------------------------------------------------------------- attack
@dataclass(frozen=True)
class Prior:
    """Public population statistics the attacker is assumed to know (pooled training data)."""
    mean: np.ndarray
    std: np.ndarray
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_data(cls, X):
        X = np.asarray(X, dtype=np.float64)
        return cls(X.mean(0), X.std(0), X.min(0), X.max(0))


def invert_update(model, broadcast, observed, *, n_rec: int, lr: float, pos_weight: float, prior: Prior,
                  iters: int, seed: int, tau: float = TAU, lam: float = LAM, step: float = STEP):
    """Reconstruct n_rec input rows (and soft labels) from one observed upload; returns (x_rec, y_rec) arrays.

    ``model`` gives the architecture (its weights are replaced by ``broadcast``); ``observed`` must be arrays of
    the model's parameter shapes (use as_model_arrays)."""
    torch.manual_seed(seed)
    model = model.eval()  # attacker does not know the client's dropout mask
    names = [n for n, _ in model.named_parameters()]
    params = {n: torch.tensor(np.asarray(w, np.float32), requires_grad=True) for n, w in zip(names, broadcast)}
    target = [torch.tensor((np.asarray(w0, np.float64) - np.asarray(w1, np.float64)) / lr, dtype=torch.float32)
              for w0, w1 in zip(broadcast, observed)]
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    mu = torch.tensor(prior.mean, dtype=torch.float32)
    sd = torch.tensor(np.maximum(prior.std, 1e-3), dtype=torch.float32)
    lo, hi = torch.tensor(prior.lo, dtype=torch.float32), torch.tensor(prior.hi, dtype=torch.float32)
    z = (0.1 * torch.randn(n_rec, len(prior.mean))).requires_grad_(True)
    # iDLG: positive last-bias update => the mean logit gradient is positive => mostly negatives (label 0).
    last_bias = float(target[-1].sum())
    logit = torch.full((n_rec,), -LABEL_LOGIT if last_bias > 0 else LABEL_LOGIT, requires_grad=n_rec > 1)
    opt = torch.optim.Adam([z, logit] if n_rec > 1 else [z], lr=step)
    for _ in range(iters):
        opt.zero_grad()
        x = mu + sd * z
        loss_fn = crit(functional_call(model, params, (x,)), torch.sigmoid(logit))
        grads = torch.autograd.grad(loss_fn, list(params.values()), create_graph=True)
        loss = lam * (z ** 2).mean()
        for g, u in zip(grads, target):
            s = g / torch.sqrt(g ** 2 + (tau * g.detach().abs().mean() + 1e-12) ** 2)
            loss = loss + 1 - nn.functional.cosine_similarity(s.flatten(), u.flatten(), dim=0)
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.copy_((torch.clamp(mu + sd * z, lo, hi) - mu) / sd)
    with torch.no_grad():
        return (mu + sd * z).numpy().astype(np.float64), torch.sigmoid(logit).numpy()


# ----------------------------------------------------------------------------- metrics
def nmse_matrix(A, B, prior: Prior, chunk: int = 20000) -> np.ndarray:
    """NMSE between every row of A and every row of B (features scaled by max(var, VAR_FLOOR))."""
    s = np.sqrt(np.maximum(prior.std ** 2, VAR_FLOOR))
    a = np.asarray(A, np.float64) / s
    out = []
    for i in range(0, len(B), chunk):
        b = np.asarray(B[i:i + chunk], np.float64) / s
        out.append(((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T) / a.shape[1])
    return np.maximum(np.concatenate(out, axis=1), 0.0)


def _corr(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def match_metrics(X_rec, X_true, prior: Prior) -> tuple[float, float]:
    """(mean NMSE, mean Pearson r) over a Hungarian matching of reconstructed rows to distinct true rows."""
    C = nmse_matrix(X_rec, X_true, prior)
    r, c = linear_sum_assignment(C)
    return float(C[r, c].mean()), float(np.mean([_corr(X_rec[i], X_true[j]) for i, j in zip(r, c)]))


def baseline_metrics(n_rec, X_true, prior: Prior) -> tuple[float, float]:
    return match_metrics(np.tile(prior.mean, (n_rec, 1)), X_true, prior)


# ----------------------------------------------------------------------------- ciphertext checks
def ciphertext_checks(obs: RoundObservation, client, broadcast_plain, bank: int = TARGET) -> dict:
    """Trusted-harness sanity checks on the ciphertext the server received from ``bank`` (needs the client's
    plaintext update and secret context, which the server never has)."""
    up = next(u for u in obs.uploads if u.bank_id == bank)
    raw = np.concatenate([np.asarray(a, np.uint8) for a in up.arrays])
    update = flat(client.last_update) - flat(broadcast_plain)
    head = raw[:update.size].astype(np.float64)
    hist = np.bincount(raw, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    again = np.concatenate([np.frombuffer(c, np.uint8) for c in
                            I.encrypt(client.last_update, client.scale, client.he_ctx)])
    n = min(len(again), len(raw))
    return {"n_bytes": int(raw.size), "corr_bytes_vs_update": _corr(head, update),
            "corr_bytes_vs_weights": _corr(head, flat(client.last_update)),
            "byte_entropy_bits": float(-(p[p > 0] * np.log2(p[p > 0])).sum()),
            "chi2_uniform_p": float(chisquare(hist).pvalue),
            "reencrypt_byte_agreement": float(np.mean(again[:n] == raw[:n]))}


# ----------------------------------------------------------------------------- experiment
def _attack(obs, model, cfg, X_true, n_rec, prior, pos_weight, iters, seed):
    shapes = [tuple(p.shape) for p in model.parameters()]
    up = next(u for u in obs.uploads if u.bank_id == TARGET)
    observed = as_model_arrays(up.arrays, shapes)
    broadcast = as_model_arrays(obs.broadcast, shapes)  # round 1: plaintext initial model in both modes
    X_rec, _ = invert_update(model, broadcast, observed, n_rec=n_rec, lr=cfg.lr, pos_weight=pos_weight,
                             prior=prior, iters=iters, seed=seed)
    return match_metrics(X_rec, X_true, prior), baseline_metrics(n_rec, X_true, prior)


def _run_seed(seed, trials, iters, batch_sizes, modes, local_epochs, realistic, alpha, rows, checks, t0):
    """All settings for one seed (federation seed, model init and trial sample seeds); appends to rows/checks."""
    base = ExperimentConfig(mode="fedavg", fraud_type=FRAUD_TYPE, n_clients=N_CLIENTS, alpha=alpha, seed=seed,
                            rounds=1, local_epochs=local_epochs)
    full = {b: I.load_client_data(b, FRAUD_TYPE, base) for b in range(N_CLIENTS)}
    prior = Prior.from_data(np.concatenate([d[0] for d in full.values()]))
    pos_weights = {b: I.pos_weight(b, FRAUD_TYPE, base) for b in full}
    pos_weight = pos_weights[TARGET]

    def record(mode, B, steps, metrics, trial_seed, setting, obs, clients, cfg):
        (rn, rc), (bn, bc) = metrics
        rows.append(dict(zip(COLUMNS, (mode, B, steps, rn, rc, bn, bc, iters, trial_seed))))
        print(f"  gradient_leakage {mode:9s} {setting:10s} B={B:<4} steps={steps:<5} seed={trial_seed}: "
              f"recon nmse {rn:.3f} r {rc:.3f} | baseline nmse {bn:.3f} r {bc:.3f}  "
              f"(elapsed {time.perf_counter() - t0:.0f}s)", flush=True)
        if mode == "fedavg_he":
            checks.append({"mode": mode, "setting": setting, "batch_size": B, "local_steps": steps,
                           "seed": trial_seed, **ciphertext_checks(obs, clients[TARGET], obs.broadcast)})

    for mode in modes:
        for B in batch_sizes:
            cfg = replace(base, mode=mode, batch_size=B, local_epochs=1)
            for t in range(trials):
                trial_seed = seed * 1000 + 100 + t  # disjoint from the pilot's sample seeds (0, 1)
                rng = np.random.default_rng(trial_seed)
                data = {}
                for b, (X, y, Xv, yv) in full.items():
                    idx = rng.choice(len(y), B, replace=False)
                    data[b] = (X[idx], y[idx], Xv, yv)
                obs, clients = observe_one_round(cfg, data, pos_weights)
                model = I.build_model(FRAUD_TYPE, prior.mean.size)
                metrics = _attack(obs, model, cfg, data[TARGET][0], B, prior, pos_weight, iters, trial_seed)
                record(mode, B, 1, metrics, trial_seed, "controlled", obs, clients, cfg)
        if realistic:
            cfg = replace(base, mode=mode)
            obs, clients = observe_one_round(cfg, full, pos_weights)
            steps = cfg.local_epochs * math.ceil(len(full[TARGET][1]) / cfg.batch_size)
            model = I.build_model(FRAUD_TYPE, prior.mean.size)
            metrics = _attack(obs, model, cfg, full[TARGET][0], N_REC_REALISTIC, prior, pos_weight, iters, seed)
            record(mode, cfg.batch_size, steps, metrics, seed, "realistic", obs, clients, cfg)


def run(seeds=(42,), trials: int = 5, iters: int = 600, batch_sizes=BATCH_SIZES, modes=("fedavg", "fedavg_he"),
        local_epochs: int = 2, realistic: bool = True, results_dir=None, quick: bool = False, alpha: float = 0.5):
    """Run all settings for each seed; write gradient_leakage.csv (+ ciphertext checks); return (rows, checks)."""
    rows, checks, t0 = [], [], time.perf_counter()
    for seed in seeds:
        _run_seed(seed, trials, iters, batch_sizes, modes, local_epochs, realistic, alpha, rows, checks, t0)
    d = out_dir(results_dir, quick)
    write_rows_csv(rows, d / "gradient_leakage.csv", overwrite=True)
    if checks:
        write_rows_csv(checks, d / "gradient_leakage_ciphertext_checks.csv", overwrite=True)
    return rows, checks


def summary_table(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return df.groupby(["mode", "batch_size", "local_steps"], sort=False)[
        ["recon_nmse", "recon_corr", "baseline_nmse", "baseline_corr"]].mean().reset_index().assign(
        trials=df.groupby(["mode", "batch_size", "local_steps"], sort=False).size().values)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Gradient-inversion attack by the server (see module docstring).")
    ap.add_argument("--quick", action="store_true", help="1 trial, 200 iterations, 1 local epoch -> results/attacks/quick")
    ap.add_argument("--seeds", nargs="+", type=int, default=[ExperimentConfig.seed])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    ap.add_argument("--modes", nargs="+", choices=["fedavg", "fedavg_he"], default=["fedavg", "fedavg_he"])
    ap.add_argument("--local-epochs", type=int, default=ExperimentConfig.local_epochs,
                    help="local epochs of the realistic setting")
    ap.add_argument("--no-realistic", action="store_true")
    ap.add_argument("--results-dir", default=None)
    a = ap.parse_args(argv)
    if a.quick:
        a.trials, a.iters, a.local_epochs = 1, 200, 1
    return a


def main(argv=None):
    a = parse_args(argv)
    rows, checks = run(a.seeds, a.trials, a.iters, a.batch_sizes, a.modes, a.local_epochs, not a.no_realistic,
                       a.results_dir, a.quick)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary_table(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if checks:
            print(pd.DataFrame(checks).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return rows, checks


if __name__ == "__main__":
    main()
