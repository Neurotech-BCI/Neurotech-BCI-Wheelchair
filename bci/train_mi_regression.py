'''
    Train the MI regression model (intended velocity x, y) from collected data.

    Usage (from the repo root):
        python bci/train_mi_regression.py --pursuit logs/*_pursuit_data.npz
        python bci/train_mi_regression.py --pursuit logs/*_pursuit_data.npz \\
            --cued-edf-dir path/to/edfs --participant andy

    What it does:
      1. Loads pursuit sessions (continuous labels) and, optionally, cued
         trials (fixed targets: LEFT (-1, 0), RIGHT (+1, 0), BOTH (0, +1), REST (0, 0)).
      2. Searches the label lag and the ridge strength with leave-run-out
         cross-validation on the pursuit runs. Cued data is only ever added
         to the training folds.
      3. Checks for eye-movement contamination by repeating the best setting
         without the frontal channels (F3, F4).
      4. Compares against a model trained on cued data only, if given.
      5. Trains the final model on everything and saves it.

    Without --pursuit it trains a first model from cued data alone, which can
    start the pursuit task with --model.
'''
from __future__ import annotations

import sys
from pathlib import Path

# Make the bci/ imports work no matter how this file is launched.
_BCI_DIR = Path(__file__).resolve().parent
for _p in (str(_BCI_DIR.parent), str(_BCI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402

import numpy as np  # noqa: E402

from config import DualArrowTaskConfig, EEGConfig, StimConfig  # noqa: E402
from dual_mi_classifier import WindowDataset, load_offline_dual_dataset  # noqa: E402
from mi_regression import (  # noqa: E402
    MIRegressor,
    concat_data,
    cross_validate,
    cued_regression_data,
    pursuit_regression_data,
    score_model,
)

FRONTAL = ["F3", "F4"]


def _expand(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for p in patterns:
        out.extend(sorted(glob.glob(p)) or [p])
    return out


def load_pursuit(paths: list[str], lag_s: float, window_s: float, step_s: float, reject):
    parts = [pursuit_regression_data(p, window_s, step_s, lag_s, reject, run_offset=1000 * i)
             for i, p in enumerate(paths)]
    return concat_data(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pursuit", nargs="*", default=[], help="Pursuit session files (*_pursuit_data.npz)")
    ap.add_argument("--cued-windows", nargs="*", default=[],
                    help="Saved cued windows (*_windows.npz from the feedback or dual arrow tasks)")
    ap.add_argument("--cued-edf-dir", default=None, help="Folder of cued MI recordings (*offline_cal*.edf)")
    ap.add_argument("--participant", default="", help="Only use EDFs whose path contains this name")
    ap.add_argument("--cued-weight", type=float, default=0.5, help="Weight of cued windows vs pursuit windows")
    ap.add_argument("--lags", default="0,0.25,0.5,0.75,1.0,1.25,1.5", help="Label lags to try (s)")
    ap.add_argument("--alphas", default="0.1,1,10,100", help="Ridge strengths to try")
    ap.add_argument("--out", default="mi_regression_model.pkl", help="Where to save the model")
    args = ap.parse_args()

    eeg_cfg = EEGConfig()
    dcfg = DualArrowTaskConfig()
    window_s, step_s, reject = dcfg.window_s, dcfg.window_step_s, eeg_cfg.reject_peak_to_peak
    lags = [float(v) for v in args.lags.split(",")]
    alphas = [float(v) for v in args.alphas.split(",")]
    pursuit_paths = _expand(args.pursuit)

    # ---- Cued data -----------------------------------------------------------
    cued_parts = []
    for path in _expand(args.cued_windows):
        cued_parts.append(cued_regression_data(WindowDataset.load(path), args.cued_weight))
    if args.cued_edf_dir:
        if not pursuit_paths and not cued_parts:
            raise SystemExit("Cued EDFs need the channel list and sample rate of a pursuit or windows file.")
        ref = np.load(pursuit_paths[0]) if pursuit_paths else None
        ch_names = [str(c) for c in ref["ch_names"]] if ref is not None else cued_parts[0].channel_names
        sfreq = float(ref["sfreq"]) if ref is not None else None
        if sfreq is None:
            raise SystemExit("Pass at least one --pursuit file so the sample rate is known for the EDFs.")
        cfg = dataclasses.replace(dcfg, data_dir=args.cued_edf_dir, calirate_on_participant=args.participant)
        ds = load_offline_dual_dataset(cfg, eeg_cfg, StimConfig(), sfreq, ch_names)
        cued_parts.append(cued_regression_data(ds, args.cued_weight))
    cued = concat_data(cued_parts) if cued_parts else None
    if cued is not None:
        print(f"Cued data: {len(cued)} windows")

    # ---- Cued only (no pursuit yet) -----------------------------------------------
    if not pursuit_paths:
        if cued is None:
            raise SystemExit("Give --pursuit files and/or cued data.")
        model = MIRegressor(alphas[len(alphas) // 2], 0.0, channel_names=cued.channel_names)
        model.fit(cued.X, cued.Y, cued.weights)
        model.save(args.out)
        print(f"Trained a first model on cued data only ({len(cued)} windows). Saved {args.out}")
        print("Use it to start the pursuit task: --model " + args.out)
        return

    # ---- Search lag and ridge strength on pursuit runs ---------------------------
    print(f"Pursuit sessions: {len(pursuit_paths)}")
    print("\n lag (s) | alpha | leave-run-out score")
    results = []
    data_by_lag = {}
    for lag in lags:
        data = load_pursuit(pursuit_paths, lag, window_s, step_s, reject)
        data_by_lag[lag] = data
        for alpha in alphas:
            score = cross_validate(data, cued, alpha)
            results.append((score.mean_r, lag, alpha, score))
            print(f"  {lag:5.2f}  | {alpha:5g} | {score.text()}")
    results = [r for r in results if not np.isnan(r[0])]
    if not results:
        raise SystemExit("Not enough runs to cross-validate (need at least 2).")
    best_r, best_lag, best_alpha, best_score = max(results, key=lambda r: r[0])
    data = data_by_lag[best_lag]
    print(f"\nBest: lag {best_lag:.2f} s, alpha {best_alpha:g} -> {best_score.text()}  ({len(data)} windows)")

    # ---- Eye-movement check ----------------------------------------------------
    keep = [c for c in data.channel_names if c not in FRONTAL]
    if len(keep) < len(data.channel_names):
        no_front = cross_validate(data, cued, best_alpha, keep_channels=keep)
        print(f"Without {', '.join(FRONTAL)}: {no_front.text()}")
        if best_score.mean_r > 0 and no_front.mean_r < 0.5 * best_score.mean_r:
            print("  WARNING: most of the performance comes from the frontal channels. That usually means eye "
                  "movements, not motor imagery. Check the gaze instructions before trusting this model.")

    # ---- Baseline: cued data only ------------------------------------------------
    if cued is not None:
        base = MIRegressor(best_alpha, best_lag, channel_names=cued.channel_names).fit(cued.X, cued.Y, cued.weights)
        print(f"Cued-only model tested on all pursuit runs: {score_model(base, data).text()}")

    # ---- Final model ---------------------------------------------------------
    final = data if cued is None else concat_data([data, cued])
    model = MIRegressor(best_alpha, best_lag, channel_names=final.channel_names)
    model.fit(final.X, final.Y, final.weights)
    model.save(args.out)
    print(f"\nSaved {args.out}  (lag {best_lag:.2f} s, alpha {best_alpha:g}, {len(final)} windows)")
    print("Use it in the pursuit task with --model " + args.out)


if __name__ == "__main__":
    main()