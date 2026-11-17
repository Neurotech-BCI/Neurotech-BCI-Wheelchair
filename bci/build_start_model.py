'''
    Build a starting model for four_classifier.py from earlier recordings.

    Usage (from the repo root):
        python bci/build_start_model.py --data-dir "path/to/MI" --participant andy --out andy_start
        python bci/build_start_model.py --data-dir "path/to/MI" --participant churek --compare

    Reads every EDF in --data-dir whose name contains --participant (online
    control sessions are skipped), cuts LEFT / RIGHT / BOTH HANDS / REST
    windows the same way the live tasks do, checks accuracy on held-out
    sessions, and saves:
        <out>_model.pkl     the trained decoder
        <out>_windows.npz   the training windows
    Pass the model to four_classifier.py with --model <out>_model.pkl. The
    windows file next to it is picked up automatically and kept in every
    retrain during the session.

    --compare scores all three decoder designs for this person, so you can
    pick the best one (then pass it with --design here and in four_classifier.py).

    --data-dir must hold MI recordings only (no ErrP or SSVEP files): they use
    the same trigger numbers for different things.
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
import logging  # noqa: E402

from config import DualArrowTaskConfig, EEGConfig, StimConfig  # noqa: E402
from dual_mi_classifier import CLASS_NAMES, DESIGN_HELP, DESIGNS, DualMIClassifier, load_offline_dual_dataset  # noqa: E402

CHANCE = {2: 50, 3: 33, 4: 25}


def report(design: str, scores: dict) -> str:
    n = scores.get("n_states", 0)
    return (f"{design:14s} {n}-state accuracy (chance {CHANCE.get(n, 0)}%): "
            f"{scores['states_per_trial'] * 100:3.0f}% per trial, {scores['states'] * 100:3.0f}% per 1 s update")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Folder of MI recordings (EDF)")
    ap.add_argument("--participant", default="", help="Only use EDFs whose name contains this (e.g. andy)")
    ap.add_argument("--design", choices=DESIGNS, default=DualArrowTaskConfig.decoder_design,
                    help="Decoder design (default: %(default)s)")
    ap.add_argument("--compare", action="store_true", help="Score all three designs before saving")
    ap.add_argument("--sfreq", type=float, default=300.0, help="Sample rate of the live stream (DSI-7: 300)")
    ap.add_argument("--out", default=None, help="Output prefix (default: <participant>_start)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("build_start_model")
    eeg_cfg = EEGConfig()
    cfg = dataclasses.replace(DualArrowTaskConfig(), data_dir=args.data_dir, calirate_on_participant=args.participant)
    channels = list(eeg_cfg.picks)

    ds = load_offline_dual_dataset(cfg, eeg_cfg, StimConfig(), args.sfreq, channels, log)
    counts = {CLASS_NAMES[k]: v for k, v in ds.trial_counts().items()}
    if not DualMIClassifier.can_fit(ds):
        raise SystemExit(f"Not enough trials to train (need LEFT, RIGHT and REST): {counts}")
    n_sessions = len(set(ds.session_ids.tolist()))
    print(f"\nTrials: {counts} from {n_sessions} session(s), {len(ds)} windows")
    if n_sessions < 3:
        print("Note: fewer than 3 sessions, so accuracy below is tested on held-out trials, not held-out sessions,"
              " and will look better than it really is.")

    print("\nHeld-out accuracy:")
    designs = DESIGNS if args.compare else (args.design,)
    results = {}
    for design in designs:
        results[design] = DualMIClassifier(cfg.logreg_C, design=design).evaluate(ds)
        print("  " + report(design, results[design]))
    if args.compare:
        best = max(results, key=lambda d: results[d]["states_per_trial"])
        print(f"\nBest for this person: {best}  ({DESIGN_HELP[best]})")
        if best != args.design:
            print(f"Saving with --design {args.design}. Rerun with --design {best} to use the best one.")

    clf = DualMIClassifier(cfg.logreg_C, design=args.design).fit(ds)
    out = args.out or f"{args.participant or 'all'}_start"
    clf.save(f"{out}_model.pkl")
    ds.save(f"{out}_windows.npz")
    print(f"\nSaved {out}_model.pkl and {out}_windows.npz  (design: {args.design})")
    print(f"Start a session with: python psychopy/four_classifier.py --participant {args.participant or '<name>'}"
          f" --model {out}_model.pkl --design {args.design}")


if __name__ == "__main__":
    main()
