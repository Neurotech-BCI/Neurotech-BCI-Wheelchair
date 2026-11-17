'''
    Two independent motor-imagery detectors: LEFT-vs-REST and RIGHT-vs-REST.

    Each detector outputs its own probability, so the two sides are read
    separately instead of competing:
      - left high, right low   -> left arrow fills
      - right high, left low   -> right arrow fills
      - both high (both arms)  -> forward arrow fills
      - both low               -> rest

    Pipeline per detector (same feature path as the existing MI model):
      covariance (OAS) -> Riemannian tangent space -> standardize -> logistic regression
    Logistic regression is used instead of LDA so the outputs are calibrated
    probabilities that can be shown as percentages.
'''
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEFT = 1    # matches StimConfig.left_code
RIGHT = 2   # matches StimConfig.right_code
BOTH = 98   # internal only: both arms at once, counts as active for BOTH detectors
REST = 99   # internal only, never sent as a trigger (matches rest_class_code)
CLASS_NAMES = {LEFT: "LEFT", RIGHT: "RIGHT", BOTH: "BOTH", REST: "REST"}


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class WindowDataset:
    """Sliding windows with labels, the trial each came from, and its source."""
    X: np.ndarray                       # (n_windows, n_channels, n_samples)
    y: np.ndarray                       # (n_windows,) in {LEFT, RIGHT, REST}
    trial_ids: np.ndarray               # (n_windows,) unique per trial
    session_ids: np.ndarray             # (n_windows,) EDF file index, or -1 for live
    weights: np.ndarray                 # (n_windows,) sample weights
    channel_names: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, n_channels: int, n_samples: int, channel_names: list[str] | None = None) -> "WindowDataset":
        return cls(
            X=np.empty((0, n_channels, n_samples), dtype=np.float32),
            y=np.empty(0, dtype=int),
            trial_ids=np.empty(0, dtype=int),
            session_ids=np.empty(0, dtype=int),
            weights=np.empty(0, dtype=np.float64),
            channel_names=list(channel_names or []),
        )

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def next_trial_id(self) -> int:
        return int(self.trial_ids.max()) + 1 if len(self) else 0

    def append(self, other: "WindowDataset") -> "WindowDataset":
        if len(other) == 0:
            return self
        if len(self) == 0:
            return WindowDataset(
                X=other.X, y=other.y, trial_ids=other.trial_ids, session_ids=other.session_ids,
                weights=other.weights, channel_names=other.channel_names or self.channel_names,
            )
        offset = self.next_trial_id()
        return WindowDataset(
            X=np.concatenate([self.X, other.X]),
            y=np.concatenate([self.y, other.y]),
            trial_ids=np.concatenate([self.trial_ids, other.trial_ids + offset]),
            session_ids=np.concatenate([self.session_ids, other.session_ids]),
            weights=np.concatenate([self.weights, other.weights]),
            channel_names=self.channel_names,
        )

    def add_trial(self, windows: np.ndarray, label: int, session_id: int = -1, weight: float = 1.0) -> int:
        """Append one trial's windows in place. Returns the number of windows added."""
        if windows.shape[0] == 0:
            return 0
        tid = self.next_trial_id()
        n = windows.shape[0]
        if len(self) == 0:
            self.X = windows.astype(np.float32, copy=False)
        else:
            self.X = np.concatenate([self.X, windows.astype(np.float32, copy=False)])
        self.y = np.concatenate([self.y, np.full(n, int(label), dtype=int)])
        self.trial_ids = np.concatenate([self.trial_ids, np.full(n, tid, dtype=int)])
        self.session_ids = np.concatenate([self.session_ids, np.full(n, int(session_id), dtype=int)])
        self.weights = np.concatenate([self.weights, np.full(n, float(weight))])
        return n

    def trial_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for label in (LEFT, RIGHT, BOTH, REST):
            counts[label] = int(np.unique(self.trial_ids[self.y == label]).size)
        return counts

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path, X=self.X, y=self.y, trial_ids=self.trial_ids, session_ids=self.session_ids,
            weights=self.weights, channel_names=np.asarray(self.channel_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WindowDataset":
        d = np.load(path, allow_pickle=False)
        return cls(
            X=d["X"], y=d["y"], trial_ids=d["trial_ids"], session_ids=d["session_ids"],
            weights=d["weights"], channel_names=[str(c) for c in d["channel_names"]],
        )


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def make_detector(C: float = 1.0) -> Pipeline:
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=float(C), class_weight="balanced", max_iter=2000)),
    ])


def _binary_subset(
    ds: WindowDataset, active: int, include_other_side_as_negative: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    other = RIGHT if active == LEFT else LEFT
    positives = (active, BOTH)
    negatives = (REST, other) if include_other_side_as_negative else (REST,)
    mask = np.isin(ds.y, positives) | np.isin(ds.y, negatives)
    y_bin = np.isin(ds.y[mask], positives).astype(int)
    return ds.X[mask], y_bin, ds.trial_ids[mask], ds.session_ids[mask], ds.weights[mask]


@dataclass
class DetectorScore:
    balanced_accuracy: float
    auc: float
    n_folds: int
    grouping: str   # "session" or "trial"

    def text(self) -> str:
        if self.n_folds == 0:
            return "n/a"
        return f"bal.acc={self.balanced_accuracy:.2f} AUC={self.auc:.2f} ({self.n_folds}-fold by {self.grouping})"


class DualMIClassifier:
    """LEFT-vs-REST and RIGHT-vs-REST detectors trained on the same windows.

    BOTH (bilateral) trials are optional; when present they count as active for
    both detectors, which teaches them to fire together for the forward arrow."""

    def __init__(self, C: float = 1.0, include_other_side_as_negative: bool = False):
        self.C = float(C)
        self.include_other_side_as_negative = bool(include_other_side_as_negative)
        self.left: Pipeline | None = None
        self.right: Pipeline | None = None
        self.n_train_windows = 0
        self.version = 0
        self.channel_names: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self.left is not None and self.right is not None

    @staticmethod
    def can_fit(ds: WindowDataset, min_trials_per_class: int = 2) -> bool:
        counts = ds.trial_counts()
        return all(counts[c] >= min_trials_per_class for c in (LEFT, RIGHT, REST))

    def fit(self, ds: WindowDataset) -> "DualMIClassifier":
        if not self.can_fit(ds):
            raise ValueError(f"Need LEFT, RIGHT and REST trials to fit; have {ds.trial_counts()}")
        models = {}
        for active in (LEFT, RIGHT):
            X, y, _, _, w = _binary_subset(ds, active, self.include_other_side_as_negative)
            m = make_detector(self.C)
            m.fit(X, y, clf__sample_weight=w)
            models[active] = m
        self.left, self.right = models[LEFT], models[RIGHT]
        self.n_train_windows = len(ds)
        self.channel_names = list(ds.channel_names)
        self.version += 1
        return self

    def predict_proba(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (p_left_active, p_right_active), each shape (n_windows,)."""
        if not self.is_fitted:
            raise RuntimeError("DualMIClassifier is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        return self.left.predict_proba(X)[:, 1], self.right.predict_proba(X)[:, 1]

    def evaluate(self, ds: WindowDataset, max_folds: int = 5) -> dict[str, DetectorScore]:
        """Cross-validate each detector. Folds are grouped by recording session
        when there are several (leave-session-out), otherwise by trial, so no
        windows from one trial end up in both train and test."""
        out: dict[str, DetectorScore] = {}
        for name, active in (("left", LEFT), ("right", RIGHT)):
            X, y, trials, sessions, w = _binary_subset(ds, active, self.include_other_side_as_negative)
            uniq_sessions = np.unique(sessions)
            if uniq_sessions.size >= 3:
                groups, grouping = sessions, "session"
            else:
                groups, grouping = trials, "trial"
            n_groups = np.unique(groups).size
            n_folds = int(min(max_folds, n_groups))
            if n_folds < 2 or np.unique(y).size < 2:
                out[name] = DetectorScore(float("nan"), float("nan"), 0, grouping)
                continue
            probs = np.full(y.shape[0], np.nan)
            for tr, te in GroupKFold(n_splits=n_folds).split(X, y, groups):
                if np.unique(y[tr]).size < 2:
                    continue
                m = make_detector(self.C)
                m.fit(X[tr], y[tr], clf__sample_weight=w[tr])
                probs[te] = m.predict_proba(X[te])[:, 1]
            ok = ~np.isnan(probs)
            if ok.sum() == 0 or np.unique(y[ok]).size < 2:
                out[name] = DetectorScore(float("nan"), float("nan"), 0, grouping)
                continue
            out[name] = DetectorScore(
                balanced_accuracy=float(balanced_accuracy_score(y[ok], probs[ok] >= 0.5)),
                auc=float(roc_auc_score(y[ok], probs[ok])),
                n_folds=n_folds,
                grouping=grouping,
            )
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> "DualMIClassifier":
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, DualMIClassifier):
            raise TypeError(f"{path} does not contain a DualMIClassifier")
        return obj


# ---------------------------------------------------------------------------
# Probabilities -> arrow fills
# ---------------------------------------------------------------------------

@dataclass
class ArrowFills:
    left: float
    right: float
    forward: float
    rest: float
    left_activation: float
    right_activation: float


def compute_arrow_fills(p_left: float, p_right: float, threshold: float = 0.5) -> ArrowFills:
    """Map the two detector probabilities to fill levels in [0, 1].

    Each side's activation rises from 0 at `threshold` to 1 at p=1. Whatever
    both sides share is the forward (both arms) component; the remainder of
    each side is its own turn; rest is how far the stronger side is from active.
    """
    thr = float(np.clip(threshold, 0.0, 0.99))
    a_l = float(np.clip((p_left - thr) / (1.0 - thr), 0.0, 1.0))
    a_r = float(np.clip((p_right - thr) / (1.0 - thr), 0.0, 1.0))
    forward = min(a_l, a_r)
    return ArrowFills(
        left=a_l - forward,
        right=a_r - forward,
        forward=forward,
        rest=1.0 - max(a_l, a_r),
        left_activation=a_l,
        right_activation=a_r,
    )


# ---------------------------------------------------------------------------
# Offline EDF loading (existing cued LEFT/RIGHT sessions -> LEFT/RIGHT/REST)
# ---------------------------------------------------------------------------

def _windows(block: np.ndarray, sfreq: float, window_s: float, step_s: float, reject: float | None) -> np.ndarray:
    from mental_command_worker import split_windows

    w = split_windows(block=block, sfreq=sfreq, window_s=window_s, step_s=step_s)
    if w.shape[0] and reject is not None:
        w = w[np.ptp(w, axis=-1).max(axis=1) <= float(reject)]
    return w


def load_offline_dual_dataset(
    task_cfg: Any,
    eeg_cfg: Any,
    stim_cfg: Any,
    target_sfreq: float,
    target_channel_names: list[str],
    logger: Any = None,
) -> WindowDataset:
    """Build LEFT / RIGHT / BOTH / REST windows from the cued MI EDFs.

    Every cued trial (LEFT, RIGHT, BOTH HANDS, REST trigger codes) uses the MI
    part of its execute epoch (task_cfg.offline_mi_*_s after the trigger).
    Older recordings that only have LEFT/RIGHT trials get REST windows from the
    fixation ITI before each trial instead (task_cfg.offline_rest_*_s).
    """
    import mne
    from mental_command_worker import (
        filter_session,
        find_stim_channel,
        resolve_channel_order,
        standardize_offline_raw,
    )

    root = Path(task_cfg.data_dir).expanduser()
    edf_paths = sorted(root.rglob(task_cfg.edf_glob)) if task_cfg.data_dir else []
    who = str(getattr(task_cfg, "calirate_on_participant", "") or "")
    edf_paths = [p for p in edf_paths if who in str(p)]
    n_ch = len(target_channel_names)
    win_n = int(round(task_cfg.window_s * target_sfreq))
    ds = WindowDataset.empty(n_ch, win_n, list(target_channel_names))
    if not edf_paths:
        if logger:
            logger.info("No offline EDFs found (data_dir=%r, glob=%r, participant=%r).",
                        task_cfg.data_dir, task_cfg.edf_glob, who)
        return ds

    mi_a = int(round(task_cfg.offline_mi_start_s * target_sfreq))
    mi_b = int(round(task_cfg.offline_mi_stop_s * target_sfreq))
    rest_a = int(round(task_cfg.offline_rest_start_s * target_sfreq))
    rest_b = int(round(task_cfg.offline_rest_stop_s * target_sfreq))
    weight = float(task_cfg.offline_sample_weight)
    reject = eeg_cfg.reject_peak_to_peak
    code_to_label = {
        int(stim_cfg.left_code): LEFT,
        int(stim_cfg.right_code): RIGHT,
        int(task_cfg.offline_both_code): BOTH,
        int(task_cfg.offline_rest_code): REST,
    }

    for session_id, path in enumerate(edf_paths):
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        raw = standardize_offline_raw(raw)
        stim = find_stim_channel(raw)
        # Some EDF writers store trigger values as e.g. 0.99997; round so they read as 1.
        raw.apply_function(lambda x: np.round(x), picks=[stim], channel_wise=False)
        events = mne.find_events(raw, stim_channel=stim, min_duration=0.0, verbose=False)
        if len(events) == 0:
            continue
        times_s = events[:, 0].astype(np.float64) / float(raw.info["sfreq"])
        if abs(float(raw.info["sfreq"]) - float(target_sfreq)) > 1e-6:
            raw.resample(float(target_sfreq), npad="auto", verbose="ERROR")
        picks, missing = resolve_channel_order(raw.ch_names, target_channel_names)
        if missing:
            raise RuntimeError(f"EDF {path} is missing channels {missing}. Available: {raw.ch_names}")
        data = raw.get_data(picks=picks).astype(np.float32) * float(task_cfg.offline_eeg_scale_to_match_live)
        data = filter_session(block=data, eeg_cfg=eeg_cfg, sfreq=float(target_sfreq))

        has_rest_trials = bool(np.any(events[:, 2] == int(task_cfg.offline_rest_code)))
        for t_s, code in zip(times_s, events[:, 2]):
            label = code_to_label.get(int(code))
            if label is None:
                continue
            onset = int(round(t_s * target_sfreq))
            if onset + mi_b <= data.shape[1]:
                w = _windows(data[:, onset + mi_a:onset + mi_b], target_sfreq, task_cfg.window_s, task_cfg.window_step_s, reject)
                ds.add_trial(w, label, session_id=session_id, weight=weight)
            if not has_rest_trials and label in (LEFT, RIGHT) and onset + rest_a >= 0 and rest_b > rest_a:
                w = _windows(data[:, onset + rest_a:onset + rest_b], target_sfreq, task_cfg.window_s, task_cfg.window_step_s, reject)
                ds.add_trial(w, REST, session_id=session_id, weight=weight)
        if logger:
            logger.info("  %s: %s", path.name, "cued REST trials" if has_rest_trials else "REST from fixation ITI (older recording)")

    if logger:
        logger.info("Offline dual dataset: %d EDFs, trials=%s, windows=%d",
                    len(edf_paths), {CLASS_NAMES[k]: v for k, v in ds.trial_counts().items()}, len(ds))
    return ds
