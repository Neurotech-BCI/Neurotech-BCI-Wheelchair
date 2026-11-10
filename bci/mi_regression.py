'''
    Continuous motor-imagery decoding: predicts intended velocity (x, y).

        x  = turning, -1 (left) to +1 (right)
        y  = forward,  0 (stop) to +1 (forward)

    Model: covariance (OAS) -> Riemannian tangent space -> standardize -> ridge
    regression with two outputs, then a Kalman filter that smooths the output
    as a slowly changing velocity.

    Training data comes from two places:
      - cued trials (mi_task.py, feedback tasks): each class gets a fixed target,
        LEFT (-1, 0), RIGHT (+1, 0), BOTH HANDS (0, +1), REST (0, 0)
      - pursuit sessions (psychopy/mi_pursuit_task.py): every window is labeled
        with the direction from the cursor to the target at that moment
'''
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dual_mi_classifier import BOTH, LEFT, REST, RIGHT, DualMIClassifier, WindowDataset, compute_arrow_fills

# Fixed (x, y) target for each cued class.
CUED_TARGETS = {LEFT: (-1.0, 0.0), RIGHT: (1.0, 0.0), BOTH: (0.0, 1.0), REST: (0.0, 0.0)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_regression_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


class MIRegressor:
    """Ridge regression from an EEG window to intended velocity (x, y)."""

    def __init__(self, alpha: float = 1.0, lag_s: float = 0.0, channel_names: list[str] | None = None,
                 keep_channels: list[str] | None = None):
        self.alpha = float(alpha)
        # How far the label lags behind the EEG window it was paired with (s).
        self.lag_s = float(lag_s)
        # Channels in the recorded / live order, and the subset the model uses.
        self.channel_names = list(channel_names or [])
        self.keep_channels = list(keep_channels or self.channel_names)
        self.pipeline: Pipeline | None = None
        self.version = 0
        self.n_train_windows = 0

    @property
    def is_fitted(self) -> bool:
        return self.pipeline is not None

    def _keep_idx(self) -> list[int]:
        if not self.channel_names:
            return []
        return [self.channel_names.index(c) for c in self.keep_channels]

    def _select(self, X: np.ndarray) -> np.ndarray:
        idx = self._keep_idx()
        return X[:, idx, :] if idx and len(idx) != X.shape[1] else X

    def fit(self, X: np.ndarray, Y: np.ndarray, weights: np.ndarray | None = None) -> "MIRegressor":
        self.pipeline = make_regression_pipeline(self.alpha)
        self.pipeline.fit(self._select(np.asarray(X, dtype=np.float64)), np.asarray(Y, dtype=np.float64),
                          ridge__sample_weight=weights)
        self.n_train_windows = int(X.shape[0])
        self.version += 1
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(n_windows, n_channels, n_samples) -> (n_windows, 2) velocity."""
        if not self.is_fitted:
            raise RuntimeError("MIRegressor is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        return self.pipeline.predict(self._select(X))

    def predict_velocity(self, window: np.ndarray) -> tuple[float, float]:
        v = self.predict(window)[0]
        return float(np.clip(v[0], -1.0, 1.0)), float(np.clip(v[1], 0.0, 1.0))

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)


class DetectorVelocity:
    """Use the two detectors (LEFT vs REST, RIGHT vs REST) as a velocity
    decoder: x = right fill minus left fill, y = forward fill. Lets the
    pursuit task run before any regression model exists."""

    def __init__(self, clf: DualMIClassifier, threshold: float = 0.5):
        self.clf = clf
        self.threshold = float(threshold)
        self.channel_names = list(clf.channel_names)

    @property
    def is_fitted(self) -> bool:
        return self.clf.is_fitted

    def predict_velocity(self, window: np.ndarray) -> tuple[float, float]:
        pl, pr = self.clf.predict_proba(window)
        f = compute_arrow_fills(float(pl[0]), float(pr[0]), self.threshold)
        return f.right - f.left, f.forward


def load_velocity_decoder(path: str | Path, threshold: float = 0.5):
    """Load a saved MIRegressor, or a DualMIClassifier wrapped as a velocity decoder."""
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, MIRegressor):
        return obj
    if isinstance(obj, DualMIClassifier):
        return DetectorVelocity(obj, threshold)
    raise TypeError(f"{path} is neither an MIRegressor nor a DualMIClassifier")


class VelocityKalman:
    """Per-axis Kalman filter that treats the decoded velocity as a slowly
    drifting value. Larger measurement_var = smoother but slower output."""

    def __init__(self, process_var: float = 0.02, measurement_var: float = 0.3):
        self.q = float(process_var)
        self.r = float(measurement_var)
        self.reset()

    def reset(self) -> None:
        self.v = np.zeros(2)
        self.p = np.ones(2)

    def update(self, measurement: tuple[float, float]) -> tuple[float, float]:
        z = np.asarray(measurement, dtype=np.float64)
        self.p = self.p + self.q
        k = self.p / (self.p + self.r)
        self.v = self.v + k * (z - self.v)
        self.p = (1.0 - k) * self.p
        return float(self.v[0]), float(self.v[1])


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class RegressionData:
    X: np.ndarray        # (n_windows, n_channels, n_samples)
    Y: np.ndarray        # (n_windows, 2) intended velocity
    groups: np.ndarray   # (n_windows,) run or trial id, for cross-validation
    weights: np.ndarray  # (n_windows,)
    channel_names: list[str]

    def __len__(self) -> int:
        return int(self.Y.shape[0])


def cued_regression_data(ds: WindowDataset, weight: float = 1.0) -> RegressionData:
    """Turn cued LEFT/RIGHT/BOTH/REST windows into fixed (x, y) targets."""
    Y = np.array([CUED_TARGETS[int(label)] for label in ds.y], dtype=np.float64).reshape(-1, 2)
    return RegressionData(X=ds.X, Y=Y, groups=-1 - ds.trial_ids, weights=np.full(len(ds), float(weight)),
                          channel_names=list(ds.channel_names))


def pursuit_regression_data(path: str | Path, window_s: float, step_s: float, lag_s: float,
                            reject_peak_to_peak: float | None, run_offset: int = 0) -> RegressionData:
    """Cut a pursuit session (<prefix>_pursuit_data.npz) into windows, each
    labeled with the intended velocity lag_s before the window ends."""
    d = np.load(path, allow_pickle=False)
    eeg, eeg_ts = d["eeg"], d["eeg_ts"]
    sfreq = float(d["sfreq"])
    lt, lrun, lvx, lvy = d["label_t"], d["label_run"], d["label_vx"], d["label_vy"]
    win_n = int(round(window_s * sfreq))
    step_n = max(1, int(round(step_s * sfreq)))

    X, Y, G = [], [], []
    for run in np.unique(lrun[lrun >= 0]):
        in_run = lrun == run
        t0, t1 = float(lt[in_run].min()), float(lt[in_run].max())
        rt, rvx, rvy = lt[in_run], lvx[in_run], lvy[in_run]
        # Window ends must leave room for the lag inside the run.
        idx = np.where((eeg_ts >= t0 + window_s + lag_s) & (eeg_ts <= t1))[0]
        for end in idx[::step_n]:
            start = end - win_n + 1
            if start < 0:
                continue
            w = eeg[:, start:end + 1]
            # Skip windows that straddle a gap in the recording.
            if eeg_ts[end] - eeg_ts[start] > window_s * 1.2:
                continue
            if reject_peak_to_peak is not None and float(np.ptp(w, axis=-1).max()) > float(reject_peak_to_peak):
                continue
            t_label = float(eeg_ts[end]) - lag_s
            X.append(w)
            Y.append((np.interp(t_label, rt, rvx), np.interp(t_label, rt, rvy)))
            G.append(int(run) + run_offset)
    n_ch = eeg.shape[0]
    return RegressionData(
        X=np.asarray(X, dtype=np.float32).reshape(-1, n_ch, win_n),
        Y=np.asarray(Y, dtype=np.float64).reshape(-1, 2),
        groups=np.asarray(G, dtype=int),
        weights=np.ones(len(Y)),
        channel_names=[str(c) for c in d["ch_names"]],
    )


def concat_data(parts: list[RegressionData]) -> RegressionData:
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("No training windows")
    names = parts[0].channel_names
    for p in parts[1:]:
        if p.channel_names != names:
            raise ValueError(f"Channel mismatch: {p.channel_names} vs {names}")
    return RegressionData(
        X=np.concatenate([p.X for p in parts]),
        Y=np.concatenate([p.Y for p in parts]),
        groups=np.concatenate([p.groups for p in parts]),
        weights=np.concatenate([p.weights for p in parts]),
        channel_names=names,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@dataclass
class RegressionScore:
    r_x: float
    r_y: float
    n_folds: int

    @property
    def mean_r(self) -> float:
        vals = [v for v in (self.r_x, self.r_y) if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def text(self) -> str:
        fy = "n/a" if np.isnan(self.r_y) else f"{self.r_y:.2f}"
        folds = f" ({self.n_folds} folds)" if self.n_folds else ""
        return f"r_x={self.r_x:.2f} r_y={fy}{folds}"


def cross_validate(test: RegressionData, extra_train: RegressionData | None, alpha: float,
                   keep_channels: list[str] | None = None, max_folds: int = 10) -> RegressionScore:
    """Leave-run-out cross-validation on `test` (grouped by run). `extra_train`
    (e.g. cued trials) is added to every training fold but never tested."""
    groups = np.unique(test.groups)
    n_folds = int(min(max_folds, groups.size))
    if n_folds < 2:
        return RegressionScore(float("nan"), float("nan"), 0)
    pred = np.zeros_like(test.Y)
    for tr, te in GroupKFold(n_splits=n_folds).split(test.X, test.Y, test.groups):
        Xtr, Ytr, Wtr = test.X[tr], test.Y[tr], test.weights[tr]
        if extra_train is not None and len(extra_train):
            Xtr = np.concatenate([Xtr, extra_train.X])
            Ytr = np.concatenate([Ytr, extra_train.Y])
            Wtr = np.concatenate([Wtr, extra_train.weights])
        m = MIRegressor(alpha, channel_names=test.channel_names, keep_channels=keep_channels).fit(Xtr, Ytr, Wtr)
        pred[te] = m.predict(test.X[te])
    return RegressionScore(_corr(pred[:, 0], test.Y[:, 0]), _corr(pred[:, 1], test.Y[:, 1]), n_folds)


def score_model(model: Any, data: RegressionData) -> RegressionScore:
    pred = model.predict(data.X)
    return RegressionScore(_corr(pred[:, 0], data.Y[:, 0]), _corr(pred[:, 1], data.Y[:, 1]), 0)