'''
    Dual-detector arrow task.

    1. Calibration: cued LEFT / RIGHT / BOTH / REST trials (counts in
       DualArrowTaskConfig). Two detectors (LEFT-vs-REST, RIGHT-vs-REST) are
       refit every few trials, and live feedback turns on as soon as there is
       enough data, so the participant sees their % on each side while they learn.
    2. Free run: the arrows fill continuously from the two probabilities.
         left arrow    = left MI only
         right arrow   = right MI only
         forward arrow = both arms
         center circle = rest

    Every decoder update is logged to <prefix>_dual_arrow_stream.csv and every
    calibration trial to <prefix>_dual_arrow_trials.csv. Plot a session with:
        python bci/plot_dual_arrow_session.py <prefix>
'''
from __future__ import annotations

import sys
from pathlib import Path

# Make the bci/ imports work no matter how this file is launched
# (python bci/<file>.py, python -m bci.<file>, IDE run button, any working directory).
_BCI_DIR = Path(__file__).resolve().parent
for _p in (str(_BCI_DIR.parent), str(_BCI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import csv
import logging
import math
from datetime import datetime

import numpy as np
from psychopy import core, event, visual

from mne_lsl.lsl import local_clock

from lsl_utils import connect_eeg_stream

from config import DualArrowTaskConfig, EEGConfig, LSLConfig, StimConfig
from bci_runtime import apply_runtime_config_overrides
from dual_mi_classifier import (
    BOTH,
    CLASS_NAMES,
    LEFT,
    REST,
    RIGHT,
    DualMIClassifier,
    WindowDataset,
    compute_arrow_fills,
    load_offline_dual_dataset,
)
from mental_command_worker import StreamingIIRFilter, resolve_channel_order, split_windows

TASK_NAME = "dual_arrow_task"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _make_task_logger(fname: str) -> logging.Logger:
    logger = logging.getLogger(f"dual_arrow.{fname}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(f"{fname}_dual_arrow.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _prompt_session_prefix() -> str:
    while True:
        raw = input("Enter participant name: ")
        name = "".join(ch for ch in "_".join(raw.strip().lower().split()) if ch.isalnum() or ch == "_").strip("_")
        if name:
            return f"{datetime.now().strftime('%m_%d_%y')}_{name}_dual_arrow"
        print("Participant name cannot be empty. Please try again.")


def build_cue_schedule(counts: dict[int, int], rng: np.random.Generator) -> list[int]:
    """Interleave classes in shuffled rounds so every stretch of the session
    has a mix of cues (and every class reaches the feedback minimum early)."""
    remaining = {k: int(v) for k, v in counts.items() if int(v) > 0}
    order: list[int] = []
    while remaining:
        round_classes = list(remaining)
        rng.shuffle(round_classes)
        for c in round_classes:
            order.append(c)
            remaining[c] -= 1
            if remaining[c] == 0:
                del remaining[c]
    return order


# ---------------------------------------------------------------------------
# Live decoding
# ---------------------------------------------------------------------------

class LiveDecoder:
    """Pulls new samples, filters them causally, keeps a rolling buffer, and
    records cued execute periods for calibration."""

    def __init__(self, stream, n_channels: int, sfreq: float, eeg_cfg: EEGConfig, task_cfg: DualArrowTaskConfig):
        self.stream = stream
        self.sfreq = float(sfreq)
        self.cfg = task_cfg
        self.reject = eeg_cfg.reject_peak_to_peak
        self.filter = StreamingIIRFilter.from_eeg_config(eeg_cfg=eeg_cfg, sfreq=sfreq, n_channels=n_channels)
        self.window_n = int(round(task_cfg.window_s * sfreq))
        self.keep_n = int(round((task_cfg.window_s + task_cfg.filter_context_s) * sfreq))
        self.pull_s = max(0.10, task_cfg.live_update_interval_s * 2.0)
        self.buf = np.empty((n_channels, 0), dtype=np.float32)
        self.last_ts: float | None = None
        self.newest_ts: float = -math.inf
        self.n_seen = 0
        self.rec_start = self.rec_stop = None
        self.rec_chunks: list[np.ndarray] = []
        self.next_pred_t = 0.0
        self.p_left = self.p_right = 0.0
        self.raw_left = self.raw_right = 0.0
        self.artifact = False
        self.has_estimate = False

    def poll(self) -> None:
        data, ts = self.stream.get_data(winsize=self.pull_s, picks="all")
        if data.size == 0 or ts is None or len(ts) == 0:
            return
        ts = np.asarray(ts)
        mask = np.ones(ts.shape, dtype=bool) if self.last_ts is None else ts > self.last_ts
        if not np.any(mask):
            return
        x = self.filter.process(np.asarray(data[:, mask], dtype=np.float32))
        ts = ts[mask]
        self.last_ts = float(ts[-1])
        self.newest_ts = self.last_ts
        self.n_seen += x.shape[1]
        self.buf = np.concatenate((self.buf, x), axis=1)[:, -self.keep_n:]
        if self.rec_start is not None:
            keep = (ts >= self.rec_start) & (ts < self.rec_stop)
            if np.any(keep):
                self.rec_chunks.append(x[:, keep])

    # -- calibration recording -------------------------------------------
    def start_recording(self, start_ts: float, stop_ts: float) -> None:
        self.rec_start, self.rec_stop, self.rec_chunks = float(start_ts), float(stop_ts), []

    def recording_complete(self) -> bool:
        return self.rec_start is not None and self.newest_ts >= self.rec_stop

    def take_recording(self) -> np.ndarray:
        block = np.concatenate(self.rec_chunks, axis=1) if self.rec_chunks else np.empty((self.buf.shape[0], 0), np.float32)
        self.rec_start = self.rec_stop = None
        self.rec_chunks = []
        return block

    def windows_from(self, block: np.ndarray) -> np.ndarray:
        w = split_windows(block=block, sfreq=self.sfreq, window_s=self.cfg.window_s, step_s=self.cfg.window_step_s)
        if w.shape[0] and self.reject is not None:
            w = w[np.ptp(w, axis=-1).max(axis=1) <= float(self.reject)]
        return w

    # -- prediction ----------------------------------------------------------
    @property
    def warmed_up(self) -> bool:
        return self.n_seen >= self.keep_n

    def maybe_predict(self, clf: DualMIClassifier | None, now: float) -> bool:
        """Update probabilities if an update is due. Returns True if updated."""
        if now < self.next_pred_t:
            return False
        self.next_pred_t = now + self.cfg.live_update_interval_s
        if clf is None or not clf.is_fitted or not self.warmed_up:
            return False
        win = self.buf[:, -self.window_n:]
        if self.reject is not None and float(np.ptp(win, axis=-1).max()) > float(self.reject):
            self.artifact = True     # hold the last estimate through artifacts
            return True
        self.artifact = False
        pl, pr = clf.predict_proba(win)
        self.raw_left, self.raw_right = float(pl[0]), float(pr[0])
        a = float(np.clip(self.cfg.ema_alpha, 0.01, 1.0))
        if not self.has_estimate:
            self.p_left, self.p_right, self.has_estimate = self.raw_left, self.raw_right, True
        else:
            self.p_left += a * (self.raw_left - self.p_left)
            self.p_right += a * (self.raw_right - self.p_right)
        return True

    def reset_estimate(self) -> None:
        self.p_left = self.p_right = 0.0
        self.has_estimate = False


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _clip_polygon_x(verts: list[tuple[float, float]], x_max: float) -> list[tuple[float, float]]:
    """Keep the part of a polygon with x <= x_max (Sutherland-Hodgman, one edge)."""
    out: list[tuple[float, float]] = []
    n = len(verts)
    for i in range(n):
        (x1, y1), (x2, y2) = verts[i], verts[(i + 1) % n]
        in1, in2 = x1 <= x_max, x2 <= x_max
        if in1:
            out.append((x1, y1))
        if in1 != in2:
            t = (x_max - x1) / (x2 - x1)
            out.append((x_max, y1 + t * (y2 - y1)))
    return out


# Arrow in local coordinates, pointing +x, base at x=0, tip at x=1.
_ARROW_LOCAL = [(0.0, -0.16), (0.62, -0.16), (0.62, -0.36), (1.0, 0.0), (0.62, 0.36), (0.62, 0.16), (0.0, 0.16)]


class FillArrow:
    def __init__(self, win, origin: tuple[float, float], angle_deg: float, length: float, color, outline):
        self.win, self.origin, self.length = win, origin, float(length)
        a = math.radians(angle_deg)
        self.cos, self.sin = math.cos(a), math.sin(a)
        self.outline = visual.ShapeStim(win, vertices=self._place(_ARROW_LOCAL), closeShape=True,
                                        fillColor=None, lineColor=outline, lineWidth=2.5)
        self.fill = visual.ShapeStim(win, vertices=self._place(_ARROW_LOCAL), closeShape=True,
                                     fillColor=color, lineColor=None, opacity=1.0)
        self.level = 0.0

    def _place(self, pts):
        L, ox, oy = self.length, self.origin[0], self.origin[1]
        return [(ox + L * (x * self.cos - y * self.sin), oy + L * (x * self.sin + y * self.cos)) for x, y in pts]

    def set_level(self, level: float) -> None:
        self.level = float(np.clip(level, 0.0, 1.0))
        if self.level > 0.01:
            self.fill.vertices = self._place(_clip_polygon_x(_ARROW_LOCAL, self.level))

    def set_highlight(self, color) -> None:
        self.outline.lineColor = color

    def draw(self) -> None:
        if self.level > 0.01:
            self.fill.draw()
        self.outline.draw()


class ArrowView:
    WHITE = (0.9, 0.9, 0.9)
    DIM = (0.45, 0.45, 0.45)
    CUE = (0.96, 0.78, 0.24)
    LEFT_C = (0.38, 0.84, 0.95)
    RIGHT_C = (0.95, 0.55, 0.38)
    FWD_C = (0.45, 0.90, 0.55)

    def __init__(self, win):
        self.win = win
        self.left = FillArrow(win, (-0.12, -0.05), 180, 0.55, self.LEFT_C, self.DIM)
        self.right = FillArrow(win, (0.12, -0.05), 0, 0.55, self.RIGHT_C, self.DIM)
        self.forward = FillArrow(win, (0.0, 0.08), 90, 0.50, self.FWD_C, self.DIM)
        self.rest = visual.Circle(win, radius=0.09, edges=64, pos=(0, -0.05),
                                  fillColor=(0.6, 0.6, 0.75), lineColor=self.DIM, lineWidth=2.0, opacity=1.0)
        self.rest_fill = visual.Circle(win, radius=0.09, edges=64, pos=(0, -0.05),
                                       fillColor=(0.6, 0.6, 0.75), lineColor=None)
        self.left_pct = visual.TextStim(win, text="", pos=(-0.42, -0.30), height=0.06, color=self.LEFT_C)
        self.right_pct = visual.TextStim(win, text="", pos=(0.42, -0.30), height=0.06, color=self.RIGHT_C)
        self.rest_txt = visual.TextStim(win, text="REST", pos=(0, -0.05), height=0.035, color=(0.1, 0.1, 0.1))
        self.title = visual.TextStim(win, text="", pos=(0, 0.88), height=0.055, color=self.WHITE)
        self.cue = visual.TextStim(win, text="", pos=(0, 0.74), height=0.07, color=self.CUE)
        self.info = visual.TextStim(win, text="", pos=(0, -0.62), height=0.040, color=self.WHITE, wrapWidth=1.8)
        self.status = visual.TextStim(win, text="", pos=(0, -0.86), height=0.036, color=(0.75, 0.75, 0.75), wrapWidth=1.8)
        self.show_feedback = False

    def set_feedback(self, p_left: float | None, p_right: float | None, threshold: float) -> None:
        if p_left is None or p_right is None:
            for a in (self.left, self.right, self.forward):
                a.set_level(0.0)
            self.rest_fill.opacity = 0.0
            self.left_pct.text = self.right_pct.text = ""
            return
        f = compute_arrow_fills(p_left, p_right, threshold)
        self.left.set_level(f.left)
        self.right.set_level(f.right)
        self.forward.set_level(f.forward)
        self.rest_fill.opacity = float(f.rest)
        self.left_pct.text = f"LEFT {100 * p_left:.0f}%"
        self.right_pct.text = f"RIGHT {100 * p_right:.0f}%"

    def highlight(self, cue: int | None) -> None:
        for a in (self.left, self.right, self.forward):
            a.set_highlight(self.DIM)
        self.rest.lineColor = self.DIM
        if cue == LEFT:
            self.left.set_highlight(self.CUE)
        elif cue == RIGHT:
            self.right.set_highlight(self.CUE)
        elif cue == BOTH:
            self.forward.set_highlight(self.CUE)
        elif cue == REST:
            self.rest.lineColor = self.CUE

    def draw(self) -> None:
        self.rest.draw()
        if self.rest_fill.opacity > 0.01:
            self.rest_fill.draw()
        self.rest_txt.draw()
        for a in (self.left, self.right, self.forward):
            a.draw()
        for t in (self.left_pct, self.right_pct, self.title, self.cue, self.info, self.status):
            t.draw()
        self.win.flip()


CUE_PROMPTS = {
    LEFT: "Imagine moving your LEFT hand",
    RIGHT: "Imagine moving your RIGHT hand",
    BOTH: "Imagine moving BOTH arms",
    REST: "Relax, rest",
}


def decoded_state(p_left: float, p_right: float, threshold: float) -> int:
    """Which of the four states the two detectors indicate (each side on if p > threshold)."""
    on_l, on_r = p_left > threshold, p_right > threshold
    if on_l and on_r:
        return BOTH
    if on_l:
        return LEFT
    if on_r:
        return RIGHT
    return REST


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def run_task(fname: str, max_trials: int | None = None) -> None:  # noqa: C901
    logger = _make_task_logger(fname)
    cfgs = apply_runtime_config_overrides(
        TASK_NAME, lsl_cfg=LSLConfig(), stim_cfg=StimConfig(), task_cfg=DualArrowTaskConfig(), eeg_cfg=EEGConfig(),
    )
    lsl_cfg, stim_cfg, task_cfg, eeg_cfg = cfgs["lsl_cfg"], cfgs["stim_cfg"], cfgs["task_cfg"], cfgs["eeg_cfg"]
    rng = np.random.default_rng()

    stream = connect_eeg_stream(lsl_cfg, bufsize=60.0, logger=logger)
    available = list(stream.info["ch_names"])
    ch_names, missing = resolve_channel_order(available, eeg_cfg.picks)
    if missing or len(ch_names) < 2:
        stream.disconnect()
        raise RuntimeError(f"Live stream is missing EEG picks {missing}. Configured: {list(eeg_cfg.picks)}. Available: {available}")
    stream.pick(ch_names)
    sfreq = float(stream.info["sfreq"])
    logger.info("Using channels %s at %.1f Hz", ch_names, sfreq)

    win = visual.Window(size=task_cfg.win_size, color=(-0.08, -0.08, -0.08), units="norm", fullscr=task_cfg.fullscreen)
    if win.size[1] > 0:
        win.viewScale = (float(win.size[1]) / float(win.size[0]), 1.0)
    view = ArrowView(win)
    view.title.text = "Left / Right / Both arrows"
    decoder = LiveDecoder(stream, len(ch_names), sfreq, eeg_cfg, task_cfg)
    clf = DualMIClassifier(C=task_cfg.logreg_C, include_other_side_as_negative=task_cfg.include_other_side_as_negative)
    session_t0 = local_clock()

    stream_fh = open(f"{fname}_dual_arrow_stream.csv", "w", newline="")
    trials_fh = open(f"{fname}_dual_arrow_trials.csv", "w", newline="")
    stream_log = csv.writer(stream_fh)
    trials_log = csv.writer(trials_fh)
    stream_log.writerow(["time_s", "phase", "trial", "cue", "stage", "p_left", "p_right", "raw_p_left", "raw_p_right",
                         "fill_left", "fill_right", "fill_forward", "fill_rest", "artifact", "model_version"])
    trials_log.writerow(["trial", "cue", "feedback_on", "model_version", "mean_p_left", "mean_p_right",
                         "mean_fill_left", "mean_fill_right", "mean_fill_forward", "mean_fill_rest",
                         "online_hit", "n_windows_saved", "n_train_trials"])

    state = {"phase": "setup", "trial": -1, "cue": "", "stage": ""}
    exec_preds: list[tuple[float, float, float]] = []   # (time, p_left, p_right) during execute

    def check_keys() -> list[str]:
        keys = event.getKeys()
        if "escape" in keys:
            raise KeyboardInterrupt
        return keys

    def frame() -> list[str]:
        """Poll EEG, update prediction + log, draw. Returns keys pressed."""
        keys = check_keys()
        decoder.poll()
        now = local_clock()
        if decoder.maybe_predict(clf if view.show_feedback else None, now) and decoder.has_estimate:
            f = compute_arrow_fills(decoder.p_left, decoder.p_right, task_cfg.activation_threshold)
            stream_log.writerow([f"{now - session_t0:.3f}", state["phase"], state["trial"], state["cue"], state["stage"],
                                 f"{decoder.p_left:.4f}", f"{decoder.p_right:.4f}",
                                 f"{decoder.raw_left:.4f}", f"{decoder.raw_right:.4f}",
                                 f"{f.left:.4f}", f"{f.right:.4f}", f"{f.forward:.4f}", f"{f.rest:.4f}",
                                 int(decoder.artifact), clf.version])
            if state["stage"] == "execute" and not decoder.artifact:
                exec_preds.append((now, decoder.p_left, decoder.p_right))
        if view.show_feedback and decoder.has_estimate:
            view.set_feedback(decoder.p_left, decoder.p_right, task_cfg.activation_threshold)
        else:
            view.set_feedback(None, None, task_cfg.activation_threshold)
        view.draw()
        return keys

    def hold(duration_s: float) -> None:
        clock = core.Clock()
        while clock.getTime() < duration_s:
            frame()

    def wait_space() -> None:
        while "space" not in frame():
            pass

    def fit_and_report(ds: WindowDataset, final: bool = False) -> None:
        view.status.text = "Updating model..."
        view.draw()
        clf.fit(ds)
        msg = f"model v{clf.version} trained on {len(ds)} windows, trials={ {CLASS_NAMES[k]: v for k, v in ds.trial_counts().items()} }"
        if final:
            scores = clf.evaluate(ds)
            msg += f" | CV left: {scores['left'].text()} | CV right: {scores['right'].text()}"
            view.info.text = f"Left detector: {scores['left'].text()}\nRight detector: {scores['right'].text()}"
        logger.info(msg)

    completed = 0
    dataset = WindowDataset.empty(len(ch_names), decoder.window_n, ch_names)
    has_offline = False
    try:
        # ---- Wait for EEG, optional offline pretraining -------------------
        view.cue.text = "Connecting to EEG..."
        view.status.text = "Waiting for enough signal to start the filter."
        while not decoder.warmed_up:
            frame()

        if task_cfg.data_dir:
            view.cue.text = "Loading previous sessions..."
            view.draw()
            offline = load_offline_dual_dataset(task_cfg, eeg_cfg, stim_cfg, sfreq, ch_names, logger)
            if len(offline):
                offline_scores = DualMIClassifier(task_cfg.logreg_C, task_cfg.include_other_side_as_negative).evaluate(offline)
                logger.info("Offline-only CV: left %s | right %s", offline_scores["left"].text(), offline_scores["right"].text())
                dataset = offline
                has_offline = DualMIClassifier.can_fit(dataset)
                if has_offline:
                    fit_and_report(dataset)

        # ---- Calibration -----------------------------------------------------
        counts = {LEFT: task_cfg.n_calibration_left, RIGHT: task_cfg.n_calibration_right,
                  BOTH: task_cfg.n_calibration_both, REST: task_cfg.n_calibration_rest}
        schedule = build_cue_schedule(counts, rng)
        if max_trials is not None:
            schedule = schedule[: int(max_trials)]
        live_counts = {k: 0 for k in counts}
        hits = {k: [0, 0] for k in counts}
        trials_since_fit = 0
        state["phase"] = "calibration"

        view.cue.text = "Calibration"
        view.info.text = (f"{len(schedule)} short trials. Follow the cue each time.\n"
                          "The arrows show what the decoder sees once it has enough data.")
        view.status.text = "Press SPACE to begin. ESC to quit."
        wait_space()

        for idx, cue in enumerate(schedule):
            state.update(trial=idx, cue=CLASS_NAMES[cue])
            enough_live = all(live_counts[c] >= task_cfg.min_trials_per_class_for_feedback
                              for c in counts if counts[c] > 0)
            view.show_feedback = clf.is_fitted and (enough_live or has_offline)

            # ITI
            state["stage"] = "iti"
            view.highlight(None)
            view.cue.text = ""
            view.status.text = f"Trial {idx + 1}/{len(schedule)}" + ("" if view.show_feedback else "   (feedback starts soon)")
            hold(task_cfg.calibration_iti_s)

            # Cue
            state["stage"] = "cue"
            view.highlight(cue)
            view.cue.text = f"Get ready: {CLASS_NAMES[cue]}"
            hold(task_cfg.calibration_cue_s)

            # Execute (recorded)
            state["stage"] = "execute"
            exec_preds.clear()
            onset = local_clock()
            decoder.start_recording(onset + task_cfg.calibration_trim_start_s, onset + task_cfg.calibration_execute_s)
            view.cue.text = CUE_PROMPTS[cue]
            hold(task_cfg.calibration_execute_s)
            state["stage"] = "flush"
            flush = core.Clock()
            while not decoder.recording_complete() and flush.getTime() < 1.0:
                frame()
            windows = decoder.windows_from(decoder.take_recording())
            n_saved = dataset.add_trial(windows, cue, session_id=-1, weight=1.0)
            if n_saved:
                live_counts[cue] += 1
                trials_since_fit += 1

            # Online score for this trial (only meaningful with feedback on). Uses the
            # same part of the trial the model trains on (after calibration_trim_start_s).
            scored = [p for p in exec_preds if p[0] >= onset + task_cfg.calibration_trim_start_s]
            if scored:
                pl = float(np.mean([p[1] for p in scored]))
                pr = float(np.mean([p[2] for p in scored]))
                f = compute_arrow_fills(pl, pr, task_cfg.activation_threshold)
                fills = {"left": f.left, "right": f.right, "forward": f.forward, "rest": f.rest}
                hit = int(decoded_state(pl, pr, task_cfg.activation_threshold) == cue)
                hits[cue][0] += hit
                hits[cue][1] += 1
                row_tail = [f"{pl:.4f}", f"{pr:.4f}"] + [f"{fills[k]:.4f}" for k in ("left", "right", "forward", "rest")] + [hit]
            else:
                row_tail = ["", "", "", "", "", "", ""]
            trials_log.writerow([idx, CLASS_NAMES[cue], int(view.show_feedback), clf.version] + row_tail
                                + [n_saved, sum(dataset.trial_counts().values())])
            trials_fh.flush()
            stream_fh.flush()
            completed = idx + 1

            # Refit
            if DualMIClassifier.can_fit(dataset) and (not clf.is_fitted or trials_since_fit >= task_cfg.retrain_every_n_trials):
                fit_and_report(dataset)
                trials_since_fit = 0

            # Break
            if (idx + 1) % task_cfg.break_every_n_trials == 0 and idx + 1 < len(schedule):
                state["stage"] = "break"
                score = "   ".join(f"{CLASS_NAMES[c]} {h}/{n}" for c, (h, n) in hits.items() if n)
                view.highlight(None)
                view.cue.text = "Break"
                view.info.text = f"Completed {idx + 1}/{len(schedule)} trials" + (f"\nOnline hits: {score}" if score else "")
                view.status.text = "Press SPACE to continue."
                wait_space()
                view.info.text = ""

        # ---- Final model -----------------------------------------------------
        if not DualMIClassifier.can_fit(dataset):
            raise RuntimeError(f"Not enough calibration data to train: {dataset.trial_counts()}")
        fit_and_report(dataset, final=True)
        clf.save(f"{fname}_dual_arrow_model.pkl")
        dataset.save(f"{fname}_dual_arrow_windows.npz")
        logger.info("Saved model and %d windows.", len(dataset))

        # ---- Free run -----------------------------------------------------------
        state.update(phase="free_run", trial=-1, cue="", stage="free")
        view.show_feedback = True
        view.highlight(None)
        view.cue.text = "Free run"
        view.status.text = "Left / right hand = turn, both arms = forward, relax = rest.   SPACE to start, ESC to finish."
        wait_space()
        view.info.text = ""
        decoder.reset_estimate()
        free = core.Clock()
        while task_cfg.free_run_duration_s is None or free.getTime() < task_cfg.free_run_duration_s:
            frame()

    except KeyboardInterrupt:
        logger.info("Session ended by user after %d calibration trials.", completed)
        if clf.is_fitted:
            clf.save(f"{fname}_dual_arrow_model.pkl")
        if len(dataset):
            dataset.save(f"{fname}_dual_arrow_windows.npz")
    finally:
        for fh in (stream_fh, trials_fh):
            try:
                fh.close()
            except Exception:
                pass
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        logger.info("Logs: %s_dual_arrow_stream.csv, %s_dual_arrow_trials.csv", fname, fname)


if __name__ == "__main__":
    prefix = _prompt_session_prefix()
    print(f"[SESSION] Using filename prefix: {prefix}")
    run_task(fname=prefix)
