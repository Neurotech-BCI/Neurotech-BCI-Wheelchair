"""Motor-imagery calibration task with live arrow feedback.

Same trials, timing and trigger codes as mi_task.py (LEFT, RIGHT, REST,
BOTH HANDS), so the EEG recording is labeled the same way. On top of that it
reads the live EEG stream and shows what the decoder sees:

    left arrow     fills with LEFT-only imagery
    right arrow    fills with RIGHT-only imagery
    forward arrow  fills when both sides are active (BOTH HANDS)
    rest circle    fills when neither side is active

Two detectors (LEFT vs REST and RIGHT vs REST) train during the session from
the 1 to 3 s part of each execute period and retrain every few trials. Pass
--model to start from a saved model and give feedback from the first trial.

Uses the live decoding code in ../bci (dual_arrow_task.py,
dual_mi_classifier.py, lsl_utils.py, config.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import argparse
import csv
import dataclasses
import json
import random
import sys
import time

import serial
import serial.tools.list_ports
from psychopy import core, event, visual

# Live decoding code lives in the bci/ folder.
_BCI_DIR = Path(__file__).resolve().parent.parent / "bci"
for _p in (str(_BCI_DIR.parent), str(_BCI_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from mne_lsl.lsl import local_clock  # noqa: E402

from config import DualArrowTaskConfig, EEGConfig, LSLConfig  # noqa: E402
from dual_arrow_task import LiveDecoder, decoded_state  # noqa: E402
from dual_mi_classifier import BOTH, LEFT, REST, RIGHT, DualMIClassifier, WindowDataset, compute_arrow_fills  # noqa: E402
from lsl_utils import connect_eeg_stream  # noqa: E402
from mental_command_worker import resolve_channel_order  # noqa: E402


# ----------------------------------------------------------------
# Embedded config
# ----------------------------------------------------------------


@dataclass(frozen=True)
class StimConfig:
    left_code: int = 1
    right_code: int = 2
    rest_code: int = 3
    both_hands_code: int = 4


@dataclass(frozen=True)
class TaskConfig:
    n_calibration_trials: int = 80
    max_trials_before_break: int = 20

    prep_duration_s: float = 3.0
    execution_duration_s: float = 3.0
    iti_duration_s: float = 3.0

    fullscreen: bool = False
    win_size: tuple[int, int] = (1200, 700)


@dataclass(frozen=True)
class FeedbackConfig:
    # Part of the execute period used for training and scoring (s after onset).
    mi_start_s: float = 1.0
    mi_stop_s: float = 3.0

    # Feedback starts once every class has this many trials, then the model
    # retrains after every retrain_every new trials.
    min_trials_per_class: int = 3
    retrain_every: int = 4

    # A detector counts as active above this probability (0.5 = chance).
    activation_threshold: float = 0.5
    # Smoothing of the displayed probabilities (1.0 = none).
    ema_alpha: float = 0.25

    log_dir: str = "logs"


@dataclass(frozen=True)
class SerialConfig:
    vid: int = 0x2341
    pid: int = 0x8037
    baudrate: int = 115200
    pulse_width_s: float = 0.01


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


class TriggerPort:
    """Send trigger codes to the hardware trigger hub via serial."""

    def __init__(self, port: str, baudrate: int, pulse_width_s: float):
        self.port = port
        self.baudrate = int(baudrate)
        self.pulse_width_s = float(pulse_width_s)
        self.ser: serial.Serial | None = None

    def open(self) -> None:
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0)
        time.sleep(0.05)

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def pulse(self, code: int) -> None:
        if self.ser is None:
            return
        code = int(code) & 0xFF
        self.ser.write(bytes([code]))
        self.ser.flush()
        core.wait(self.pulse_width_s)
        self.ser.write(bytes([0]))
        self.ser.flush()


class BalancedBlockScheduler:
    """Generate balanced LEFT/RIGHT/REST/BOTH HANDS codes in shuffled blocks."""

    def __init__(self, block_size: int, left_code: int, right_code: int, rest_code: int, both_hands_code: int, seed: int | None = None):
        if block_size < 2:
            raise ValueError("block_size must be >= 2")
        self.block_size = int(block_size)
        self.left_code = int(left_code)
        self.right_code = int(right_code)
        self.rest_code = int(rest_code)
        self.both_hands_code = int(both_hands_code)
        self.rng = random.Random(seed)
        self._bag: list[int] = []

    def _refill(self) -> None:
        n_left = self.block_size // 4
        n_right = (self.block_size - n_left) // 3
        n_rest = (self.block_size - n_left - n_right) // 2
        n_both_hands = self.block_size - n_left - n_right - n_rest
        self._bag = [self.left_code] * n_left + [self.right_code] * n_right + [self.rest_code] * n_rest + [self.both_hands_code] * n_both_hands
        self.rng.shuffle(self._bag)

    def next_code(self) -> int:
        if not self._bag:
            self._refill()
        return self._bag.pop()


def find_port_by_vid_pid(vid: int, pid: int) -> str:
    for p in serial.tools.list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    raise RuntimeError(
        f"No trigger hub found for VID:PID {vid:#06x}:{pid:#06x}. "
        "Connect device or pass --port explicitly."
    )


def sanitize_participant_name(raw_name: str) -> str:
    cleaned = "_".join(raw_name.strip().lower().split())
    cleaned = "".join(ch for ch in cleaned if ch.isalnum() or ch == "_")
    return cleaned.strip("_")


def build_session_prefix(participant: str) -> str:
    date_prefix = datetime.now().strftime("%m_%d_%y")
    return f"{date_prefix}_{participant}_feedback_offline_cal"


def grow_from_tail(verts: list[tuple[float, float]], level: float, axis: int) -> list[tuple[float, float]]:
    """Shrink an arrow toward its tail along one axis (0 = x, 1 = y), so a
    level of 0.0 to 1.0 grows the fill from the tail up to the full arrow.
    The first vertex of every arrow below sits on its tail."""
    tail = verts[0][axis]
    out = []
    for x, y in verts:
        if axis == 0:
            out.append((tail + (x - tail) * level, y))
        else:
            out.append((x, tail + (y - tail) * level))
    return out


_ARROW_VERTS_RIGHT = [
    (-0.25, 0.04), (0.05, 0.04), (0.05, 0.10),
    (0.25, 0.0),
    (0.05, -0.10), (0.05, -0.04), (-0.25, -0.04),
]
_ARROW_VERTS_LEFT = [(-x, y) for x, y in _ARROW_VERTS_RIGHT]
_ARROW_VERTS_UP = [(-y, x) for x, y in _ARROW_VERTS_RIGHT]

# Smaller copies of the same arrows for the feedback row under the cross.
_FB_VERTS_RIGHT = [(0.5 * x, 0.5 * y) for x, y in _ARROW_VERTS_RIGHT]
_FB_VERTS_LEFT = [(-x, y) for x, y in _FB_VERTS_RIGHT]
_FB_VERTS_UP = [(-y, x) for x, y in _FB_VERTS_RIGHT]


# ----------------------------------------------------------------
# Task
# ----------------------------------------------------------------


def run_task(session_name: str, port: str, task_cfg: TaskConfig, stim_cfg: StimConfig, ser_cfg: SerialConfig,
             fb_cfg: FeedbackConfig, model_path: str | None = None) -> None:
    left = int(stim_cfg.left_code)
    right = int(stim_cfg.right_code)
    rest = int(stim_cfg.rest_code)
    both_hands = int(stim_cfg.both_hands_code)

    def code_to_name(code: int) -> str:
        if int(code) == left:
            return "LEFT"
        if int(code) == right:
            return "RIGHT"
        if int(code) == rest:
            return "REST"
        return "BOTH HANDS"

    # Trigger code -> class label used by the detectors.
    code_to_label = {left: LEFT, right: RIGHT, rest: REST, both_hands: BOTH}
    label_to_name = {LEFT: "LEFT", RIGHT: "RIGHT", REST: "REST", BOTH: "BOTH HANDS"}

    # Live EEG
    eeg_cfg = EEGConfig()
    dec_cfg = dataclasses.replace(DualArrowTaskConfig(), ema_alpha=fb_cfg.ema_alpha,
                                  activation_threshold=fb_cfg.activation_threshold)
    stream = connect_eeg_stream(LSLConfig(), bufsize=60.0)
    ch_names, missing = resolve_channel_order(list(stream.info["ch_names"]), eeg_cfg.picks)
    if missing:
        stream.disconnect()
        raise RuntimeError(f"Live stream is missing EEG channels {missing}. Available: {stream.info['ch_names']}")
    stream.pick(ch_names)
    sfreq = float(stream.info["sfreq"])
    decoder = LiveDecoder(stream, len(ch_names), sfreq, eeg_cfg, dec_cfg)
    dataset = WindowDataset.empty(len(ch_names), decoder.window_n, ch_names)

    clf = DualMIClassifier.load(model_path) if model_path else DualMIClassifier(C=dec_cfg.logreg_C)
    if model_path and clf.channel_names and list(clf.channel_names) != list(ch_names):
        stream.disconnect()
        raise RuntimeError(f"Model channels {clf.channel_names} do not match the stream {ch_names}")

    # Session logs
    log_dir = Path(fb_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = log_dir / session_name
    with open(f"{base}_config.json", "w", encoding="utf-8") as fh:
        json.dump({"session_name": session_name, "created_at": datetime.now().isoformat(timespec="seconds"),
                   "task_config": asdict(task_cfg), "feedback_config": asdict(fb_cfg),
                   "trigger_codes": asdict(stim_cfg), "eeg_channels": ch_names, "sfreq": sfreq,
                   "start_model": model_path}, fh, indent=2)
    trials_fh = open(f"{base}_trials.csv", "w", newline="")
    trials_log = csv.writer(trials_fh)
    trials_log.writerow(["trial", "condition", "trigger_code", "onset_lsl_s", "feedback_on", "model_version",
                         "mean_p_left", "mean_p_right", "decoded", "hit", "n_windows_saved"])

    trig = TriggerPort(port=port, baudrate=ser_cfg.baudrate, pulse_width_s=ser_cfg.pulse_width_s)
    trig.open()

    bg = (-0.1, -0.1, -0.1)
    white = (0.9, 0.9, 0.9)
    lit = (0.9, 0.9, 0.2)

    win = visual.Window(
        size=task_cfg.win_size,
        color=bg,
        units="norm",
        fullscr=task_cfg.fullscreen,
    )

    fixation = visual.TextStim(win, text="+", pos=(0, 0), height=0.16, color=white)
    cue_text = visual.TextStim(win, text="", pos=(0, 0.35), height=0.08, color=white)
    status_text = visual.TextStim(win, text="", pos=(0, -0.45), height=0.05, color=white)
    prep_arrow = visual.ShapeStim(
        win,
        vertices=_ARROW_VERTS_RIGHT,
        pos=(0, 0.15),
        fillColor=lit,
        lineColor=white,
        lineWidth=2,
        opacity=0,
    )

    # Feedback row under the cross: outline arrows that fill from the tail.
    fb_y = -0.24
    fb_left_outline = visual.ShapeStim(win, vertices=_FB_VERTS_LEFT, pos=(-0.3, fb_y), fillColor=None, lineColor=white, lineWidth=2)
    fb_right_outline = visual.ShapeStim(win, vertices=_FB_VERTS_RIGHT, pos=(0.3, fb_y), fillColor=None, lineColor=white, lineWidth=2)
    fb_up_outline = visual.ShapeStim(win, vertices=_FB_VERTS_UP, pos=(0, fb_y), fillColor=None, lineColor=white, lineWidth=2)
    fb_left_fill = visual.ShapeStim(win, vertices=_FB_VERTS_LEFT, pos=(-0.3, fb_y), fillColor=lit, lineColor=None, opacity=0)
    fb_right_fill = visual.ShapeStim(win, vertices=_FB_VERTS_RIGHT, pos=(0.3, fb_y), fillColor=lit, lineColor=None, opacity=0)
    fb_up_fill = visual.ShapeStim(win, vertices=_FB_VERTS_UP, pos=(0, fb_y), fillColor=lit, lineColor=None, opacity=0)
    fb_rest = visual.Circle(win, radius=0.11, edges=64, pos=(0, 0), fillColor=lit, lineColor=white, lineWidth=2, opacity=0)
    fb_rest.size = (float(win.size[1]) / float(win.size[0]), 1.0)  # keep it round in norm units
    fb_left_pct = visual.TextStim(win, text="", pos=(-0.6, fb_y), height=0.05, color=white)
    fb_right_pct = visual.TextStim(win, text="", pos=(0.6, fb_y), height=0.05, color=white)

    feedback = {"on": False}

    def update_feedback() -> None:
        decoder.poll()
        decoder.maybe_predict(clf if clf.is_fitted else None, local_clock())
        feedback["on"] = clf.is_fitted and decoder.has_estimate
        if not feedback["on"]:
            fb_left_fill.opacity = fb_right_fill.opacity = fb_up_fill.opacity = fb_rest.opacity = 0
            fb_left_pct.text = fb_right_pct.text = ""
            return
        f = compute_arrow_fills(decoder.p_left, decoder.p_right, fb_cfg.activation_threshold)
        fb_left_fill.vertices = grow_from_tail(_FB_VERTS_LEFT, max(f.left, 0.01), axis=0)
        fb_right_fill.vertices = grow_from_tail(_FB_VERTS_RIGHT, max(f.right, 0.01), axis=0)
        fb_up_fill.vertices = grow_from_tail(_FB_VERTS_UP, max(f.forward, 0.01), axis=1)
        fb_left_fill.opacity = 1 if f.left > 0.01 else 0
        fb_right_fill.opacity = 1 if f.right > 0.01 else 0
        fb_up_fill.opacity = 1 if f.forward > 0.01 else 0
        fb_rest.opacity = 0.6 * f.rest
        fb_left_pct.text = f"L {100 * decoder.p_left:.0f}%"
        fb_right_pct.text = f"R {100 * decoder.p_right:.0f}%"

    def draw_scene() -> None:
        if feedback["on"]:
            fb_rest.draw()
        fixation.draw()
        prep_arrow.draw()
        for stim in (fb_left_fill, fb_right_fill, fb_up_fill):
            if stim.opacity > 0:
                stim.draw()
        for stim in (fb_left_outline, fb_right_outline, fb_up_outline, fb_left_pct, fb_right_pct):
            stim.draw()
        cue_text.draw()
        status_text.draw()

    def wait_with_display(duration: float) -> None:
        clock = core.Clock()
        while clock.getTime() < duration:
            update_feedback()
            draw_scene()
            win.flip()
            if "escape" in event.getKeys():
                raise KeyboardInterrupt

    def wait_for_space() -> None:
        while True:
            update_feedback()
            draw_scene()
            win.flip()
            keys = event.getKeys()
            if "space" in keys:
                return
            if "escape" in keys:
                raise KeyboardInterrupt

    def retrain() -> None:
        cue_text.text = "Updating model..."
        draw_scene()
        win.flip()
        clf.fit(dataset)
        print(f"[MODEL] v{clf.version} trained on {len(dataset)} windows")

    scheduler = BalancedBlockScheduler(
        block_size=max(2, task_cfg.max_trials_before_break),
        left_code=left,
        right_code=right,
        rest_code=rest,
        both_hands_code=both_hands
    )

    print(f"[SESSION] {session_name}")
    print(f"[SERIAL] Sending triggers on {port} @ {ser_cfg.baudrate}")
    print(f"[EEG] {ch_names} @ {sfreq:.0f} Hz")

    trials_done = 0
    try:
        cue_text.text = "Connecting to EEG..."
        while not decoder.warmed_up:
            wait_with_display(0.1)

        cue_text.text = (
            "Motor Imagery Calibration with Feedback\n\n"
            "Fixate on the center cross.\n"
            "An arrow will cue LEFT, RIGHT, REST, or BOTH HANDS preparation.\n"
            "Then execute the cued motor imagery when prompted.\n"
            "The arrows below the cross show what the decoder sees.\n\n"
            "Press SPACE to begin. ESC to quit."
        )
        status_text.text = ""
        fixation.text = "+"
        wait_for_space()

        trials_per_class = {LEFT: 0, RIGHT: 0, REST: 0, BOTH: 0}
        since_retrain = 0

        for cal_idx in range(task_cfg.n_calibration_trials):
            y_true = scheduler.next_code()
            label = code_to_label[y_true]
            feedback_on = clf.is_fitted

            # ITI
            prep_arrow.opacity = 0
            cue_text.text = ""
            status_text.text = f"Calibration Trial {cal_idx + 1}/{task_cfg.n_calibration_trials}"
            fixation.text = "+"
            wait_with_display(task_cfg.iti_duration_s)

            # Prepare phase
            cue_text.text = "Prepare"
            status_text.text = f"Get ready: {code_to_name(y_true)}"
            if y_true == left:
                prep_arrow.vertices = _ARROW_VERTS_LEFT
                prep_arrow.opacity = 1
            elif y_true == right:
                prep_arrow.vertices = _ARROW_VERTS_RIGHT
                prep_arrow.opacity = 1
            elif y_true == rest:
                prep_arrow.opacity = 0
            else:
                prep_arrow.vertices = _ARROW_VERTS_UP
                prep_arrow.opacity = 1

            wait_with_display(task_cfg.prep_duration_s)

            # Execute phase: show cue then pulse trigger
            prep_arrow.opacity = 0
            cue_text.text = f"EXECUTE: {code_to_name(y_true)} MOTOR IMAGERY"
            status_text.text = "Go"
            update_feedback()
            draw_scene()
            win.flip()
            onset = local_clock()
            trig.pulse(y_true)
            decoder.start_recording(onset + fb_cfg.mi_start_s, onset + fb_cfg.mi_stop_s)

            # Score the trial from the predictions made 1 to 3 s after onset
            mi_preds = []
            clock = core.Clock()
            while clock.getTime() < task_cfg.execution_duration_s or not decoder.recording_complete():
                if clock.getTime() > task_cfg.execution_duration_s + 1.0:
                    break
                update_feedback()
                draw_scene()
                win.flip()
                if "escape" in event.getKeys():
                    raise KeyboardInterrupt
                now = local_clock()
                if feedback["on"] and not decoder.artifact and onset + fb_cfg.mi_start_s <= now <= onset + fb_cfg.mi_stop_s:
                    mi_preds.append((decoder.p_left, decoder.p_right))

            # Save this trial's data for the model
            n_saved = dataset.add_trial(decoder.windows_from(decoder.take_recording()), label)
            if n_saved:
                trials_per_class[label] += 1
                since_retrain += 1

            if mi_preds:
                p_left = float(np.mean([p[0] for p in mi_preds]))
                p_right = float(np.mean([p[1] for p in mi_preds]))
                decoded = decoded_state(p_left, p_right, fb_cfg.activation_threshold)
                score = [f"{p_left:.4f}", f"{p_right:.4f}", label_to_name[decoded], int(decoded == label)]
            else:
                score = ["", "", "", ""]
            trials_log.writerow([cal_idx, code_to_name(y_true), y_true, f"{onset:.4f}", int(feedback_on),
                                 clf.version] + score + [n_saved])
            trials_fh.flush()
            trials_done = cal_idx + 1

            # Retrain once every class has enough trials, then every few trials
            enough = all(n >= fb_cfg.min_trials_per_class for n in trials_per_class.values())
            if enough and (not clf.is_fitted or since_retrain >= fb_cfg.retrain_every):
                retrain()
                since_retrain = 0

            # Breaks
            if (
                (cal_idx + 1) % task_cfg.max_trials_before_break == 0
                and (cal_idx + 1) < task_cfg.n_calibration_trials
            ):
                cue_text.text = "Break\n\nPress SPACE to continue"
                status_text.text = f"Completed {cal_idx + 1}/{task_cfg.n_calibration_trials} trials"
                prep_arrow.opacity = 0
                fixation.text = "+"
                wait_for_space()

        cue_text.text = "Calibration complete\n\nPress ESC to close"
        status_text.text = ""
        prep_arrow.opacity = 0
        fixation.text = "+"
        while "escape" not in event.getKeys():
            update_feedback()
            draw_scene()
            win.flip()

    except KeyboardInterrupt:
        print(f"\n[SESSION] Interrupted by user after {trials_done} trials")

    finally:
        if clf.is_fitted:
            clf.save(f"{base}_model.pkl")
        if len(dataset):
            dataset.save(f"{base}_windows.npz")
        trials_fh.close()
        trig.close()
        try:
            stream.disconnect()
        except Exception:
            pass
        try:
            win.close()
        except Exception:
            pass
        print(f"[SESSION] Logs saved to {log_dir.resolve()}")
        core.quit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MI calibration task with live arrow feedback")
    parser.add_argument("--participant", type=str, default=None, help="Participant name for session label")
    parser.add_argument("--port", type=str, default=None, help="Serial port override (e.g., COM6 or /dev/ttyUSB0)")
    parser.add_argument("--trials", type=int, default=None, help="Override number of calibration trials")
    parser.add_argument("--model", type=str, default=None, help="Saved model (.pkl) to give feedback from the first trial")
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stim_cfg = StimConfig()
    task_cfg = TaskConfig(
        n_calibration_trials=args.trials if args.trials is not None else TaskConfig.n_calibration_trials,
        fullscreen=bool(args.fullscreen),
    )
    ser_cfg = SerialConfig()
    fb_cfg = FeedbackConfig()

    participant = sanitize_participant_name(args.participant or input("Enter participant name: "))
    if not participant:
        raise ValueError("Participant name cannot be empty")

    session_name = build_session_prefix(participant)
    port = args.port or find_port_by_vid_pid(ser_cfg.vid, ser_cfg.pid)

    run_task(
        session_name=session_name,
        port=port,
        task_cfg=task_cfg,
        stim_cfg=stim_cfg,
        ser_cfg=ser_cfg,
        fb_cfg=fb_cfg,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()