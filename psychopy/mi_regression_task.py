"""Motor-imagery pursuit task for collecting regression training data.

The participant moves a cursor to targets with motor imagery:
    LEFT hand imagery   -> cursor moves left
    RIGHT hand imagery  -> cursor moves right
    BOTH HANDS imagery  -> cursor moves up (forward)
    REST                -> cursor stays still

Every frame is labeled with the INTENDED velocity: the direction from the
cursor to the target (or zero during REST). That label, paired with the EEG,
is what the regression model learns from.

Each run alternates REST periods and target trials. Targets appear above and
to the sides of the start point, at random angles, so the labels cover
in-between directions (e.g. up and to the left), not just four classes.

Shared control: the cursor moves by
    assist * (true direction to target) + (1 - assist) * (decoded direction)
Early runs use full assist (the cursor moves on its own while the participant
imagines along with it), so no model is needed to start. After each run a
regression model is trained on everything so far, and the assist drops run by
run so the participant's own decoding takes over.

Saves to logs/:
    <prefix>_pursuit_data.npz     EEG (causally filtered, same as live) + labels
    <prefix>_pursuit_frames.csv   one row per frame (cursor, target, labels, decoder)
    <prefix>_pursuit_config.json  all settings
    <prefix>_pursuit_model.pkl    the last regression model trained in the session

Uses the live decoding code in ../bci (mi_regression.py, dual_arrow_task.py,
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
import math
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
from dual_arrow_task import LiveDecoder  # noqa: E402
from lsl_utils import connect_eeg_stream  # noqa: E402
from mental_command_worker import resolve_channel_order  # noqa: E402
from mi_regression import MIRegressor, VelocityKalman, load_velocity_decoder, pursuit_regression_data  # noqa: E402


# ----------------------------------------------------------------
# Embedded config
# ----------------------------------------------------------------


@dataclass(frozen=True)
class StimConfig:
    # Different from the MI cue codes (1-4) so these recordings are never
    # mistaken for cued MI trials.
    run_start_code: int = 10
    run_end_code: int = 11
    target_on_code: int = 12
    rest_on_code: int = 13
    sync_code: int = 14


@dataclass(frozen=True)
class TaskConfig:
    n_runs: int = 8
    run_duration_s: float = 90.0
    # "2d" = left/right + forward, "1d" = left/right only.
    mode: str = "2d"
    # Assist per run (1.0 = cursor moves on its own, 0.0 = fully decoded).
    # Runs past the end of the list use the last value.
    assist_schedule: tuple[float, ...] = (1.0, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.2)

    rest_duration_s: float = 3.0
    max_reach_s: float = 8.0
    sync_every_s: float = 10.0

    cursor_speed: float = 0.15        # norm units per second at full velocity
    target_distance: float = 0.55
    target_radius: float = 0.07

    fullscreen: bool = False
    win_size: tuple[int, int] = (1200, 700)


@dataclass(frozen=True)
class DecoderConfig:
    update_interval_s: float = 0.1
    # Kalman smoothing of the decoded velocity.
    process_var: float = 0.02
    measurement_var: float = 0.3
    # Model retrained after each run on the pursuit data so far.
    retrain_after_each_run: bool = True
    ridge_alpha: float = 1.0
    # Label lag = brain delay + half the 1 s window; the training script searches it.
    label_lag_s: float = 1.0
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


class RecordingDecoder(LiveDecoder):
    """LiveDecoder that also keeps every filtered sample, so the session's EEG
    can be saved with the same filtering the live decoder used."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pull_s = 1.0  # pull generously so no samples are missed between frames
        self.saved_x: list[np.ndarray] = []
        self.saved_t: list[np.ndarray] = []

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
        self.saved_x.append(x)
        self.saved_t.append(ts)


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
    return f"{date_prefix}_{participant}_pursuit"


def intended_velocity(cursor: tuple[float, float], target: tuple[float, float] | None, mode: str) -> tuple[float, float]:
    """Unit direction from cursor to target. There is no 'backward' imagery,
    so the forward part is never negative. Zero when there is no target."""
    if target is None:
        return 0.0, 0.0
    dx, dy = target[0] - cursor[0], target[1] - cursor[1]
    if mode == "1d":
        return (float(np.sign(dx)), 0.0) if abs(dx) > 1e-6 else (0.0, 0.0)
    dy = max(dy, 0.0)
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return 0.0, 0.0
    return dx / norm, dy / norm


# ----------------------------------------------------------------
# Task
# ----------------------------------------------------------------


def run_task(session_name: str, port: str, task_cfg: TaskConfig, stim_cfg: StimConfig, ser_cfg: SerialConfig,
             dec_cfg: DecoderConfig, model_path: str | None = None) -> None:  # noqa: C901
    rng = random.Random()

    # Live EEG
    eeg_cfg = EEGConfig()
    live_cfg = dataclasses.replace(DualArrowTaskConfig(), live_update_interval_s=dec_cfg.update_interval_s)
    stream = connect_eeg_stream(LSLConfig(), bufsize=60.0)
    ch_names, missing = resolve_channel_order(list(stream.info["ch_names"]), eeg_cfg.picks)
    if missing:
        stream.disconnect()
        raise RuntimeError(f"Live stream is missing EEG channels {missing}. Available: {stream.info['ch_names']}")
    stream.pick(ch_names)
    sfreq = float(stream.info["sfreq"])
    decoder = RecordingDecoder(stream, len(ch_names), sfreq, eeg_cfg, live_cfg)

    model = load_velocity_decoder(model_path) if model_path else None
    if model is not None and model.channel_names and list(model.channel_names) != list(ch_names):
        stream.disconnect()
        raise RuntimeError(f"Model channels {model.channel_names} do not match the stream {ch_names}")
    kalman = VelocityKalman(dec_cfg.process_var, dec_cfg.measurement_var)

    # Session logs
    log_dir = Path(dec_cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    base = log_dir / session_name
    with open(f"{base}_config.json", "w", encoding="utf-8") as fh:
        json.dump({"session_name": session_name, "created_at": datetime.now().isoformat(timespec="seconds"),
                   "task_config": asdict(task_cfg), "decoder_config": asdict(dec_cfg),
                   "trigger_codes": asdict(stim_cfg), "eeg_channels": ch_names, "sfreq": sfreq,
                   "start_model": model_path}, fh, indent=2)
    frames_fh = open(f"{base}_frames.csv", "w", newline="")
    frames_log = csv.writer(frames_fh)
    frames_log.writerow(["time_lsl_s", "run", "trial", "phase", "assist", "cursor_x", "cursor_y", "target_x", "target_y",
                         "intended_vx", "intended_vy", "decoded_vx", "decoded_vy", "artifact", "model_version"])
    labels = {"t": [], "run": [], "vx": [], "vy": []}

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
    round_size = (float(win.size[1]) / float(win.size[0]), 1.0)  # keep circles round in norm units

    start_pos = (0.0, -0.1) if task_cfg.mode == "1d" else (0.0, -0.35)
    cue_text = visual.TextStim(win, text="", pos=(0, 0.35), height=0.08, color=white)
    status_text = visual.TextStim(win, text="", pos=(0, -0.45), height=0.05, color=white)
    start_marker = visual.Circle(win, radius=0.03, edges=48, pos=start_pos, fillColor=None, lineColor=white, lineWidth=1.5)
    target = visual.Circle(win, radius=task_cfg.target_radius, edges=48, pos=start_pos, fillColor=lit,
                           lineColor=white, lineWidth=2, opacity=0, name="target")
    cursor = visual.Circle(win, radius=0.035, edges=48, pos=start_pos, fillColor=white, lineColor=None, name="cursor")
    for stim in (start_marker, target, cursor):
        stim.size = round_size

    state = {"run": -1, "trial": -1, "phase": "setup", "assist": 1.0, "target": None,
             "cursor": start_pos, "decoded": (0.0, 0.0), "next_pred": 0.0, "last_frame": None}

    def draw_scene() -> None:
        start_marker.draw()
        if state["target"] is not None:
            target.draw()
        cursor.draw()
        cue_text.draw()
        status_text.draw()

    def update_decoder(now: float) -> None:
        decoder.poll()
        if now < state["next_pred"]:
            return
        state["next_pred"] = now + dec_cfg.update_interval_s
        if model is None or not model.is_fitted or not decoder.warmed_up:
            state["decoded"] = (0.0, 0.0)
            return
        window = decoder.buf[:, -decoder.window_n:]
        decoder.artifact = decoder.reject is not None and float(np.ptp(window, axis=-1).max()) > float(decoder.reject)
        if decoder.artifact:
            return  # hold the last estimate through artifacts
        state["decoded"] = kalman.update(model.predict_velocity(window))

    def frame(moving: bool) -> list[str]:
        """Poll EEG, move the cursor, log the frame, draw. Returns keys pressed."""
        keys = event.getKeys()
        if "escape" in keys:
            raise KeyboardInterrupt
        now = local_clock()
        dt = 0.0 if state["last_frame"] is None else min(now - state["last_frame"], 0.1)
        state["last_frame"] = now
        update_decoder(now)

        ivx, ivy = intended_velocity(state["cursor"], state["target"], task_cfg.mode)
        if moving:
            a = state["assist"]
            dvx, dvy = state["decoded"]
            vx = a * ivx + (1 - a) * dvx
            vy = 0.0 if task_cfg.mode == "1d" else a * ivy + (1 - a) * dvy
            cx = float(np.clip(state["cursor"][0] + vx * task_cfg.cursor_speed * dt, -0.95, 0.95))
            cy = float(np.clip(state["cursor"][1] + vy * task_cfg.cursor_speed * dt, -0.95, 0.95))
            state["cursor"] = (cx, cy)
            cursor.pos = state["cursor"]

        if state["run"] >= 0 and state["phase"] in ("rest", "target"):
            tx, ty = state["target"] if state["target"] is not None else ("", "")
            frames_log.writerow([f"{now:.4f}", state["run"], state["trial"], state["phase"], f"{state['assist']:.2f}",
                                 f"{state['cursor'][0]:.4f}", f"{state['cursor'][1]:.4f}", tx, ty,
                                 f"{ivx:.4f}", f"{ivy:.4f}", f"{state['decoded'][0]:.4f}", f"{state['decoded'][1]:.4f}",
                                 int(decoder.artifact), getattr(model, "version", 0)])
            labels["t"].append(now)
            labels["run"].append(state["run"])
            labels["vx"].append(ivx)
            labels["vy"].append(ivy)

        draw_scene()
        win.flip()
        return keys

    def wait_for_space() -> None:
        while "space" not in frame(moving=False):
            pass

    def new_target() -> tuple[float, float]:
        if task_cfg.mode == "1d":
            angle = rng.choice([0.0, math.pi])
        else:
            angle = rng.uniform(0.0, math.pi)  # right, up, left and everything between
        return (start_pos[0] + task_cfg.target_distance * math.cos(angle),
                start_pos[1] + task_cfg.target_distance * math.sin(angle))

    def save_data() -> None:
        if not decoder.saved_x or not labels["t"]:
            return
        np.savez_compressed(
            f"{base}_data.npz",
            eeg=np.concatenate(decoder.saved_x, axis=1).astype(np.float32),
            eeg_ts=np.concatenate(decoder.saved_t),
            ch_names=np.asarray(ch_names),
            sfreq=sfreq,
            label_t=np.asarray(labels["t"]),
            label_run=np.asarray(labels["run"], dtype=int),
            label_vx=np.asarray(labels["vx"]),
            label_vy=np.asarray(labels["vy"]),
            mode=task_cfg.mode,
        )

    def retrain() -> None:
        nonlocal model
        save_data()
        data = pursuit_regression_data(f"{base}_data.npz", live_cfg.window_s, live_cfg.window_step_s,
                                       dec_cfg.label_lag_s, eeg_cfg.reject_peak_to_peak)
        if len(data) < 20:
            return
        new_model = MIRegressor(dec_cfg.ridge_alpha, dec_cfg.label_lag_s, channel_names=ch_names)
        if isinstance(model, MIRegressor):
            new_model.version = model.version
        model = new_model.fit(data.X, data.Y)
        model.save(f"{base}_model.pkl")
        print(f"[MODEL] Regression v{model.version} trained on {len(data)} windows")

    print(f"[SESSION] {session_name}")
    print(f"[SERIAL] Sending triggers on {port} @ {ser_cfg.baudrate}")
    print(f"[EEG] {ch_names} @ {sfreq:.0f} Hz")

    runs_done = 0
    try:
        cue_text.text = "Connecting to EEG..."
        while not decoder.warmed_up:
            frame(moving=False)

        moves = "LEFT hand = left, RIGHT hand = right" + ("" if task_cfg.mode == "1d" else ", BOTH HANDS = up")
        cue_text.text = (
            "Motor Imagery Pursuit\n\n"
            "Move the white dot to the yellow target with motor imagery.\n"
            f"{moves}.\n"
            "Relax when it says REST.\n"
            "Keep your eyes on the white dot and your body still.\n\n"
            "Press SPACE to begin. ESC to quit."
        )
        wait_for_space()

        for run in range(task_cfg.n_runs):
            schedule = task_cfg.assist_schedule
            state["assist"] = float(schedule[min(run, len(schedule) - 1)])
            state.update(run=run, trial=-1)
            kalman.reset()

            cue_text.text = f"Run {run + 1}/{task_cfg.n_runs}"
            status_text.text = ("The dot moves on its own: imagine along with it" if state["assist"] >= 1.0
                                else f"Assist {100 * state['assist']:.0f}%")
            state["phase"] = "ready"
            wait_for_space()

            trig.pulse(stim_cfg.run_start_code)
            run_clock = core.Clock()
            next_sync = task_cfg.sync_every_s
            trial = 0
            while run_clock.getTime() < task_cfg.run_duration_s:
                # REST
                state.update(phase="rest", target=None, trial=trial)
                state["cursor"] = start_pos
                cursor.pos = start_pos
                cue_text.text = "REST"
                trig.pulse(stim_cfg.rest_on_code)
                phase_clock = core.Clock()
                while phase_clock.getTime() < task_cfg.rest_duration_s:
                    frame(moving=False)
                    if run_clock.getTime() >= next_sync:
                        trig.pulse(stim_cfg.sync_code)
                        next_sync += task_cfg.sync_every_s

                # TARGET
                state["target"] = new_target()
                state["phase"] = "target"
                target.pos = state["target"]
                target.opacity = 1
                cue_text.text = ""
                trig.pulse(stim_cfg.target_on_code)
                phase_clock = core.Clock()
                while phase_clock.getTime() < task_cfg.max_reach_s:
                    frame(moving=True)
                    if run_clock.getTime() >= next_sync:
                        trig.pulse(stim_cfg.sync_code)
                        next_sync += task_cfg.sync_every_s
                    if math.dist(state["cursor"], state["target"]) < task_cfg.target_radius:
                        break
                trial += 1
                status_text.text = f"Run {run + 1}/{task_cfg.n_runs}   targets: {trial}"

            trig.pulse(stim_cfg.run_end_code)
            state.update(phase="between", target=None)
            frames_fh.flush()
            runs_done = run + 1

            if dec_cfg.retrain_after_each_run and run + 1 < task_cfg.n_runs:
                cue_text.text = "Updating model..."
                status_text.text = ""
                draw_scene()
                win.flip()
                retrain()

            if run + 1 < task_cfg.n_runs:
                cue_text.text = "Break\n\nPress SPACE to continue"
                status_text.text = f"Completed {run + 1}/{task_cfg.n_runs} runs"
                wait_for_space()

        cue_text.text = "Session complete\n\nPress ESC to close"
        status_text.text = ""
        while True:
            frame(moving=False)

    except KeyboardInterrupt:
        print(f"\n[SESSION] Ended after {runs_done} runs")

    finally:
        save_data()
        frames_fh.close()
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
    parser = argparse.ArgumentParser(description="MI pursuit task for regression training data")
    parser.add_argument("--participant", type=str, default=None, help="Participant name for session label")
    parser.add_argument("--port", type=str, default=None, help="Serial port override (e.g., COM6 or /dev/ttyUSB0)")
    parser.add_argument("--runs", type=int, default=None, help="Override number of runs")
    parser.add_argument("--mode", choices=["1d", "2d"], default=None, help="1d = left/right only, 2d = adds forward")
    parser.add_argument("--model", type=str, default=None,
                        help="Start from a saved model: a regression model or the two-detector model (.pkl)")
    parser.add_argument("--fullscreen", action="store_true", help="Run fullscreen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    stim_cfg = StimConfig()
    task_cfg = TaskConfig(
        n_runs=args.runs if args.runs is not None else TaskConfig.n_runs,
        mode=args.mode or TaskConfig.mode,
        fullscreen=bool(args.fullscreen),
    )
    ser_cfg = SerialConfig()
    dec_cfg = DecoderConfig()

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
        dec_cfg=dec_cfg,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()