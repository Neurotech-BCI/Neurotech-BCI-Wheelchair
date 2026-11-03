"""Pygame-based ErrP driving task with trigger-hub support.

This protocol is built around precise command-execution events:

- The participant starts every trial in the centre lane.
- Each obstacle cluster blocks the centre lane plus one side lane.
- The participant presses LEFT or RIGHT to take the only open lane.
- On a subset of correct trials, the game intentionally executes the opposite
  lane to elicit an ErrP.
- Wrong-lane choices and missed responses are marked with separate triggers.

The EEG epoch of interest is centred on *command execution onset*, not on the
keypress. Triggers are sent when the state machine enters the execution
animation for the chosen command.

Recommended runtime dependencies:
    - pygame
    - pyserial

Example:
    python3 errpRacingGame.py --trials 120 --error-prob 0.2 --fullscreen
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import random
import threading
import time
from typing import Any

import pygame
import serial
import serial.tools.list_ports


LANE_LEFT = -1
LANE_CENTER = 0
LANE_RIGHT = 1

SAFE_LEFT = LANE_LEFT
SAFE_RIGHT = LANE_RIGHT


@dataclass(frozen=True)
class TriggerCodes:
    correct_left: int = 1
    correct_right: int = 2
    system_error_left: int = 3
    system_error_right: int = 4
    user_mistake_left: int = 5
    user_mistake_right: int = 6
    timeout_center: int = 7
    cue_safe_left: int = 8
    cue_safe_right: int = 9
    block_start: int = 10
    session_end: int = 11


@dataclass(frozen=True)
class SerialConfig:
    port: str | None = None
    baudrate: int = 115200
    pulse_width_s: float = 0.01
    vid: int = 0x2341
    pid: int = 0x8037
    auto_detect: bool = True
    enabled: bool = True


@dataclass(frozen=True)
class TaskConfig:
    trials: int = 250
    break_every: int = 25
    error_prob: float = 0.30
    window_width: int = 1440
    window_height: int = 900
    fullscreen: bool = False
    target_fps: int = 120
    seed: int | None = None
    lead_in_s: float = 0.40
    response_min_s: float = 1.05
    response_max_s: float = 1.30
    decision_depth: float = 0.76
    post_trial_min_s: float = 0.35
    post_trial_max_s: float = 0.60
    correct_points: int = 100
    system_error_points: int = -150
    user_error_points: int = -100
    timeout_points: int = -120
    action_duration_s: float = 0.88
    collision_hold_s: float = 0.18
    log_dir: str = "."


@dataclass
class TrialPlan:
    trial_index: int
    safe_lane: int
    response_window_s: float
    iti_s: float


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_in_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ease_out_cubic(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def lane_name(lane: int) -> str:
    if lane == LANE_LEFT:
        return "left"
    if lane == LANE_RIGHT:
        return "right"
    return "center"


class TriggerHub:
    """Optional serial trigger sender that degrades to simulation mode."""

    def __init__(self, cfg: SerialConfig):
        self.cfg = cfg
        self.ser: serial.Serial | None = None
        self.port_used: str | None = None
        self._lock = threading.Lock()

    def open(self) -> None:
        if not self.cfg.enabled:
            print("[TRIGGER] Disabled; running in simulation mode.")
            return

        port = self.cfg.port
        if port is None and self.cfg.auto_detect:
            for info in serial.tools.list_ports.comports():
                if info.vid == self.cfg.vid and info.pid == self.cfg.pid:
                    port = info.device
                    break

        if port is None:
            print("[TRIGGER] No trigger hub detected; running in simulation mode.")
            return

        try:
            self.ser = serial.Serial(port, self.cfg.baudrate, timeout=0)
            time.sleep(0.05)
        except Exception as exc:
            self.ser = None
            print(f"[TRIGGER] Failed to open {port}: {exc}. Running in simulation mode.")
            return

        self.port_used = port
        print(f"[TRIGGER] Connected on {port} @ {self.cfg.baudrate} baud.")

    def close(self) -> None:
        with self._lock:
            if self.ser is not None:
                try:
                    self.ser.close()
                finally:
                    self.ser = None

    def pulse(self, code: int, wait: bool = False) -> None:
        if self.ser is None:
            return
        code = int(code) & 0xFF
        if wait:
            self._pulse_blocking(code)
            return
        thread = threading.Thread(target=self._pulse_blocking, args=(code,), daemon=True)
        thread.start()

    def _pulse_blocking(self, code: int) -> None:
        with self._lock:
            if self.ser is None:
                return
            self.ser.write(bytes([code]))
            self.ser.flush()
            time.sleep(self.cfg.pulse_width_s)
            self.ser.write(bytes([0]))
            self.ser.flush()


class BalancedLaneScheduler:
    def __init__(self, rng: random.Random, block_size: int = 8):
        self.rng = rng
        self.block_size = max(2, int(block_size))
        self._bag: list[int] = []

    def next_safe_lane(self) -> int:
        if not self._bag:
            n_left = self.block_size // 2
            n_right = self.block_size - n_left
            self._bag = [SAFE_LEFT] * n_left + [SAFE_RIGHT] * n_right
            self.rng.shuffle(self._bag)
        return self._bag.pop()


class SessionLogger:
    def __init__(self, log_dir: Path, session_name: str, config: TaskConfig, triggers: TriggerCodes):
        self.log_dir = log_dir
        self.session_name = session_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trial_path = self.log_dir / f"{session_name}_trials.csv"
        self.meta_path = self.log_dir / f"{session_name}_config.json"
        self._trial_file = self.trial_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._trial_file,
            fieldnames=[
                "session",
                "trial_index",
                "safe_lane",
                "blocked_side_lane",
                "response_window_s",
                "user_input_lane",
                "user_input_rt_ms",
                "execution_onset_perf_s",
                "execution_onset_monotonic_ns",
                "outcome",
                "trigger_code",
                "trigger_name",
                "executed_lane",
                "system_error_applied",
                "collision",
                "score_before",
                "score_after",
                "streak_after",
                "command_to_collision_ms",
                "lead_in_s",
                "post_trial_iti_s",
            ],
        )
        self._writer.writeheader()

        with self.meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(
                {
                    "session_name": session_name,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "task_config": asdict(config),
                    "trigger_codes": asdict(triggers),
                },
                meta_file,
                indent=2,
            )

    def write_trial(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._trial_file.flush()

    def close(self) -> None:
        self._trial_file.close()


class RacingErrPGame:
    def __init__(self, cfg: TaskConfig, serial_cfg: SerialConfig, session_name: str):
        self.cfg = cfg
        self.serial_cfg = serial_cfg
        self.session_name = session_name
        self.triggers = TriggerCodes()
        self.rng = random.Random(cfg.seed)
        self.scheduler = BalancedLaneScheduler(self.rng)
        self.trigger_hub = TriggerHub(serial_cfg)
        self.logger = SessionLogger(Path(cfg.log_dir), session_name, cfg, self.triggers)

        self.width = cfg.window_width
        self.height = cfg.window_height
        self.horizon_y = 150
        self.bottom_y = self.height - 60
        self.road_half_top = 150
        self.road_half_bottom = 530
        self.car_y = self.height - 165

        pygame.init()
        pygame.font.init()
        flags = pygame.DOUBLEBUF
        if cfg.fullscreen:
            flags |= pygame.FULLSCREEN
        try:
            self.screen = pygame.display.set_mode((self.width, self.height), flags, vsync=1)
        except TypeError:
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("ErrP Highway")
        self.clock = pygame.time.Clock()

        self.font_title = pygame.font.SysFont("arial", 36, bold=True)
        self.font_hud = pygame.font.SysFont("arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("arial", 12, bold=True)
        self.font_body = pygame.font.SysFont("arial", 19)
        self.font_overlay = pygame.font.SysFont("arial", 24)
        self.font_countdown = pygame.font.SysFont("arial", 100, bold=True)

        self.background_surface = self._build_background()
        self.overlay_surface = self._build_overlay()
        self.flash_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        self.running = True
        self.closed = False
        self.state = "intro"
        self.last_frame_time = time.perf_counter()
        self.state_started_at = self.last_frame_time
        self.phase_started_at = self.last_frame_time
        self.last_frame_now = self.last_frame_time
        self.world_scroll = 0.0

        self.score = 0
        self.streak = 0
        self.completed_trials = 0
        self.pending_input: tuple[int, float, float] | None = None
        self.pending_space = False
        self.current_trial: TrialPlan | None = None
        self.current_obstacle_depth = 0.0
        self.obstacle_depth_on_action = 0.0

        self.camera_shake = 0.0
        self.flash_alpha = 0.0
        self.feedback_text = ""
        self.feedback_color = (255, 255, 255)
        self.feedback_until = 0.0

        self.action_outcome = ""
        self.action_trigger_code = 0
        self.action_trigger_name = ""
        self.action_user_lane: int | None = None
        self.action_user_rt_ms: float | None = None
        self.action_executed_lane = 0
        self.action_collision = False
        self.action_system_error = False
        self.action_onset_s = 0.0
        self.action_onset_ns = 0
        self.action_collision_offset_ms = 0.0
        self.score_before_action = 0
        self.post_trial_deadline = 0.0

        self.trigger_hub.open()

    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.target_fps) / 1000.0
                now = time.perf_counter()
                self.last_frame_now = now
                self.world_scroll += dt * 1.6

                self._process_events(now)
                self._update(now, dt)
                self._render(now)
                pygame.display.flip()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.running = False
        try:
            self.trigger_hub.pulse(self.triggers.session_end, wait=True)
        except Exception:
            pass
        self.trigger_hub.close()
        self.logger.close()
        pygame.quit()

    def _build_background(self) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height))
        for y in range(self.height):
            t = y / max(1, self.height - 1)
            if y < self.horizon_y + 100:
                r = int(8 + 40 * (1 - t))
                g = int(22 + 80 * (1 - t))
                b = int(34 + 110 * (1 - t))
            else:
                r = int(7 + 8 * (1 - t))
                g = int(18 + 16 * (1 - t))
                b = int(24 + 18 * (1 - t))
            pygame.draw.line(surface, (r, g, b), (0, y), (self.width, y))

        glow_center = (self.width // 2, self.horizon_y - 25)
        for radius in range(220, 20, -10):
            alpha = int(10 + 40 * (1 - radius / 220))
            glow = pygame.Surface((radius * 2, int(radius * 1.1)), pygame.SRCALPHA)
            pygame.draw.ellipse(glow, (250, 170, 70, alpha), glow.get_rect())
            surface.blit(glow, (glow_center[0] - radius, glow_center[1] - radius * 0.55))

        star_rng = random.Random(7)
        for _ in range(80):
            x = star_rng.randint(0, self.width - 1)
            y = star_rng.randint(0, self.horizon_y)
            shade = star_rng.randint(160, 240)
            surface.fill((shade, shade, shade), (x, y, 2, 2))
        return surface.convert()

    def _build_overlay(self) -> pygame.Surface:
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.rect(overlay, (0, 0, 0, 115), pygame.Rect(0, 0, self.width, 120))
        pygame.draw.rect(overlay, (0, 0, 0, 95), pygame.Rect(0, self.height - 110, self.width, 110))
        return overlay

    def _process_events(self, now: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                self.running = False
                return

            if event.key == pygame.K_SPACE:
                self.pending_space = True
                continue

            if event.key not in (pygame.K_LEFT, pygame.K_RIGHT):
                continue

            if self.state != "waiting_input" or self.current_trial is None or self.pending_input is not None:
                continue

            user_lane = LANE_LEFT if event.key == pygame.K_LEFT else LANE_RIGHT
            rt_ms = (now - self.phase_started_at) * 1000.0
            self.pending_input = (user_lane, now, rt_ms)

    def _update(self, now: float, dt: float) -> None:
        self.camera_shake *= 0.88
        self.flash_alpha = max(0.0, self.flash_alpha - dt * 1.8)

        if self.state == "intro":
            if self.pending_space:
                self.pending_space = False
                self._transition("countdown", now)
            return

        if self.state == "countdown":
            if now - self.state_started_at >= 3.4:
                self._begin_next_trial(now)
            return

        if self.state == "waiting_input":
            self._update_waiting(now)
            return

        if self.state == "action":
            self._update_action(now)
            return

        if self.state == "intertrial":
            if now >= self.post_trial_deadline:
                self._after_trial(now)
            return

        if self.state == "break":
            if self.pending_space:
                self.pending_space = False
                self._begin_next_trial(now)
            return

        if self.state == "complete":
            if self.pending_space:
                self.pending_space = False
                self.running = False

    def _transition(self, state: str, now: float) -> None:
        self.state = state
        self.state_started_at = now

    def _begin_next_trial(self, now: float) -> None:
        if self.completed_trials >= self.cfg.trials:
            self._transition("complete", now)
            return

        safe_lane = self.scheduler.next_safe_lane()
        response_window = self.rng.uniform(self.cfg.response_min_s, self.cfg.response_max_s)
        iti = self.rng.uniform(self.cfg.post_trial_min_s, self.cfg.post_trial_max_s)
        self.current_trial = TrialPlan(
            trial_index=self.completed_trials + 1,
            safe_lane=safe_lane,
            response_window_s=response_window,
            iti_s=iti,
        )
        self.pending_input = None
        self.current_obstacle_depth = 0.0
        self.feedback_text = ""
        self.action_outcome = ""
        self.action_collision = False
        self.action_system_error = False
        self.trigger_hub.pulse(
            self.triggers.cue_safe_left if safe_lane == SAFE_LEFT else self.triggers.cue_safe_right
        )
        self._transition("waiting_input", now)
        self.phase_started_at = now

    def _update_waiting(self, now: float) -> None:
        assert self.current_trial is not None

        trial_elapsed = now - self.phase_started_at
        if trial_elapsed < self.cfg.lead_in_s:
            self.current_obstacle_depth = 0.0
            return

        travel_elapsed = trial_elapsed - self.cfg.lead_in_s
        travel_fraction = clamp(travel_elapsed / self.current_trial.response_window_s, 0.0, 1.0)
        self.current_obstacle_depth = self.cfg.decision_depth * ease_in_out(travel_fraction)

        if self.pending_input is not None:
            user_lane, _, rt_ms = self.pending_input
            self.pending_input = None
            self._begin_action(now, user_lane=user_lane, user_rt_ms=rt_ms)
            return

        if travel_elapsed >= self.current_trial.response_window_s:
            self._begin_action(now, user_lane=None, user_rt_ms=None)

    def _begin_action(self, now: float, user_lane: int | None, user_rt_ms: float | None) -> None:
        assert self.current_trial is not None

        safe_lane = self.current_trial.safe_lane
        self.score_before_action = self.score
        self.action_user_lane = user_lane
        self.action_user_rt_ms = user_rt_ms
        self.action_system_error = False
        self.action_collision = False
        self.action_executed_lane = LANE_CENTER
        self.action_collision_offset_ms = 0.0

        if user_lane is None:
            self.action_outcome = "timeout"
            self.action_trigger_code = self.triggers.timeout_center
            self.action_trigger_name = "timeout_center"
            self.action_executed_lane = LANE_CENTER
            self.action_collision = True
            self.action_collision_offset_ms = 160.0
            self.score += self.cfg.timeout_points
            self.streak = 0
            self.feedback_text = "Too late"
            self.feedback_color = (255, 209, 102)
        elif user_lane != safe_lane:
            self.action_outcome = "user_mistake"
            self.action_trigger_code = (
                self.triggers.user_mistake_left if user_lane == LANE_LEFT else self.triggers.user_mistake_right
            )
            self.action_trigger_name = (
                "user_mistake_left" if user_lane == LANE_LEFT else "user_mistake_right"
            )
            self.action_executed_lane = user_lane
            self.action_collision = True
            self.action_collision_offset_ms = 290.0
            self.score += self.cfg.user_error_points
            self.streak = 0
            self.feedback_text = "Driver error"
            self.feedback_color = (255, 140, 105)
        else:
            if self.rng.random() < self.cfg.error_prob:
                self.action_outcome = "system_error"
                self.action_system_error = True
                self.action_executed_lane = -safe_lane
                self.action_trigger_code = (
                    self.triggers.system_error_left
                    if self.action_executed_lane == LANE_LEFT
                    else self.triggers.system_error_right
                )
                self.action_trigger_name = (
                    "system_error_left"
                    if self.action_executed_lane == LANE_LEFT
                    else "system_error_right"
                )
                self.action_collision = True
                self.action_collision_offset_ms = 300.0
                self.score += self.cfg.system_error_points
                self.streak = 0
                self.feedback_text = "Autopilot glitch"
                self.feedback_color = (255, 93, 143)
            else:
                self.action_outcome = "correct"
                self.action_executed_lane = safe_lane
                self.action_trigger_code = (
                    self.triggers.correct_left if safe_lane == LANE_LEFT else self.triggers.correct_right
                )
                self.action_trigger_name = "correct_left" if safe_lane == LANE_LEFT else "correct_right"
                self.action_collision = False
                self.action_collision_offset_ms = 0.0
                self.score += self.cfg.correct_points
                self.streak += 1
                self.feedback_text = "Perfect dodge"
                self.feedback_color = (109, 227, 181)

        self.action_onset_s = now
        self.action_onset_ns = time.monotonic_ns()
        self.obstacle_depth_on_action = self.current_obstacle_depth
        self.feedback_until = now + 0.9
        self.trigger_hub.pulse(self.action_trigger_code)
        self._transition("action", now)

    def _update_action(self, now: float) -> None:
        elapsed = now - self.action_onset_s
        action_t = clamp(elapsed / self.cfg.action_duration_s, 0.0, 1.0)
        self.current_obstacle_depth = lerp(self.obstacle_depth_on_action, 1.08, ease_out_cubic(action_t))

        if self.action_collision:
            self.camera_shake = max(self.camera_shake, 10.0 * (1.0 - action_t))
            self.flash_alpha = max(self.flash_alpha, 0.28 * (1.0 - action_t))

        if elapsed >= self.cfg.action_duration_s:
            self.post_trial_deadline = now + (self.current_trial.iti_s if self.current_trial is not None else 0.4)
            self._transition("intertrial", now)

    def _after_trial(self, now: float) -> None:
        assert self.current_trial is not None

        self.logger.write_trial(
            {
                "session": self.session_name,
                "trial_index": self.current_trial.trial_index,
                "safe_lane": lane_name(self.current_trial.safe_lane),
                "blocked_side_lane": lane_name(-self.current_trial.safe_lane),
                "response_window_s": f"{self.current_trial.response_window_s:.4f}",
                "user_input_lane": "" if self.action_user_lane is None else lane_name(self.action_user_lane),
                "user_input_rt_ms": "" if self.action_user_rt_ms is None else f"{self.action_user_rt_ms:.3f}",
                "execution_onset_perf_s": f"{self.action_onset_s:.6f}",
                "execution_onset_monotonic_ns": self.action_onset_ns,
                "outcome": self.action_outcome,
                "trigger_code": self.action_trigger_code,
                "trigger_name": self.action_trigger_name,
                "executed_lane": lane_name(self.action_executed_lane),
                "system_error_applied": int(self.action_system_error),
                "collision": int(self.action_collision),
                "score_before": self.score_before_action,
                "score_after": self.score,
                "streak_after": self.streak,
                "command_to_collision_ms": f"{self.action_collision_offset_ms:.3f}",
                "lead_in_s": f"{self.cfg.lead_in_s:.4f}",
                "post_trial_iti_s": f"{self.current_trial.iti_s:.4f}",
            }
        )
        self.completed_trials += 1

        if self.completed_trials >= self.cfg.trials:
            self._transition("complete", now)
            return

        if self.cfg.break_every > 0 and self.completed_trials % self.cfg.break_every == 0:
            self.trigger_hub.pulse(self.triggers.block_start)
            self._transition("break", now)
            return

        self._begin_next_trial(now)

    def _road_bounds(self, depth: float) -> tuple[float, float, float]:
        depth = clamp(depth, 0.0, 1.0)
        y = lerp(self.horizon_y, self.bottom_y, depth)
        half_w = lerp(self.road_half_top, self.road_half_bottom, depth)
        center_x = self.width / 2
        return center_x, y, half_w

    def _lane_screen_x(self, lane_position: float, depth: float) -> float:
        center_x, _, half_w = self._road_bounds(depth)
        return center_x + lane_position * half_w * 0.62

    def _car_lane_position(self, now: float) -> float:
        if self.state != "action":
            return 0.0

        elapsed = now - self.action_onset_s
        total = self.cfg.action_duration_s
        impact_t = self.action_collision_offset_ms / 1000.0 if self.action_collision else 0.34
        return_start = impact_t + self.cfg.collision_hold_s if self.action_collision else 0.48

        if self.action_executed_lane == LANE_CENTER:
            return 0.0
        if elapsed <= impact_t:
            return self.action_executed_lane * ease_out_cubic(elapsed / max(impact_t, 1e-6))
        if elapsed <= return_start:
            return float(self.action_executed_lane)
        return self.action_executed_lane * (
            1.0 - ease_in_out((elapsed - return_start) / max(total - return_start, 1e-6))
        )

    def _draw_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: float,
        y: float,
        anchor: str = "topleft",
    ) -> None:
        surface = font.render(text, True, color)
        rect = surface.get_rect()
        if anchor == "topleft":
            rect.topleft = (x, y)
        elif anchor == "topright":
            rect.topright = (x, y)
        elif anchor == "midtop":
            rect.midtop = (x, y)
        elif anchor == "center":
            rect.center = (x, y)
        elif anchor == "midbottom":
            rect.midbottom = (x, y)
        self.screen.blit(surface, rect)

    def _draw_multiline(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        x: float,
        y: float,
        line_gap: int = 8,
        anchor: str = "midtop",
    ) -> None:
        lines = text.splitlines()
        rendered = [font.render(line, True, color) for line in lines]
        total_h = sum(s.get_height() for s in rendered) + max(0, len(rendered) - 1) * line_gap
        current_y = y
        if anchor == "center":
            current_y = y - total_h / 2
        elif anchor == "midbottom":
            current_y = y - total_h

        for surf in rendered:
            rect = surf.get_rect()
            if anchor in {"midtop", "center", "midbottom"}:
                rect.midtop = (x, current_y)
            else:
                rect.topleft = (x, current_y)
            self.screen.blit(surf, rect)
            current_y += surf.get_height() + line_gap

    def _render(self, now: float) -> None:
        self.screen.blit(self.background_surface, (0, 0))

        shake_x = self.rng.uniform(-self.camera_shake, self.camera_shake)
        shake_y = self.rng.uniform(-self.camera_shake * 0.3, self.camera_shake * 0.3)

        self._draw_world(now, shake_x, shake_y)
        self.screen.blit(self.overlay_surface, (0, 0))
        self._draw_hud(now)
        self._draw_state_overlay(now)

        if self.flash_alpha > 0.0:
            self.flash_surface.fill((255, 207, 122, int(clamp(self.flash_alpha, 0.0, 1.0) * 255)))
            self.screen.blit(self.flash_surface, (0, 0))

    def _draw_world(self, now: float, shake_x: float, shake_y: float) -> None:
        center = self.width / 2 + shake_x
        top_left = (center - self.road_half_top, self.horizon_y + shake_y)
        top_right = (center + self.road_half_top, self.horizon_y + shake_y)
        bottom_right = (center + self.road_half_bottom, self.bottom_y + shake_y)
        bottom_left = (center - self.road_half_bottom, self.bottom_y + shake_y)

        pygame.draw.polygon(self.screen, (29, 34, 40), [top_left, top_right, bottom_right, bottom_left])
        pygame.draw.lines(self.screen, (84, 97, 107), True, [top_left, top_right, bottom_right, bottom_left], 2)

        shoulder_left = [
            (center - self.road_half_top - 24, self.horizon_y + shake_y),
            (center - self.road_half_top - 40, self.horizon_y + shake_y),
            (center - self.road_half_bottom - 70, self.bottom_y + shake_y),
            (center - self.road_half_bottom - 38, self.bottom_y + shake_y),
        ]
        shoulder_right = [
            (center + self.road_half_top + 24, self.horizon_y + shake_y),
            (center + self.road_half_top + 40, self.horizon_y + shake_y),
            (center + self.road_half_bottom + 70, self.bottom_y + shake_y),
            (center + self.road_half_bottom + 38, self.bottom_y + shake_y),
        ]
        pygame.draw.polygon(self.screen, (59, 68, 77), shoulder_left)
        pygame.draw.polygon(self.screen, (59, 68, 77), shoulder_right)

        dash_period = 0.18
        dash_len = 0.07
        dash_offset = (self.world_scroll * 0.9) % dash_period
        for lane_divider in (-0.5, 0.5):
            z = -dash_offset
            while z < 1.15:
                z0 = clamp(z, 0.0, 1.0)
                z1 = clamp(z + dash_len, 0.0, 1.0)
                if z1 <= 0:
                    z += dash_period
                    continue
                x0 = self._lane_screen_x(lane_divider, z0) + shake_x
                x1 = self._lane_screen_x(lane_divider, z1) + shake_x
                _, y0, hw0 = self._road_bounds(z0)
                _, y1, hw1 = self._road_bounds(z1)
                w0 = max(2.0, hw0 * 0.018)
                w1 = max(2.0, hw1 * 0.018)
                pygame.draw.polygon(
                    self.screen,
                    (245, 240, 213),
                    [
                        (x0 - w0, y0 + shake_y),
                        (x0 + w0, y0 + shake_y),
                        (x1 + w1, y1 + shake_y),
                        (x1 - w1, y1 + shake_y),
                    ],
                )
                z += dash_period

        pygame.draw.line(
            self.screen,
            (45, 52, 58),
            (center, self.horizon_y + shake_y),
            (center, self.bottom_y + shake_y),
            2,
        )

        if self.current_trial is not None and self.state in {"waiting_input", "action"}:
            self._draw_obstacles(shake_x, shake_y)

        self._draw_roadside_lights(shake_x, shake_y)
        self._draw_car(now, shake_x, shake_y)

    def _draw_roadside_lights(self, shake_x: float, shake_y: float) -> None:
        for idx in range(12):
            z = idx / 11
            _, y, half_w = self._road_bounds(z)
            left_x = self.width / 2 - half_w - 85 + shake_x
            right_x = self.width / 2 + half_w + 85 + shake_x
            pole_h = lerp(18, 80, z)
            lamp_r = lerp(2, 10, z)
            tint = 140 + int(50 * z)
            for x in (left_x, right_x):
                pygame.draw.line(self.screen, (112, 121, 129), (x, y + shake_y), (x, y - pole_h + shake_y), 2)
                pygame.draw.circle(self.screen, (tint, int(tint * 0.85), 85), (int(x), int(y - pole_h + shake_y)), int(lamp_r))

    def _draw_obstacles(self, shake_x: float, shake_y: float) -> None:
        assert self.current_trial is not None
        depth = clamp(self.current_obstacle_depth, 0.02, 1.10)
        blocked_lanes = [LANE_CENTER, -self.current_trial.safe_lane]

        for lane in blocked_lanes:
            x = self._lane_screen_x(lane, depth) + shake_x
            _, y, half_w = self._road_bounds(depth)
            barrier_w = max(44.0, half_w * 0.43)
            barrier_h = max(28.0, half_w * 0.18)
            rect = pygame.Rect(0, 0, int(barrier_w), int(barrier_h))
            rect.center = (int(x), int(y + shake_y))

            pygame.draw.rect(self.screen, (215, 117, 54), rect, border_radius=8)
            pygame.draw.rect(self.screen, (247, 230, 181), rect, width=3, border_radius=8)

            stripe_h = rect.height / 4
            stripe_colors = [(248, 245, 234), (239, 92, 67), (248, 245, 234)]
            for idx, color in enumerate(stripe_colors):
                stripe = pygame.Rect(rect.left + 4, rect.top + idx * stripe_h, rect.width - 8, stripe_h - 2)
                pygame.draw.rect(self.screen, color, stripe)

            lamp_r = max(4.0, half_w * 0.03)
            pygame.draw.circle(self.screen, (255, 209, 102), (int(x - barrier_w * 0.28), int(rect.top - barrier_h * 0.18)), int(lamp_r))
            pygame.draw.circle(self.screen, (255, 209, 102), (int(x + barrier_w * 0.28), int(rect.top - barrier_h * 0.18)), int(lamp_r))

            for sign in (-1, 1):
                leg_x = x + sign * barrier_w * 0.22
                leg_y0 = rect.bottom
                leg_y1 = rect.bottom + barrier_h * 0.45
                pygame.draw.line(
                    self.screen,
                    (170, 182, 191),
                    (leg_x, leg_y0),
                    (leg_x - sign * barrier_w * 0.08, leg_y1),
                    4,
                )

    def _draw_car(self, now: float, shake_x: float, shake_y: float) -> None:
        lane_position = self._car_lane_position(now)
        car_x = self._lane_screen_x(lane_position, 1.0) + shake_x
        car_y = self.car_y + shake_y
        wobble = 0.0
        if self.state == "action" and self.action_collision:
            wobble = math.sin((now - self.action_onset_s) * 42.0) * self.camera_shake * 0.8
        car_x += wobble

        body_w = 132
        body_h = 210
        shadow_rect = pygame.Rect(0, 0, 160, 38)
        shadow_rect.center = (int(car_x), int(car_y + body_h / 2 + 10))
        shadow = pygame.Surface((shadow_rect.width, shadow_rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 130), shadow.get_rect())
        self.screen.blit(shadow, shadow_rect)

        wheel_offset_x = 52
        for sign in (-1, 1):
            upper = pygame.Rect(int(car_x + sign * wheel_offset_x - 13), int(car_y - 58), 26, 48)
            lower = pygame.Rect(int(car_x + sign * wheel_offset_x - 13), int(car_y + 18), 26, 48)
            pygame.draw.rect(self.screen, (16, 18, 21), upper, border_radius=6)
            pygame.draw.rect(self.screen, (16, 18, 21), lower, border_radius=6)
            pygame.draw.rect(self.screen, (46, 49, 54), upper, width=2, border_radius=6)
            pygame.draw.rect(self.screen, (46, 49, 54), lower, width=2, border_radius=6)

        body_points = [
            (car_x - body_w * 0.35, car_y - body_h * 0.48),
            (car_x + body_w * 0.35, car_y - body_h * 0.48),
            (car_x + body_w * 0.48, car_y - body_h * 0.16),
            (car_x + body_w * 0.42, car_y + body_h * 0.36),
            (car_x + body_w * 0.18, car_y + body_h * 0.50),
            (car_x - body_w * 0.18, car_y + body_h * 0.50),
            (car_x - body_w * 0.42, car_y + body_h * 0.36),
            (car_x - body_w * 0.48, car_y - body_h * 0.16),
        ]
        pygame.draw.polygon(self.screen, (200, 46, 58), body_points)
        pygame.draw.lines(self.screen, (240, 215, 215), True, body_points, 3)

        windshield = [
            (car_x - body_w * 0.22, car_y - body_h * 0.28),
            (car_x + body_w * 0.22, car_y - body_h * 0.28),
            (car_x + body_w * 0.17, car_y + body_h * 0.08),
            (car_x - body_w * 0.17, car_y + body_h * 0.08),
        ]
        pygame.draw.polygon(self.screen, (134, 211, 234), windshield)
        pygame.draw.lines(self.screen, (215, 245, 251), True, windshield, 2)
        pygame.draw.rect(self.screen, (145, 29, 40), pygame.Rect(int(car_x - body_w * 0.09), int(car_y + body_h * 0.14), int(body_w * 0.18), int(body_h * 0.21)))

        light_y = int(car_y - body_h * 0.40)
        for sign in (-1, 1):
            pygame.draw.ellipse(
                self.screen,
                (255, 246, 191),
                pygame.Rect(int(car_x + sign * body_w * 0.23 - 10), light_y - 8, 20, 16),
            )

        if self.state == "action" and self.action_collision:
            impact_t = self.action_collision_offset_ms / 1000.0
            elapsed = now - self.action_onset_s
            if elapsed >= impact_t:
                self._draw_explosion(car_x, car_y - 10, elapsed - impact_t)

    def _draw_explosion(self, x: float, y: float, elapsed: float) -> None:
        progress = clamp(elapsed / 0.32, 0.0, 1.0)
        radius = 28 + progress * 84
        colors = [(255, 243, 173), (255, 179, 71), (255, 112, 67), (230, 57, 70)]
        for idx, color in enumerate(colors):
            r = radius * (1.0 - idx * 0.18)
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(r))

        for idx in range(12):
            angle = idx * (math.pi * 2.0 / 12.0) + elapsed * 12.0
            spoke_len = radius * 1.2
            x1 = x + math.cos(angle) * radius * 0.3
            y1 = y + math.sin(angle) * radius * 0.3
            x2 = x + math.cos(angle) * spoke_len
            y2 = y + math.sin(angle) * spoke_len
            pygame.draw.line(self.screen, (255, 224, 102), (x1, y1), (x2, y2), 4)

    def _draw_hud(self, now: float) -> None:
        trial_number = self.completed_trials if self.state == "complete" else min(self.cfg.trials, self.completed_trials + 1)
        trial_label = f"TRIAL {trial_number:03d}/{self.cfg.trials:03d}"
        self._draw_text(trial_label, self.font_hud, (216, 229, 239), 40, 28)
        self._draw_text("ERRP DRIVING PROTOCOL", self.font_small, (143, 184, 204), 40, 62)

        self._draw_text(f"SCORE {self.score:+05d}", self.font_hud, (240, 244, 247), self.width - 40, 28, anchor="topright")
        self._draw_text(f"STREAK {self.streak:02d}", self.font_small, (125, 223, 183), self.width - 40, 62, anchor="topright")

        if self.current_trial is not None and self.state in {"waiting_input", "action"}:
            safe_text = "OPEN LANE  LEFT" if self.current_trial.safe_lane == SAFE_LEFT else "OPEN LANE  RIGHT"
            self._draw_text(safe_text, self.font_body, (245, 242, 208), self.width / 2, 50, anchor="midtop")

        if now < self.feedback_until and self.feedback_text:
            self._draw_text(self.feedback_text.upper(), self.font_hud, self.feedback_color, self.width / 2, self.height - 70, anchor="midbottom")

    def _draw_panel(self, rect: pygame.Rect, fill: tuple[int, int, int, int], border: tuple[int, int, int]) -> None:
        panel = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(panel, fill, panel.get_rect(), border_radius=20)
        pygame.draw.rect(panel, border, panel.get_rect(), width=2, border_radius=20)
        self.screen.blit(panel, rect)

    def _draw_state_overlay(self, now: float) -> None:
        if self.state == "intro":
            rect = pygame.Rect(150, 120, self.width - 300, self.height - 240)
            self._draw_panel(rect, (5, 17, 27, 220), (142, 202, 230))
            body = (
                "You start in the center lane.\n"
                "Each barrier blocks the center lane and one side lane.\n"
                "Press LEFT or RIGHT to take the only open lane.\n\n"
                "Most correct responses execute normally.\n"
                "On some correct trials the system will execute the opposite lane.\n"
                "That system-induced crash is the ErrP event of interest.\n\n"
                "If you choose the blocked lane, the game obeys you and marks a user mistake.\n"
                "If you respond too late, the center-lane collision is marked separately.\n\n"
                "Press SPACE to start. ESC to quit."
            )
            self._draw_text("ERRP HIGHWAY", self.font_title, (232, 241, 247), rect.centerx, rect.top + 52, anchor="midtop")
            self._draw_multiline(body, self.font_body, (196, 214, 223), rect.centerx, rect.top + 128, line_gap=8, anchor="midtop")
            return

        if self.state == "countdown":
            elapsed = now - self.state_started_at
            if elapsed < 1.0:
                text = "3"
            elif elapsed < 2.0:
                text = "2"
            elif elapsed < 3.0:
                text = "1"
            else:
                text = "GO"
            self._draw_text(text, self.font_countdown, (255, 244, 196), self.width / 2, self.height / 2 - 10, anchor="center")
            return

        if self.state == "break":
            rect = pygame.Rect(210, 200, self.width - 420, self.height - 400)
            self._draw_panel(rect, (4, 16, 25, 225), (123, 223, 242))
            body = (
                f"Block complete\n\n"
                f"Trials finished: {self.completed_trials}/{self.cfg.trials}\n"
                f"Current score: {self.score:+05d}\n"
                f"Current streak: {self.streak:02d}\n\n"
                "Blink, relax your hands, and reset posture.\n"
                "Press SPACE when ready for the next block."
            )
            self._draw_multiline(body, self.font_overlay, (215, 230, 238), rect.centerx, rect.centery, line_gap=10, anchor="center")
            return

        if self.state == "complete":
            rect = pygame.Rect(200, 185, self.width - 400, self.height - 370)
            self._draw_panel(rect, (5, 17, 27, 225), (152, 245, 225))
            body = (
                f"Session complete\n\n"
                f"Trials finished: {self.completed_trials}/{self.cfg.trials}\n"
                f"Final score: {self.score:+05d}\n"
                f"Trial log: {self.logger.trial_path}\n"
                f"Config log: {self.logger.meta_path}\n\n"
                "Press SPACE to close."
            )
            self._draw_multiline(body, self.font_overlay, (231, 245, 239), rect.centerx, rect.centery, line_gap=10, anchor="center")
            return

        if self.state == "waiting_input" and self.current_trial is not None:
            elapsed = now - self.phase_started_at
            if elapsed < self.cfg.lead_in_s:
                self._draw_text("READY", self.font_body, (205, 232, 246), self.width / 2, self.horizon_y + 20, anchor="midtop")



def build_session_name() -> str:
    return datetime.now().strftime("errp_racing_%Y%m%d_%H%M%S")


def main() -> None:
    cfg = TaskConfig()
    serial_cfg = SerialConfig()
    session_name = build_session_name()

    print(f"[SESSION] {session_name}")
    print(f"[TASK] trials={cfg.trials}, error_prob={cfg.error_prob:.2f}, break_every={cfg.break_every}")
    print("[TRIGGERS]")
    for key, value in asdict(TriggerCodes()).items():
        print(f"  {key}: {value}")

    game = RacingErrPGame(cfg=cfg, serial_cfg=serial_cfg, session_name=session_name)
    game.run()


if __name__ == "__main__":
    main()
