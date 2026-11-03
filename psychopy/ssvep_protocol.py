"""
ssvep_protocol.py - SSVEP calibration + idle blocks for the DSI-7, with automatic markers.

What it runs (defaults, about 10 minutes total):
  1. Idle baseline: 120 s, static gray square, no flicker
  2. For each frequency (12, 15, 20 Hz): 3 x (10 s flicker ON, 10 s OFF), then a 60 s idle block
  3. "Ignore" block: 60 s of 12 Hz flicker while the participant looks at a cross off to the side
     (tests whether a light that's on but NOT attended triggers the detector)
  4. Idle end: 120 s

Markers are sent three ways (use whichever you have):
  - Hardware trigger hub (Arduino, same one used in psychopy/mi_task.py). Codes land in the
    "Trigger" column of the DSI-Streamer recording, sample-locked to the EEG. This is the main method.
  - A CSV log next to this script (always written), as a backup and for checking.
  - An LSL marker stream, if pylsl is installed (for LabRecorder setups).

Trigger codes (start, end):
  12 Hz trial 11/12   15 Hz trial 21/22   20 Hz trial 31/32
  idle block  91/92   ignore block 81/82  session 100/101

Run:
  python ssvep_protocol.py --participant omer --fullscreen
  python ssvep_protocol.py --test-trigger          # pulses 1..5 so you can confirm DSI-Streamer sees them
Press ESC at any time to quit (an end marker is still sent).
"""
import argparse
import csv
import math
import random
from datetime import datetime
from pathlib import Path

from psychopy import core, event, visual

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

try:
    from pylsl import StreamInfo, StreamOutlet
except ImportError:
    StreamOutlet = None

CODES = {12: (11, 12), 15: (21, 22), 20: (31, 32)}
IDLE = (91, 92)
IGNORE = (81, 82)
SESSION = (100, 101)


def find_trigger_port(vid=0x2341, pid=0x8037):
    """Same Arduino trigger hub lookup as bci/config.py."""
    if serial is None:
        return None
    for p in serial.tools.list_ports.comports():
        if p.vid == vid and p.pid == pid:
            return p.device
    return None


class Markers:
    """Sends a code to every available marker channel and logs it."""

    def __init__(self, port, log_path):
        self.ser = None
        if port and serial is not None:
            self.ser = serial.Serial(port, 115200, timeout=0)
            print(f"Trigger hub: {port}")
        else:
            print("WARNING: no trigger hub found. Markers go to the CSV log (and LSL if available) only.")
        self.outlet = None
        if StreamOutlet is not None:
            info = StreamInfo("SSVEPMarkers", "Markers", 1, 0, "int32", "ssvep_protocol")
            self.outlet = StreamOutlet(info)
        self.clock = core.Clock()
        self.log_f = open(log_path, "w", newline="")
        self.log = csv.writer(self.log_f)
        self.log.writerow(["wall_time", "psychopy_time_s", "code", "label"])

    def send_raw(self, code):
        """Write one byte to the hub (0 resets the line). Called on a screen flip."""
        if self.ser is not None:
            self.ser.write(bytes([code]))

    def mark(self, win, code, label):
        """Schedule the code for the next flip, log it, and reset to 0 on the following flip."""
        win.callOnFlip(self.send_raw, code)
        win.callOnFlip(self._record, code, label)

    def _record(self, code, label):
        if self.outlet is not None:
            self.outlet.push_sample([code])
        self.log.writerow([datetime.now().isoformat(timespec="milliseconds"),
                           f"{self.clock.getTime():.4f}", code, label])
        self.log_f.flush()

    def reset_next_flip(self, win):
        win.callOnFlip(self.send_raw, 0)

    def close(self):
        if self.ser is not None:
            self.ser.write(bytes([0]))
            self.ser.close()
        self.log_f.close()


def check_quit(win, markers):
    if "escape" in event.getKeys(["escape"]):
        markers.mark(win, SESSION[1], "session_end_ESC")
        win.flip()
        markers.close()
        win.close()
        core.quit()


def show_text(win, text, seconds=None, wait_key=False):
    msg = visual.TextStim(win, text=text, height=0.04, color=[0.8] * 3, wrapWidth=1.4)
    msg.draw()
    win.flip()
    if wait_key:
        event.waitKeys(keyList=["space"])
    elif seconds:
        core.wait(seconds)


def run_block(win, markers, refresh, seconds, codes, label, freq=None, fix_pos=(0, 0)):
    """One block. freq=None -> static gray square (idle/off); otherwise sampled-sinusoid flicker."""
    stim = visual.Rect(win, width=0.45, height=0.45, fillColor=[0, 0, 0], lineColor=None)
    fix = visual.TextStim(win, text="+", height=0.06, color=[1, 0, 0], pos=fix_pos)
    # Small photodiode patch (bottom-left) that mirrors the flicker, for timing checks if you have a sensor.
    diode = visual.Rect(win, width=0.05, height=0.05, pos=(-0.85, -0.45), fillColor=[-1] * 3, lineColor=None)
    n_frames = int(round(seconds * refresh))
    for i in range(n_frames):
        if freq is None:
            lum = 0.0
        else:
            # Sampled sinusoid: works for any frequency, including ones where refresh/f is odd (12 Hz at 60 Hz = 5 frames).
            lum = math.sin(2 * math.pi * freq * i / refresh)
        stim.fillColor = [lum] * 3
        diode.fillColor = [lum] * 3
        stim.draw()
        diode.draw()
        fix.draw()
        if i == 0 and codes:
            markers.mark(win, codes[0], f"{label}_start")
        elif i == 1 and codes:
            markers.reset_next_flip(win)
        win.flip()
        check_quit(win, markers)
    if codes:
        markers.mark(win, codes[1], f"{label}_end")
        win.flip()
        markers.reset_next_flip(win)
        win.flip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--participant", default="participant")
    ap.add_argument("--freqs", type=float, nargs="+", default=[12, 15, 20])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--on", type=float, default=10.0)
    ap.add_argument("--off", type=float, default=10.0)
    ap.add_argument("--idle-start", type=float, default=120.0)
    ap.add_argument("--idle-between", type=float, default=60.0)
    ap.add_argument("--idle-end", type=float, default=120.0)
    ap.add_argument("--ignore-block", type=float, default=60.0, help="seconds of unattended 12 Hz flicker (0 = skip)")
    ap.add_argument("--shuffle", action="store_true", help="randomize frequency order")
    ap.add_argument("--port", default=None, help="trigger hub port override, e.g. COM6")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--test-trigger", action="store_true")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(__file__).with_name(f"ssvep_markers_{args.participant}_{stamp}.csv")
    win = visual.Window(fullscr=args.fullscreen, size=(1280, 720), color=[-1, -1, -1],
                        units="height", waitBlanking=True)
    win.recordFrameIntervals = True
    markers = Markers(args.port or find_trigger_port(), log_path)

    if args.test_trigger:
        for code in (1, 2, 3, 4, 5):
            markers.mark(win, code, f"test_{code}")
            win.flip()
            markers.reset_next_flip(win)
            win.flip()
            core.wait(1.0)
        print("Sent test codes 1-5. Check the Trigger channel in DSI-Streamer.")
        markers.close()
        win.close()
        core.quit()

    measured = win.getActualFrameRate(nIdentical=60, nMaxFrames=300, nWarmUpFrames=60)
    refresh = round(measured) if measured else 60
    print(f"Measured refresh: {measured} Hz -> using {refresh} Hz")
    for f in args.freqs:
        frames = refresh / f
        if abs(frames - round(frames)) > 1e-6:
            print(f"WARNING: {f} Hz is not refresh/integer at {refresh} Hz ({frames:.2f} frames per cycle).")

    freqs = list(args.freqs)
    if args.shuffle:
        random.shuffle(freqs)

    show_text(win, "Sit comfortably, relax your jaw, and keep your eyes on the red cross.\n\n"
                   "When the square flickers, keep looking at the cross in its center.\n\n"
                   "Press SPACE to begin.", wait_key=True)
    markers.mark(win, SESSION[0], "session_start")
    win.flip()
    markers.reset_next_flip(win)

    if args.idle_start > 0:
        show_text(win, "Idle block: just relax and look at the cross. Blink normally.", 3)
        run_block(win, markers, refresh, args.idle_start, IDLE, "idle_start")

    for fi, f in enumerate(freqs):
        show_text(win, f"Next: {f:g} Hz. Look at the cross in the square.", 3)
        codes = CODES.get(int(f), (int(f), int(f) + 1))
        for rep in range(args.reps):
            run_block(win, markers, refresh, args.on, codes, f"{f:g}Hz_trial{rep + 1}", freq=f)
            run_block(win, markers, refresh, args.off, None, "off")
        if args.idle_between > 0 and fi < len(freqs) - 1:
            show_text(win, "Idle block: relax and look at the cross. Blink normally.", 3)
            run_block(win, markers, refresh, args.idle_between, IDLE, f"idle_after_{f:g}Hz")

    if args.ignore_block > 0:
        show_text(win, "Ignore block: look at the cross on the LEFT, NOT at the square.", 4)
        run_block(win, markers, refresh, args.ignore_block, IGNORE, "ignore_12Hz", freq=12, fix_pos=(-0.6, 0))

    if args.idle_end > 0:
        show_text(win, "Final idle block: relax and look at the cross.", 3)
        run_block(win, markers, refresh, args.idle_end, IDLE, "idle_end")

    markers.mark(win, SESSION[1], "session_end")
    show_text(win, "Done. Thank you!")
    dropped = win.nDroppedFrames
    print(f"Dropped frames: {dropped}. Marker log: {log_path}")
    if dropped > 10:
        print("WARNING: many dropped frames. Close other programs and rerun; flicker timing may be off.")
    core.wait(2)
    markers.close()
    win.close()
    core.quit()


if __name__ == "__main__":
    main()
