"""DSI headset recorder: background acquisition, event marks, EDF+ export.

Loads Wearable Sensing's dsi.py wrapper from a given folder, records all
channels while a task runs, and saves them as an EDF+ file with annotations.

Requires: numpy, pyedflib, and dsi.py + the DSI API library in `dsi_dir`.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import importlib
import os
import sys
import threading

import numpy as np
import pyedflib


class DSIRecorder:
    """Record all DSI channels in the background and save them as EDF+."""

    def __init__(self, com_port: str, dsi_dir: str):
        self.com_port = com_port
        self.dsi_dir = Path(dsi_dir)
        sys.path.insert(0, str(self.dsi_dir))  # so dsi.py can import the DSI API
        self.dsi = None  # dsi module, loaded in connect()
        self.headset = None
        self.channels: list = []
        self.channel_names: list[str] = []
        self.sample_rate: int = 0
        self.start_time: datetime | None = None

        self._rows: list[list[float]] = []
        self._events: list[tuple[int, str]] = []  # (sample_index, description)
        self._lock = threading.Lock()

    # Called by the DSI API on every incoming data packet (background thread).
    def _on_sample(self, headset_ptr, packet_time, user_data) -> None:
        try:
            row = [ch.ReadBuffered() for ch in self.channels]
        except Exception:
            return
        with self._lock:
            self._rows.append(row)

    def _load_dsi(self) -> None:
        """Make the DSI API folder importable, then import the dsi wrapper."""
        if not self.dsi_dir.is_dir():
            raise RuntimeError(f"DSI API folder not found: {self.dsi_dir} (use --dsi-dir)")
        sys.path.insert(0, str(self.dsi_dir))
        if hasattr(os, "add_dll_directory"):  # Windows, Python 3.8+
            os.add_dll_directory(str(self.dsi_dir))
        self.dsi = importlib.import_module("dsi")

    def connect(self) -> None:
        self._load_dsi()
        self.headset = self.dsi.Headset()
        self.headset.Connect(self.com_port)
        self.sample_rate = int(self.headset.GetSamplingRate())
        self.channels = list(self.headset.Channels())
        self.channel_names = [ch.GetName() for ch in self.channels]
        print(f"[HEADSET] Connected to {self.headset.GetInfoString()} at {self.sample_rate} Hz on {self.com_port}")
        print(f"[HEADSET] Channels ({len(self.channel_names)}): {self.channel_names}")

    def start(self) -> None:
        if self.headset is None:
            raise RuntimeError("Headset not connected")
        # dsi.py needs a ctypes-wrapped callback. Keep a reference on self,
        # otherwise it gets garbage-collected and the program crashes.
        self._callback = self.dsi.SampleCallback(self._on_sample)
        self.headset.SetSampleCallback(self._callback, 0)
        # Background acquisition keeps samples flowing while PsychoPy runs the task.
        self.headset.StartBackgroundAcquisition()
        self.start_time = datetime.now()
        print("[HEADSET] Recording started")

    def mark(self, description: str) -> None:
        """Mark an event at the current sample position (saved as an EDF+ annotation)."""
        with self._lock:
            self._events.append((len(self._rows), description))

    def stop(self) -> None:
        if self.headset is None:
            return
        try:
            self.headset.StopBackgroundAcquisition()
        except Exception as exc:
            print(f"[HEADSET] Warning while stopping acquisition: {exc}")
        try:
            self.headset.Disconnect()
        except Exception as exc:
            print(f"[HEADSET] Warning while disconnecting: {exc}")
        self.headset = None
        print("[HEADSET] Disconnected")

    def save_edf(self, path: Path, patient_code: str = "") -> None:
        with self._lock:
            rows = list(self._rows)
            events = list(self._events)

        if not rows:
            print("[HEADSET] No samples recorded; nothing to save")
            return

        fs = self.sample_rate
        data = np.array(rows, dtype=np.float64).T  # (n_channels, n_samples)

        # EDF stores whole 1-second records, so drop any partial final second.
        n_keep = (data.shape[1] // fs) * fs
        if n_keep == 0:
            print("[HEADSET] Recording shorter than 1 second; nothing to save")
            return
        data = data[:, :n_keep]

        n_channels = data.shape[0]
        path.parent.mkdir(parents=True, exist_ok=True)

        signal_headers = []
        for name, sig in zip(self.channel_names, data):
            lo, hi = float(np.min(sig)), float(np.max(sig))
            pad = max(1.0, 0.05 * (hi - lo))  # headroom so nothing clips
            signal_headers.append({
                "label": name[:16],
                "dimension": "uV",
                "sample_frequency": fs,
                "physical_min": lo - pad,
                "physical_max": hi + pad,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "Dry EEG",
                "prefilter": "",
            })

        with pyedflib.EdfWriter(str(path), n_channels, file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
            if self.start_time is not None:
                writer.setStartdatetime(self.start_time)
            if patient_code:
                writer.setPatientCode(patient_code)
            writer.setSignalHeaders(signal_headers)
            writer.writeSamples(data)
            for sample_idx, desc in events:
                if sample_idx < n_keep:
                    writer.writeAnnotation(sample_idx / fs, -1, desc)

        print(f"[HEADSET] Saved {data.shape[0]} channels x {data.shape[1]} samples "
              f"({data.shape[1] / fs:.1f} s) and {len(events)} events to {path}")