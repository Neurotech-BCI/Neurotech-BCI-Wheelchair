'''
    DSI-Streamer TCP/IP -> LSL bridge.

    Lets the live tasks run while DSI-Streamer stays connected to the headset
    and keeps recording as usual. DSI-Streamer streams the live data over its
    TCP/IP socket; this bridge reads it and republishes it as the LSL stream
    the tasks expect (LSLConfig.name, "WS-default" by default).

    In DSI-Streamer: open the TCP IP tab, check "Activate TCP/IP Socket"
    (default port 8844), and start TCP/IP streaming. Then:

      - Run any task as usual. If no LSL stream is found, connect_eeg_stream()
        in lsl_utils.py starts this bridge automatically inside the task.
      - Or run the bridge on its own (e.g. for LabRecorder or another program):
            python bci/dsi_tcp_bridge.py

    Packet format (DSI-Streamer User Manual, TCP/IP and Packet Structure
    sections): big-endian; 12-byte header "@ABCD" + type (1 byte) + payload
    length (2 bytes) + packet number (4 bytes). Type 5 = event (code 9 sensor
    map, code 10 data rate), type 1 = one EEG sample: timestamp (float),
    counter (1 byte), ADC status (6 bytes), then one float per channel in uV,
    with the trigger as the last channel.

    Re-referencing: offline EDFs are re-referenced to the LE channel before
    training (mental_command_worker.standardize_offline_raw). If the TCP data
    contains an LE channel, the bridge does the same, so live data matches the
    data the models were trained on. Turn it off with LSLConfig.dsi_tcp_reref_le
    or --no-reref.
'''
from __future__ import annotations

import logging
import socket
import struct
import threading
import time

import numpy as np
from mne_lsl.lsl import StreamInfo, StreamOutlet, local_clock

_log = logging.getLogger("bci.dsi_tcp")

HEADER_START = b"@ABCD"
TYPE_EEG = 1
TYPE_EVENT = 5
EVENT_GREETING, EVENT_START, EVENT_STOP, EVENT_SENSOR_MAP, EVENT_DATA_RATE = 1, 2, 3, 9, 10
TRIGGER_NAMES = {"TRG", "TRIGGER"}


def _canon(name: str) -> str:
    return name.strip().upper().replace("EEG ", "").replace("-PZ", "")


def parse_data_rate(message: str, default: float = 300.0) -> float:
    """The data-rate event carries the mains frequency and the sampling rate."""
    numbers = []
    for token in message.replace(";", ",").replace(" ", ",").split(","):
        try:
            numbers.append(float(token))
        except ValueError:
            pass
    for value in numbers:
        if value in (300.0, 600.0, 900.0):
            return value
    rates = [v for v in numbers if v not in (50.0, 60.0)]
    return max(rates) if rates else default


class DSITcpBridge:
    """Reads DSI-Streamer's TCP/IP output in a background thread and pushes
    every sample to an LSL outlet."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8844, stream_name: str = "WS-default",
                 stream_type: str = "EEG", reref_to_le: bool = True, logger: logging.Logger | None = None):
        self.host, self.port = host, int(port)
        self.stream_name, self.stream_type = stream_name, stream_type
        self.reref_to_le = bool(reref_to_le)
        self.log = logger or _log
        self.sock: socket.socket | None = None
        self.outlet: StreamOutlet | None = None
        self.sensor_names: list[str] = []
        self.sfreq: float | None = None
        self.out_names: list[str] = []
        self.n_samples = 0
        self.last_trigger = 0.0
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._clock_offset: float | None = None
        self._last_ts: float | None = None

    # ---- connection -----------------------------------------------------
    def start(self, connect_timeout_s: float = 2.0, ready_timeout_s: float = 10.0) -> "DSITcpBridge":
        """Connect and wait until the first samples are flowing. Raises
        ConnectionError if DSI-Streamer is not listening, TimeoutError if it
        is but sends no EEG (TCP/IP streaming not started)."""
        self.sock = socket.create_connection((self.host, self.port), timeout=connect_timeout_s)
        self.sock.settimeout(1.0)
        self._thread = threading.Thread(target=self._run, name="dsi-tcp-bridge", daemon=True)
        self._thread.start()
        if not self._ready.wait(ready_timeout_s):
            self.stop()
            if self._error:
                raise ConnectionError(f"DSI-Streamer TCP bridge failed: {self._error}")
            raise TimeoutError(
                f"Connected to DSI-Streamer on {self.host}:{self.port} but no EEG arrived within "
                f"{ready_timeout_s:.0f} s. Press Start in DSI-Streamer's TCP IP tab and make sure the headset is streaming.")
        self.log.info("DSI-Streamer TCP bridge running: %s:%d -> LSL %r, %d channels %s at %.0f Hz",
                      self.host, self.port, self.stream_name, len(self.out_names), self.out_names, self.sfreq)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---- reading -------------------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                raise ConnectionAbortedError("bridge stopped")
            try:
                chunk = self.sock.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                raise ConnectionResetError("DSI-Streamer closed the TCP connection")
            buf.extend(chunk)
        return bytes(buf)

    def _read_packet(self) -> tuple[int, bytes]:
        # Find the start marker (resynchronizes if a byte is ever dropped).
        window = self._read_exact(len(HEADER_START))
        while window != HEADER_START:
            window = window[1:] + self._read_exact(1)
        ptype, length, _number = struct.unpack(">BHI", self._read_exact(7))
        return ptype, self._read_exact(length)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                ptype, payload = self._read_packet()
                if ptype == TYPE_EVENT:
                    self._on_event(payload)
                elif ptype == TYPE_EEG:
                    self._on_sample(payload)
        except Exception as exc:  # noqa: BLE001 - reported to the caller / log
            if not self._stop.is_set():
                self._error = exc
                self.log.error("DSI-Streamer TCP bridge stopped: %s", exc)

    def _on_event(self, payload: bytes) -> None:
        code, _node = struct.unpack(">II", payload[:8])
        message = ""
        if len(payload) >= 12:
            (msg_len,) = struct.unpack(">I", payload[8:12])
            message = payload[12:12 + msg_len].decode("ascii", errors="replace").strip("\x00 ")
        if code == EVENT_SENSOR_MAP:
            self.sensor_names = [n.strip() for n in message.split(",") if n.strip()]
            self.log.info("DSI-Streamer sensor map: %s", self.sensor_names)
        elif code == EVENT_DATA_RATE:
            self.sfreq = parse_data_rate(message)
            self.log.info("DSI-Streamer data rate: %r -> %.0f Hz", message, self.sfreq)
        elif code == EVENT_GREETING:
            self.log.info("DSI-Streamer: %s", message)
        elif code == EVENT_STOP:
            self.log.warning("DSI-Streamer stopped acquisition")

    def _setup_outlet(self, n_values: int) -> None:
        names = list(self.sensor_names)
        if len(names) == n_values - 1:
            names.append("TRG")  # the trigger comes after the last sensor
        if len(names) != n_values:
            self.log.warning("Sensor map has %d names but packets have %d values; using generic names",
                             len(names), n_values)
            names = [f"Ch{i + 1}" for i in range(n_values - 1)] + ["TRG"]
        names = [n if n != "-" else f"unused{i + 1}" for i, n in enumerate(names)]  # disconnected sensors
        canon = [_canon(n) for n in names]
        self._trg_idx = next((i for i, c in enumerate(canon) if c in TRIGGER_NAMES), n_values - 1)
        self._le_idx = canon.index("LE") if (self.reref_to_le and "LE" in canon) else None
        self._add_pz = self._le_idx is not None and "PZ" not in canon
        out = list(names)
        if self._add_pz:
            out.insert(self._trg_idx, "Pz")  # reference electrode, recovered by the re-reference
        self.out_names = out
        if self.sfreq is None:
            self.sfreq = 300.0
            self.log.warning("No data-rate event received; assuming 300 Hz")
        info = StreamInfo(self.stream_name, self.stream_type, len(out), self.sfreq, "float32",
                          f"dsi-streamer-tcp-{self.host}-{self.port}")
        info.set_channel_names(out)
        self.outlet = StreamOutlet(info)
        if self._le_idx is not None:
            self.log.info("Re-referencing to LE to match the offline EDF pipeline%s",
                          " (adding Pz = -LE)" if self._add_pz else "")

    def _on_sample(self, payload: bytes) -> None:
        (ts,) = struct.unpack(">f", payload[:4])
        values = np.frombuffer(payload[11:], dtype=">f4").astype(np.float64)
        if self.outlet is None:
            self._setup_outlet(values.size)
        if self._le_idx is not None:
            le = values[self._le_idx]
            eeg = np.ones(values.size, dtype=bool)
            eeg[self._trg_idx] = False
            values = np.where(eeg, values - le, values)
            if self._add_pz:
                values = np.insert(values, self._trg_idx, -le)
        # Map DSI-Streamer's timestamps onto the LSL clock, keeping its sample spacing.
        if self._clock_offset is None or (self._last_ts is not None and ts < self._last_ts - 1.0):
            self._clock_offset = local_clock() - ts
        self._last_ts = ts
        self.outlet.push_sample(values.astype(np.float32), timestamp=ts + self._clock_offset)
        self.last_trigger = float(values[self._trg_idx + (1 if self._add_pz else 0)])
        self.n_samples += 1
        if self.n_samples == 1:
            self._ready.set()


_ACTIVE: DSITcpBridge | None = None


def start_background_bridge(lsl_cfg=None, logger: logging.Logger | None = None) -> DSITcpBridge:
    """Start one bridge per process (reused if already running), using the
    TCP settings in LSLConfig when given."""
    global _ACTIVE
    if _ACTIVE is not None and _ACTIVE.alive:
        return _ACTIVE
    _ACTIVE = DSITcpBridge(
        host=getattr(lsl_cfg, "dsi_tcp_host", "127.0.0.1"),
        port=getattr(lsl_cfg, "dsi_tcp_port", 8844),
        stream_name=getattr(lsl_cfg, "name", None) or "WS-default",
        stream_type=getattr(lsl_cfg, "stype", None) or "EEG",
        reref_to_le=getattr(lsl_cfg, "dsi_tcp_reref_le", True),
        logger=logger,
    ).start()
    return _ACTIVE


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Republish DSI-Streamer's TCP/IP output as an LSL stream.")
    ap.add_argument("--host", default="127.0.0.1", help="Computer running DSI-Streamer (default: this one)")
    ap.add_argument("--port", type=int, default=8844, help="DSI-Streamer TCP/IP port (default: 8844)")
    ap.add_argument("--name", default="WS-default", help="LSL stream name (default: WS-default)")
    ap.add_argument("--no-reref", action="store_true", help="Do not re-reference to LE")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    bridge = DSITcpBridge(args.host, args.port, args.name, reref_to_le=not args.no_reref).start()
    print("Bridge running. Leave this window open; press Ctrl+C to stop.")
    try:
        while bridge.alive:
            time.sleep(5.0)
            print(f"  {bridge.n_samples} samples sent, last trigger value {bridge.last_trigger:.0f}")
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    main()
