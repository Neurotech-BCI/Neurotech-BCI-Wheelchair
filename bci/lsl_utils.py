from __future__ import annotations

import logging
import time
from typing import Any

from mne_lsl.lsl import resolve_streams
from mne_lsl.stream import StreamLSL

_log = logging.getLogger("bci.lsl")


def _describe(info: Any) -> str:
    return (
        f"name={info.name!r}, stype={info.stype!r}, source_id={info.source_id!r}, "
        f"n_channels={info.n_channels}, sfreq={info.sfreq}"
    )


def _matches(info: Any, name: str | None, stype: str | None, source_id: str | None) -> bool:
    if name and info.name != name:
        return False
    if stype and info.stype != stype:
        return False
    if source_id and info.source_id != source_id:
        return False
    return True


def _resolve_all(timeout_s: float, attempts: int) -> list[Any]:
    streams: list[Any] = []
    for attempt in range(attempts):
        streams = list(resolve_streams(timeout=timeout_s))
        if streams:
            return streams
        if attempt < attempts - 1:
            _log.warning("No LSL streams visible yet (attempt %d/%d), retrying...", attempt + 1, attempts)
            time.sleep(1.0)
    return streams


def connect_eeg_stream(
    lsl_cfg: Any,
    bufsize: float = 60.0,
    logger: logging.Logger | None = None,
    resolve_timeout_s: float = 3.0,
    attempts: int = 3,
) -> StreamLSL:
    """Find the EEG stream on the network and connect to it.

    Matching order:
      1. Exact match on the configured name, stype and source_id.
      2. If nothing matches the configured name, fall back to the only stream
         of the configured type (e.g. the only "EEG" stream visible).

    Raises a readable error listing every visible stream when the choice is
    ambiguous or nothing usable is found.
    """
    log = logger or _log
    name = getattr(lsl_cfg, "name", None)
    stype = getattr(lsl_cfg, "stype", None)
    source_id = getattr(lsl_cfg, "source_id", None)

    use_tcp = getattr(lsl_cfg, "dsi_tcp_fallback", True)
    # With the DSI-Streamer fallback on, look once before trying it, so a task
    # started with only DSI-Streamer open does not wait through every retry.
    streams = _resolve_all(resolve_timeout_s, 1 if use_tcp else attempts)
    visible = "\n".join(f"  - {_describe(s)}" for s in streams) or "  (none)"

    if not streams and use_tcp:
        # No LSL stream: read DSI-Streamer's TCP/IP output directly instead
        # (DSI-Streamer stays open and keeps recording). See dsi_tcp_bridge.py.
        try:
            from dsi_tcp_bridge import start_background_bridge
        except ImportError:
            from bci.dsi_tcp_bridge import start_background_bridge
        host = getattr(lsl_cfg, "dsi_tcp_host", "127.0.0.1")
        port = getattr(lsl_cfg, "dsi_tcp_port", 8844)
        try:
            start_background_bridge(lsl_cfg, logger=log)
            streams = _resolve_all(resolve_timeout_s, attempts)
            visible = "\n".join(f"  - {_describe(s)}" for s in streams) or "  (none)"
        except (ConnectionError, OSError) as exc:
            log.info("No DSI-Streamer TCP/IP output on %s:%s (%s); looking for LSL again.", host, port, exc)
            if attempts > 1:
                streams = _resolve_all(resolve_timeout_s, attempts - 1)
                visible = "\n".join(f"  - {_describe(s)}" for s in streams) or "  (none)"
        except TimeoutError as exc:
            raise RuntimeError(str(exc)) from exc

    if not streams:
        raise RuntimeError(
            "No EEG stream found.\n"
            "Either: in DSI-Streamer, open the TCP IP tab, check 'Activate TCP/IP Socket' and press Start "
            "(DSI-Streamer can keep recording), or run an LSL streamer such as dsi2lslGUI.\n"
            "Also check that Python is allowed through the firewall."
        )

    candidates = [s for s in streams if _matches(s, name, stype, source_id)]
    if not candidates:
        candidates = [s for s in streams if _matches(s, None, stype, source_id)]
        if len(candidates) == 1:
            log.warning(
                "No LSL stream named %r was found. Using the only %r stream instead: %s. "
                "Set LSLConfig.name in bci/config.py to %r to silence this warning.",
                name,
                stype,
                _describe(candidates[0]),
                candidates[0].name,
            )

    if not candidates:
        raise RuntimeError(
            f"No LSL stream matches name={name!r}, stype={stype!r}, source_id={source_id!r}.\n"
            f"Visible streams:\n{visible}\n"
            "Set LSLConfig.name / stype in bci/config.py to match your EEG stream."
        )

    if len(candidates) > 1:
        ids = [c.source_id for c in candidates]
        if len(set(ids)) < len(ids):
            hint = "Two of these streams are identical, so close the duplicate streamer window and try again."
        else:
            hint = "Set LSLConfig.source_id in bci/config.py to the source_id of the stream you want."
        raise RuntimeError(
            f"More than one LSL stream matches name={name!r}, stype={stype!r}, source_id={source_id!r}:\n"
            + "\n".join(f"  - {_describe(c)}" for c in candidates)
            + f"\n{hint}"
        )

    chosen = candidates[0]
    stream = StreamLSL(
        bufsize=bufsize,
        name=chosen.name,
        stype=chosen.stype,
        source_id=chosen.source_id or None,
    )
    stream.connect(acquisition_delay=0.001, processing_flags="all")
    log.info("Connected to LSL stream: %s", _describe(chosen))
    return stream
