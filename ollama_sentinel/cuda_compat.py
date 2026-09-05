"""CUDA / display-driver compatibility for local Ollama (read-only).

Ollama bundles its own CUDA runtime (``libdirs=ollama,cuda_v13``). The NVIDIA
display driver only has to be new enough to expose that user-mode CUDA. When it
is not, inference silently lands on CPU — the failure mode people call a
\"CUDA mismatch\".

We read two facts:

* ``nvidia-smi`` banner / driver query → what CUDA the *driver* advertises
* latest ``inference compute`` line in ``server.log`` → which ``cuda_vN`` Ollama
  actually selected, and the ``driver=`` UMD it saw

Warn only when the driver major is behind Ollama's bundled major, or when the
log shows NVIDIA present but no CUDA library was chosen.
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# msg="inference compute" id=0 library=CUDA ... libdirs=ollama,cuda_v13 driver=13.4
_INFERENCE_COMPUTE = re.compile(
    r'msg="inference compute".*?library=(?P<library>\S+)'
    r".*?libdirs=(?P<libdirs>\S+)"
    r".*?driver=(?P<driver>\S+)",
    re.I,
)
_CUDA_V = re.compile(r"cuda_v(\d+)", re.I)
_SMI_CUDA_UMD = re.compile(r"CUDA\s+UMD\s+Version:\s*([0-9]+(?:\.[0-9]+)*)", re.I)
_SMI_CUDA_LEGACY = re.compile(r"CUDA\s+Version:\s*([0-9]+(?:\.[0-9]+)*)", re.I)
_VERSION_TUPLE = re.compile(r"(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class CudaProbe:
    """One reading of driver CUDA vs what Ollama bound."""

    driver_version: str | None = None
    driver_cuda: str | None = None  # from nvidia-smi (e.g. "13.4")
    ollama_library: str | None = None  # CUDA / Vulkan / CPU
    ollama_cuda_major: int | None = None  # from cuda_v13
    ollama_driver_cuda: str | None = None  # driver= on inference line
    ok: bool = True
    reason: str | None = None


def _version_major(raw: str | None) -> int | None:
    if not raw:
        return None
    m = _VERSION_TUPLE.search(raw)
    return int(m.group(1)) if m else None


def parse_smi_cuda_version(text: str) -> str | None:
    """Extract CUDA UMD / CUDA Version from ``nvidia-smi`` table header."""
    m = _SMI_CUDA_UMD.search(text) or _SMI_CUDA_LEGACY.search(text)
    return m.group(1) if m else None


def parse_inference_compute(text: str) -> dict[str, Any] | None:
    """Return the *last* NVIDIA-relevant inference compute record in a log tail."""
    last: dict[str, Any] | None = None
    for m in _INFERENCE_COMPUTE.finditer(text):
        libdirs = m.group("libdirs")
        library = m.group("library")
        driver = m.group("driver")
        cuda_maj = None
        cv = _CUDA_V.search(libdirs)
        if cv:
            cuda_maj = int(cv.group(1))
        # Prefer a CUDA row; keep last match overall as fallback.
        rec = {
            "library": library,
            "libdirs": libdirs,
            "driver": driver.rstrip(","),
            "cuda_major": cuda_maj,
        }
        if library.upper() == "CUDA" or cuda_maj is not None:
            last = rec
        elif last is None:
            last = rec
    return last


def evaluate_cuda_probe(
    *,
    driver_cuda: str | None,
    inference: dict[str, Any] | None,
    driver_version: str | None = None,
) -> CudaProbe:
    """Decide whether driver CUDA can host Ollama's bundled runtime."""
    if inference is None:
        return CudaProbe(
            driver_version=driver_version,
            driver_cuda=driver_cuda,
            ok=True,
            reason="no inference-compute line yet — start Ollama once to measure",
        )

    library = str(inference.get("library") or "")
    ollama_driver = inference.get("driver")
    cuda_major = inference.get("cuda_major")
    if isinstance(cuda_major, str) and cuda_major.isdigit():
        cuda_major = int(cuda_major)

    probe = CudaProbe(
        driver_version=driver_version,
        driver_cuda=driver_cuda,
        ollama_library=library,
        ollama_cuda_major=cuda_major if isinstance(cuda_major, int) else None,
        ollama_driver_cuda=str(ollama_driver) if ollama_driver else None,
        ok=True,
    )

    # Ollama saw an NVIDIA path but did not bind CUDA — classic mismatch/fallback.
    if cuda_major is not None and library.upper() != "CUDA":
        return CudaProbe(
            driver_version=probe.driver_version,
            driver_cuda=probe.driver_cuda,
            ollama_library=probe.ollama_library,
            ollama_cuda_major=probe.ollama_cuda_major,
            ollama_driver_cuda=probe.ollama_driver_cuda,
            ok=False,
            reason=(
                f"Ollama bundled cuda_v{cuda_major} but selected library={library} "
                "instead of CUDA — update the NVIDIA display driver and restart Ollama"
            ),
        )

    if cuda_major is None:
        return probe

    # Compare smi-advertised CUDA major to bundled cuda_vN.
    smi_major = _version_major(driver_cuda)
    if smi_major is not None and smi_major < cuda_major:
        return CudaProbe(
            driver_version=probe.driver_version,
            driver_cuda=probe.driver_cuda,
            ollama_library=probe.ollama_library,
            ollama_cuda_major=probe.ollama_cuda_major,
            ollama_driver_cuda=probe.ollama_driver_cuda,
            ok=False,
            reason=(
                f"Driver advertises CUDA {driver_cuda} but Ollama needs cuda_v{cuda_major} "
                f"(display driver {driver_version or '?'}) — update the NVIDIA driver"
            ),
        )

    # Ollama's own driver= field behind its cuda_vN (stale process after a bad upgrade).
    ollama_major = _version_major(str(ollama_driver) if ollama_driver else None)
    if ollama_major is not None and ollama_major < cuda_major:
        return CudaProbe(
            driver_version=probe.driver_version,
            driver_cuda=probe.driver_cuda,
            ollama_library=probe.ollama_library,
            ollama_cuda_major=probe.ollama_cuda_major,
            ollama_driver_cuda=probe.ollama_driver_cuda,
            ok=False,
            reason=(
                f"Ollama reports driver CUDA {ollama_driver} with cuda_v{cuda_major} — "
                "restart Ollama after updating the NVIDIA display driver"
            ),
        )

    return probe


def _no_window() -> dict:
    if sys.platform != "win32":
        return {}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": flags, "startupinfo": si}


def query_driver_cuda() -> tuple[str | None, str | None]:
    """Return ``(driver_version, cuda_umd_version)`` from nvidia-smi, or nulls."""
    driver_version = None
    try:
        ver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            **_no_window(),
        )
        if ver.returncode == 0 and ver.stdout.strip():
            driver_version = ver.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass

    cuda = None
    try:
        banner = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
            **_no_window(),
        )
        if banner.returncode == 0:
            cuda = parse_smi_cuda_version(banner.stdout or "")
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        pass
    return driver_version, cuda


def probe_cuda_compat(
    *,
    log_path: Path | None = None,
    log_text: str | None = None,
    driver_version: str | None = None,
    driver_cuda: str | None = None,
) -> CudaProbe:
    """Build a probe from log text and optional pre-queried smi fields."""
    if driver_version is None and driver_cuda is None:
        driver_version, driver_cuda = query_driver_cuda()

    text = log_text
    if text is None and log_path is not None and log_path.is_file():
        try:
            # Tail only — same idea as activity/ctx_pressure.
            size = log_path.stat().st_size
            with log_path.open("rb") as fh:
                if size > 512_000:
                    fh.seek(size - 512_000)
                    fh.readline()
                text = fh.read().decode("utf-8", errors="replace")
        except OSError:
            text = None

    inference = parse_inference_compute(text or "")
    return evaluate_cuda_probe(
        driver_cuda=driver_cuda,
        inference=inference,
        driver_version=driver_version,
    )
