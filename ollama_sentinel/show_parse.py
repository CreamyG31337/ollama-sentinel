"""Pure parsers for Ollama GET /api/show payloads."""

from __future__ import annotations

import re
from typing import Any

# ggml file_type -> label (subset; unknown falls back to details / tensor types)
FILE_TYPE_LABELS: dict[int, str] = {
    0: "F32",
    1: "F16",
    7: "Q8_0",
    8: "Q8_0",
    15: "Q4_K_M",
}

# Approximate bits per weight for tensor footprint sums (good enough for fit advisories)
TENSOR_BITS: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.0,
    "Q8_1": 8.5,
    "Q6_K": 6.5625,
    "Q5_K": 5.5,
    "Q5_0": 5.5,
    "Q5_1": 5.5,
    "Q4_K": 4.5,
    "Q4_0": 4.5,
    "Q4_1": 4.5,
    "Q4_K_M": 4.5,
    "Q4_K_S": 4.5,
    "Q3_K": 3.5,
    "Q3_K_M": 3.5,
    "Q3_K_S": 3.5,
    "Q2_K": 2.5,
    "IQ3_XXS": 3.06,
    "IQ3_S": 3.44,
    "IQ4_XS": 4.25,
    "MXFP4": 4.5,
}

_PARAM_INT = re.compile(r"^(\w+)\s+(-?\d+)\s*$", re.MULTILINE)
_NEXTN_TENSOR = re.compile(r"blk\.\d+\.nextn\.", re.IGNORECASE)


def parse_parameters_block(parameters: str | None) -> dict[str, str]:
    if not parameters:
        return {}
    out: dict[str, str] = {}
    for line in parameters.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _PARAM_INT.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_draft_num_predict(parameters: str | None) -> int | None:
    params = parse_parameters_block(parameters)
    raw = params.get("draft_num_predict")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def modelfile_has_draft(modelfile: str | None) -> bool:
    if not modelfile:
        return False
    for line in modelfile.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("DRAFT "):
            return True
    return False


def tensor_weight_bytes(tensors: list[dict[str, Any]] | None) -> int | None:
    if not tensors:
        return None
    total_bits = 0.0
    counted = 0
    for t in tensors:
        shape = t.get("shape")
        typ = (t.get("type") or "").upper()
        if not shape or not typ:
            continue
        elems = 1
        for dim in shape:
            elems *= int(dim)
        bits = TENSOR_BITS.get(typ)
        if bits is None:
            continue
        total_bits += elems * bits
        counted += 1
    if counted == 0:
        return None
    return int(total_bits / 8)


def quant_from_show(show: dict[str, Any]) -> str | None:
    details = show.get("details") or {}
    q = details.get("quantization_level")
    if q and str(q).lower() not in ("unknown", ""):
        return str(q)
    info = show.get("model_info") or {}
    ft = info.get("general.file_type")
    if isinstance(ft, int) and ft in FILE_TYPE_LABELS:
        return FILE_TYPE_LABELS[ft]
    tensors = show.get("tensors") or []
    if tensors:
        return str(tensors[0].get("type") or "") or None
    return None


def mtp_layers_from_show(show: dict[str, Any]) -> int | None:
    info = show.get("model_info") or {}
    for key, val in info.items():
        if key.endswith("nextn_predict_layers"):
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    tensors = show.get("tensors") or []
    if any(_NEXTN_TENSOR.search(t.get("name") or "") for t in tensors):
        return 1
    return None


def requires_ollama_version(show: dict[str, Any]) -> str | None:
    req = show.get("requires")
    if req is None:
        return None
    s = str(req).strip()
    return s or None


def parse_show_bundle(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a successful /api/show body."""
    if raw.get("error"):
        return {"error": str(raw["error"])}
    params = parse_parameters_block(raw.get("parameters"))
    draft = parse_draft_num_predict(raw.get("parameters"))
    return {
        "error": None,
        "quantization": quant_from_show(raw),
        "weight_bytes": tensor_weight_bytes(raw.get("tensors")),
        "mtp_layers": mtp_layers_from_show(raw),
        "draft_num_predict": draft,
        "has_draft_modelfile": modelfile_has_draft(raw.get("modelfile")),
        "requires": requires_ollama_version(raw),
        "capabilities": list(raw.get("capabilities") or []),
        "num_ctx": int(params["num_ctx"]) if params.get("num_ctx", "").lstrip("-").isdigit() else None,
        "family": (raw.get("details") or {}).get("family"),
        "parameter_size": (raw.get("details") or {}).get("parameter_size"),
        "has_ssm": any(str(k).endswith("ssm.conv_kernel") for k in (raw.get("model_info") or {})),
        "has_vision": "vision" in (raw.get("capabilities") or [])
        or bool(raw.get("projector_info")),
        "expert_count": _model_info_int(raw, "expert_count"),
        "expert_used_count": _model_info_int(raw, "expert_used_count"),
    }


def _model_info_int(raw: dict[str, Any], suffix: str) -> int | None:
    info = raw.get("model_info") or {}
    for key, val in info.items():
        if key.endswith(suffix):
            try:
                return int(val)
            except (TypeError, ValueError):
                return None
    return None
