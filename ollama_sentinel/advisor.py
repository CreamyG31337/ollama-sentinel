"""Model optimization advisories — heuristics with confidence, not guarantees."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from ollama_sentinel.hf_mapping import hf_search_query_from_tag
from ollama_sentinel.inventory import build_inventory, free_vram_bytes
from ollama_sentinel.mtp_matrix import mtp_platform_note

# Minimum installed generation map (fleet-seeded, maintain manually)
GENERATION_SUPERSEDED: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"qwen2(\.5)?", re.I), "qwen3.5/qwen3.6", "qwen2/qwen2.5 -> qwen3 -> qwen35"),
    (re.compile(r"gemma3", re.I), "gemma4", "gemma3 -> gemma4"),
    (re.compile(r"llama3\.1", re.I), "llama3.2+", "llama3.1 -> 3.2"),
    (re.compile(r"deepseek-r1", re.I), "newer reasoning models", "deepseek-r1 may be superseded"),
    (re.compile(r"glm4", re.I), "newer glm", "glm4 generation may be outdated"),
]

HEAVY_QUANTS = frozenset({"Q8_0", "Q8_1", "F16", "F32", "BF16"})


@dataclass
class AdvisorFinding:
    category: str  # runtime | fit | config | model | name | client | info
    severity: str  # info | warn | unknown
    confidence: str  # high | medium | low
    id: str
    message: str
    remedy: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_version(ver: str | None) -> tuple[int, int, int] | None:
    if not ver:
        return None
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", ver.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _version_lt(current: str | None, required: str) -> bool:
    cur = _parse_version(current)
    req = _parse_version(required)
    if cur is None or req is None:
        return False
    return cur < req


def _is_heavy_quant(quant: str | None) -> bool:
    if not quant:
        return False
    q = quant.upper()
    return q in HEAVY_QUANTS or q.startswith("Q8") or q.startswith("F16")


def _fit_bytes(row: dict[str, Any]) -> int:
    return int(row.get("weight_bytes") or row.get("size") or 0)


def evaluate_advisories(
    snapshot: dict[str, Any],
    *,
    show_by_model: dict[str, dict[str, Any]] | None = None,
    log_cfg: dict[str, str] | None = None,
    keep_alive: str | None = None,
    client_missing: list[tuple[str, str]] | None = None,
    gpu_data_available: bool = True,
) -> list[AdvisorFinding]:
    if not snapshot.get("reachable"):
        return []

    findings: list[AdvisorFinding] = []
    show_by_model = show_by_model or {}
    log_cfg = log_cfg or {}
    server = snapshot.get("server", "local")
    ollama_version = snapshot.get("version")
    inv = build_inventory(snapshot)
    free = free_vram_bytes(snapshot.get("gpus"))
    ctx_cfg = log_cfg.get("OLLAMA_CONTEXT_LENGTH")
    kv_cfg = (log_cfg.get("OLLAMA_KV_CACHE_TYPE") or "f16").lower()
    mtp_note = mtp_platform_note()

    # Client model missing (fleet-wide, attached per snapshot pass)
    if client_missing:
        for cname, model in client_missing:
            findings.append(
                AdvisorFinding(
                    category="client",
                    severity="warn",
                    confidence="high",
                    id=f"config:client_model_missing:{cname}:{model}",
                    message=f"Client {cname} expects {model!r} but it is not installed on any server",
                    remedy="Pull the model or update the client config",
                    evidence={"client": cname, "model": model},
                )
            )

    if not gpu_data_available:
        findings.append(
            AdvisorFinding(
                category="fit",
                severity="info",
                confidence="high",
                id=f"fit:gpu_unknown:{server}",
                message=(
                    f"[{server}] No GPU telemetry — fit advisories use tag size only or are skipped"
                ),
                remedy="Run sentinel on the host with nvidia-smi or rocm-smi for VRAM fit checks",
            )
        )

    loaded_by_name = {
        (m.get("name") or m.get("model")): m
        for m in snapshot.get("models") or []
        if m.get("name") or m.get("model")
    }

    for row in inv:
        name = row["name"]
        show = show_by_model.get(name) or {}
        if show.get("error"):
            findings.append(
                AdvisorFinding(
                    category="model",
                    severity="unknown",
                    confidence="high",
                    id=f"model:show_error:{name}",
                    message=f"{name}: /api/show failed — {show['error']}",
                    remedy="Model metadata unavailable; reinstall or inspect the blob",
                    evidence={"error": show["error"]},
                    model=name,
                )
            )
            continue

        if show:
            q_show = show.get("quantization")
            if q_show and q_show != row.get("quantization"):
                row = dict(row)
                row["quantization"] = q_show
            if show.get("weight_bytes"):
                row = dict(row)
                row["weight_bytes"] = show["weight_bytes"]

        # requires newer ollama
        req = show.get("requires")
        if req and _version_lt(ollama_version, req):
            findings.append(
                AdvisorFinding(
                    category="model",
                    severity="warn",
                    confidence="high",
                    id=f"model:requires_newer_ollama:{name}",
                    message=f"{name} requires Ollama {req} but server is {ollama_version or '?'}",
                    remedy="Upgrade Ollama or use an older model build",
                    evidence={"requires": req, "server_version": ollama_version},
                    model=name,
                )
            )

        # MTP dormant / disabled
        mtp_layers = show.get("mtp_layers")
        draft = show.get("draft_num_predict")
        if mtp_layers and mtp_layers > 0:
            if draft is None or draft == 0:
                findings.append(
                    AdvisorFinding(
                        category="model",
                        severity="warn",
                        confidence="high",
                        id=f"model:mtp_dormant:{name}",
                        message=(
                            f"{name} has MTP layers in GGUF but draft_num_predict is "
                            f"{draft if draft is not None else 'unset'}"
                        ),
                        remedy=(
                            "Add PARAMETER draft_num_predict 4 to the Modelfile and recreate the tag"
                        ),
                        evidence={"mtp_layers": mtp_layers, "draft_num_predict": draft},
                        model=name,
                    )
                )
            elif draft == 0:
                findings.append(
                    AdvisorFinding(
                        category="model",
                        severity="warn",
                        confidence="high",
                        id=f"model:mtp_disabled:{name}",
                        message=f"{name} has MTP tensors but draft_num_predict is 0 (disabled)",
                        remedy="Set draft_num_predict to 4 or remove -mtp from the tag name",
                        evidence={"draft_num_predict": draft},
                        model=name,
                    )
                )
            elif not mtp_note.acceleration_likely:
                findings.append(
                    AdvisorFinding(
                        category="model",
                        severity="info",
                        confidence="medium",
                        id=f"model:mtp_platform:{name}",
                        message=f"{name}: {mtp_note.summary}",
                        remedy="Do not expect speedup without verifying tokens/s on this platform",
                        model=name,
                    )
                )

        if show.get("has_draft_modelfile"):
            findings.append(
                AdvisorFinding(
                    category="model",
                    severity="info",
                    confidence="high",
                    id=f"model:draft_separate:{name}",
                    message=f"{name} Modelfile references a separate DRAFT model",
                    remedy="Confirm draft GGUF path is valid after moves between hosts",
                    model=name,
                )
            )

        num_ctx = show.get("num_ctx")
        if num_ctx and ctx_cfg:
            try:
                if int(num_ctx) != int(ctx_cfg):
                    findings.append(
                        AdvisorFinding(
                            category="model",
                            severity="info",
                            confidence="medium",
                            id=f"model:num_ctx_override:{name}",
                            message=(
                                f"{name} Modelfile num_ctx={num_ctx} differs from server "
                                f"OLLAMA_CONTEXT_LENGTH={ctx_cfg}"
                            ),
                            remedy="Expected for per-model overrides; verify VRAM headroom",
                            evidence={"num_ctx": num_ctx, "OLLAMA_CONTEXT_LENGTH": ctx_cfg},
                            model=name,
                        )
                    )
            except (TypeError, ValueError):
                pass

        # Fit advisories
        fit_bytes = _fit_bytes(row)
        if gpu_data_available and free is not None and not row.get("loaded"):
            if fit_bytes > free:
                quant = row.get("quantization")
                remedy = "Try a lower quant or free VRAM before loading"
                if _is_heavy_quant(quant):
                    remedy = "Consider Q4_K_M or lighter; current quant is heavy for free VRAM"
                findings.append(
                    AdvisorFinding(
                        category="fit",
                        severity="warn",
                        confidence="medium" if row.get("weight_bytes") else "low",
                        id=f"fit:would_spill:{name}",
                        message=(
                            f"{name} ({fit_bytes / 1e9:.1f} GB weights) may not fit in "
                            f"{free / 1e9:.1f} GB free VRAM"
                        ),
                        remedy=remedy,
                        evidence={"weight_bytes": fit_bytes, "free_vram": free, "quant": quant},
                        suggestions=[
                            f"ollama-sentinel search {hf_search_query_from_tag(name, family=row.get('family'))}"
                        ],
                        model=name,
                    )
                )
            elif _is_heavy_quant(row.get("quantization")) and free < fit_bytes * 1.25:
                findings.append(
                    AdvisorFinding(
                        category="fit",
                        severity="info",
                        confidence="medium",
                        id=f"fit:heavy_quant:{name}",
                        message=(
                            f"{name} uses {row.get('quantization')} — tight on "
                            f"{free / 1e9:.1f} GB free VRAM"
                        ),
                        remedy="A lighter quant may leave headroom for KV cache and co-loaded models",
                        model=name,
                    )
                )

        # Generation staleness
        for pattern, newer, lineage in GENERATION_SUPERSEDED:
            if pattern.search(name):
                findings.append(
                    AdvisorFinding(
                        category="name",
                        severity="info",
                        confidence="low",
                        id=f"name:generation_stale:{name}",
                        message=f"{name} may be an older generation ({lineage}); consider {newer}",
                        remedy="Use Discover to compare newer bases — not a requirement",
                        suggestions=[
                            f"ollama-sentinel search {hf_search_query_from_tag(name)} --sort lastModified"
                        ],
                        model=name,
                    )
                )
                break

        if re.search(r"(512k|1m|1000k|uncensored-1m)", name, re.I):
            findings.append(
                AdvisorFinding(
                    category="name",
                    severity="info",
                    confidence="low",
                    id=f"name:extreme_ctx:{name}",
                    message=f"{name} advertises a very large context window — KV cache may dominate VRAM",
                    remedy="Lower num_ctx or OLLAMA_CONTEXT_LENGTH if loads spill",
                    model=name,
                )
            )

    # Runtime: spill pinned
    ka = keep_alive or log_cfg.get("OLLAMA_KEEP_ALIVE")
    pinned = ka is not None and str(ka).strip() in ("-1",)
    for model in snapshot.get("models") or []:
        mn = model.get("name") or model.get("model") or "unknown"
        size = model.get("size") or 0
        sv = model.get("size_vram") or 0
        if size > 0 and sv < size and pinned:
            findings.append(
                AdvisorFinding(
                    category="runtime",
                    severity="warn",
                    confidence="high",
                    id=f"runtime:spill_pinned:{mn}",
                    message=(
                        f"{mn} is spilled to CPU/RAM and KEEP_ALIVE=-1 — it may stay slow until unloaded"
                    ),
                    remedy=f'ollama-sentinel unload "{mn}" -y',
                    evidence={"size": size, "size_vram": sv, "keep_alive": ka},
                    model=mn,
                )
            )

    # KV suboptimal when tight (server-level)
    if (
        gpu_data_available
        and free is not None
        and ctx_cfg
        and int(ctx_cfg) >= 32768
        and kv_cfg in ("f16", "")
        and any(r.get("would_spill") for r in inv)
    ):
        findings.append(
            AdvisorFinding(
                category="config",
                severity="warn",
                confidence="high",
                id=f"config:kv_suboptimal:{server}",
                message=(
                    f"[{server}] OLLAMA_KV_CACHE_TYPE={kv_cfg or 'f16'} with large context and "
                    "tight VRAM — q8_0 may help after restart"
                ),
                remedy="Set OLLAMA_KV_CACHE_TYPE=q8_0 and restart Ollama (server start only)",
                evidence={"OLLAMA_KV_CACHE_LENGTH": ctx_cfg, "OLLAMA_KV_CACHE_TYPE": kv_cfg},
            )
        )

    # Multi-model sum vs VRAM
    if gpu_data_available and free is not None:
        loaded_sum = sum((m.get("size_vram") or m.get("size") or 0) for m in snapshot.get("models") or [])
        total = sum(g.get("memory_total") or 0 for g in snapshot.get("gpus") or [])
        if len(snapshot.get("models") or []) > 1 and total > 0 and loaded_sum > total * 0.92:
            findings.append(
                AdvisorFinding(
                    category="fit",
                    severity="warn",
                    confidence="medium",
                    id=f"fit:multi_model:{server}",
                    message=(
                        f"[{server}] {len(snapshot['models'])} models loaded — "
                        f"{loaded_sum / 1e9:.1f} GB resident vs {total / 1e9:.1f} GB GPU"
                    ),
                    remedy="Co-resident models can evict or spill each other (see process VRAM panel)",
                )
            )

    return findings


def advisor_log_context() -> tuple[dict[str, str], str | None]:
    """Server log env vars for advisor checks (best-effort)."""
    try:
        from ollama_sentinel.doctor import collect_doctor_inputs

        inputs = collect_doctor_inputs()
        log_cfg = inputs.get("log_cfg") or {}
        return log_cfg, log_cfg.get("OLLAMA_KEEP_ALIVE")
    except Exception:
        return {}, None


def evaluate_advisor_alarms(findings: list[AdvisorFinding]) -> list[dict[str, Any]]:
    """Map high/medium warn advisories to passive GUI alarms (not --once exit codes)."""
    out: list[dict[str, Any]] = []
    for f in findings:
        if f.severity != "warn":
            continue
        if f.confidence == "low":
            continue
        out.append(
            {
                "id": f.id,
                "type": "advisor",
                "message": f.message,
            }
        )
    return out


def advisories_for_model(findings: list[AdvisorFinding], model: str) -> list[AdvisorFinding]:
    return [f for f in findings if f.model == model]
