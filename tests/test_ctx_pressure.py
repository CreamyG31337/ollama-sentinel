"""Context-window pressure detection and the client-overcommit check.

Fixture lines are real ones from server.log during the 2026-09-04 truncation
incident, so a regression here means the tool would miss that failure again.
"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from ollama_sentinel.advisor import evaluate_advisories
from ollama_sentinel.client_config import load_client_config, overcommitted_clients
from ollama_sentinel.client_probe import _parse_simple_yaml_ints, resolve_client_context
from ollama_sentinel.ctx_pressure import (
    CtxPressureReport,
    find_ladder,
    parse_ctx_pressure,
    read_tail,
)

PROMPT = (
    "slot   operator(): id  0 | task {t} | new prompt, "
    "n_ctx_slot = 65536, n_keep = 4, task.n_tokens = {n}"
)
RELEASE = "slot      release: id  0 | task {t} | stop processing: n_tokens = {n}, truncated = {tr}"
KV_SHIFT = (
    "cmn  common_init_: KV cache shifting is not supported for this context, "
    "disabling KV cache shifting"
)

HAS_YAML = importlib.util.find_spec("yaml") is not None


def log(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def prompts(*sizes: int) -> str:
    return log(*(PROMPT.format(t=i, n=n) for i, n in enumerate(sizes)))


def snapshot() -> dict:
    return {"reachable": True, "server": "local", "models": [], "tags": [], "gpus": []}


def finding_ids(**kwargs) -> set[str]:
    kwargs.setdefault("gpu_data_available", False)
    return {f.id for f in evaluate_advisories(snapshot(), **kwargs)}


class ParseTest(unittest.TestCase):
    def test_prompt_headroom(self):
        report = parse_ctx_pressure(prompts(65506))
        self.assertEqual(report.n_ctx_slot, 65536)
        self.assertEqual(report.prompts[0].headroom, 30)
        self.assertGreater(report.prompts[0].fill, 0.999)

    def test_truncated_flag_is_ground_truth(self):
        report = parse_ctx_pressure(
            log(
                RELEASE.format(t=1, n=500, tr=0),
                RELEASE.format(t=2, n=65535, tr=1),
                RELEASE.format(t=3, n=65535, tr=1),
            )
        )
        self.assertEqual(report.truncated_count, 2)
        self.assertEqual(report.truncated_tasks, [2, 3])

    def test_kv_shift_detection(self):
        self.assertTrue(parse_ctx_pressure(log(KV_SHIFT)).kv_shift_disabled)
        self.assertFalse(parse_ctx_pressure(prompts(10)).kv_shift_disabled)

    def test_zero_slot_does_not_divide_by_zero(self):
        line = (
            "slot operator(): id 0 | task 1 | new prompt, n_ctx_slot = 0, "
            "n_keep = 4, task.n_tokens = 5"
        )
        self.assertEqual(parse_ctx_pressure(log(line)).prompts[0].fill, 0.0)


class RecencyTest(unittest.TestCase):
    """A fixed problem has to stop alarming, or people learn to ignore the tool."""

    def test_only_the_current_runner_is_considered(self):
        # Task numbers restart at 0 on reload, so mixing runners would compare
        # unrelated requests — and keep a pre-restart incident alive forever.
        text = log(
            PROMPT.format(t=1, n=65500),
            RELEASE.format(t=1, n=65535, tr=1),
            "srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536",
            PROMPT.format(t=1, n=1000),
            RELEASE.format(t=1, n=1200, tr=0),
        )
        report = parse_ctx_pressure(text)
        self.assertEqual(report.truncated_count, 0)
        self.assertEqual(len(report.prompts), 1)

    def test_old_truncations_age_out_of_the_window(self):
        lines = [PROMPT.format(t=0, n=65500), RELEASE.format(t=0, n=65535, tr=1)]
        for i in range(1, 12):
            lines.append(PROMPT.format(t=i, n=1000))
            lines.append(RELEASE.format(t=i, n=1200, tr=0))
        report = parse_ctx_pressure(log(*lines), recent=5)
        self.assertEqual(report.truncated_count, 0)

    def test_a_truncation_inside_the_window_still_counts(self):
        lines = [PROMPT.format(t=0, n=65500), RELEASE.format(t=0, n=65535, tr=1)]
        for i in range(1, 3):
            lines.append(PROMPT.format(t=i, n=1000))
        report = parse_ctx_pressure(log(*lines), recent=5)
        self.assertEqual(report.truncated_count, 1)

    def test_kv_shift_survives_runner_scoping(self):
        # The notice is printed once at load, before the requests it applies to.
        text = log(KV_SHIFT, "srv    load_model: initializing", PROMPT.format(t=1, n=65500))
        self.assertTrue(parse_ctx_pressure(text).kv_shift_disabled)


class LadderTest(unittest.TestCase):
    """The retry ladder is the incident's distinctive signature."""

    def test_climbing_near_ceiling_prompts_are_a_ladder(self):
        report = parse_ctx_pressure(prompts(65303, 65358, 65409, 65460, 65506))
        self.assertEqual(
            [p.n_tokens for p in report.ladder], [65303, 65358, 65409, 65460, 65506]
        )

    def test_single_big_prompt_is_not_a_ladder(self):
        # One full prompt is a different problem, with a different remedy.
        self.assertEqual(parse_ctx_pressure(prompts(65500)).ladder, [])

    def test_client_backing_off_is_not_a_ladder(self):
        # Shrinking history is the healthy case and must stay silent.
        self.assertEqual(parse_ctx_pressure(prompts(65500, 65400, 65300)).ladder, [])

    def test_growth_far_below_ceiling_is_just_a_conversation(self):
        self.assertEqual(parse_ctx_pressure(prompts(1000, 2000, 3000, 4000)).ladder, [])

    def test_minimum_run_length_is_enforced(self):
        report = parse_ctx_pressure(prompts(65400, 65450))
        self.assertEqual(report.ladder, [])
        self.assertTrue(find_ladder(report.prompts, min_len=2))


class ReadTailTest(unittest.TestCase):
    def test_drops_partial_first_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "server.log"
            p.write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
            self.assertTrue(read_tail(p, max_bytes=8).startswith("cccc"))

    def test_missing_file_is_empty(self):
        self.assertEqual(read_tail(Path("nope-does-not-exist.log")), "")


class AdvisorTest(unittest.TestCase):
    def test_reports_truncation_ladder_and_kv_shift(self):
        ctx = parse_ctx_pressure(
            log(
                *(PROMPT.format(t=i, n=n) for i, n in enumerate([65303, 65358, 65409, 65460])),
                RELEASE.format(t=9, n=65535, tr=1),
                KV_SHIFT,
            )
        )
        ids = finding_ids(ctx=ctx)
        self.assertIn("runtime:ctx_truncated:local", ids)
        self.assertIn("runtime:ctx_retry_ladder:local", ids)
        self.assertIn("config:kv_shift_disabled:local", ids)

    def test_headroom_warns_only_before_truncation_is_observed(self):
        ids = finding_ids(ctx=parse_ctx_pressure(prompts(64000)))
        self.assertIn("runtime:ctx_headroom:local", ids)

        # Once truncation is a fact, the leading indicator is noise.
        hit = parse_ctx_pressure(
            log(PROMPT.format(t=1, n=64000), RELEASE.format(t=1, n=65535, tr=1))
        )
        ids = finding_ids(ctx=hit)
        self.assertNotIn("runtime:ctx_headroom:local", ids)
        self.assertIn("runtime:ctx_truncated:local", ids)

    def test_quiet_when_nothing_presses_on_the_window(self):
        ctx = parse_ctx_pressure(
            log(PROMPT.format(t=1, n=1000), RELEASE.format(t=1, n=1200, tr=0))
        )
        ids = finding_ids(ctx=ctx)
        self.assertFalse(any(i.startswith(("runtime:ctx_", "config:kv_shift")) for i in ids))

    def test_kv_shift_alone_is_not_worth_reporting(self):
        ctx = parse_ctx_pressure(log(PROMPT.format(t=1, n=1000), KV_SHIFT))
        self.assertNotIn("config:kv_shift_disabled:local", finding_ids(ctx=ctx))

    def test_no_log_means_no_findings(self):
        ids = finding_ids(ctx=CtxPressureReport())
        self.assertFalse(any(i.startswith("runtime:ctx_") for i in ids))

    def test_overcommit_finding_names_both_windows(self):
        findings = evaluate_advisories(
            snapshot(),
            log_cfg={"OLLAMA_CONTEXT_LENGTH": "65536"},
            client_overcommitted=[("hermes", 262144, "file")],
            gpu_data_available=False,
        )
        hit = [f for f in findings if f.id == "client:ctx_overcommit:hermes"]
        self.assertTrue(hit)
        self.assertIn("65,536", hit[0].message)
        self.assertIn("262,144", hit[0].message)


    def test_overcommit_needs_a_known_served_window(self):
        # Unprovable is not the same as passing — and must not crash on formatting.
        for cfg in ({}, {"OLLAMA_CONTEXT_LENGTH": "not-a-number"}):
            ids = finding_ids(
                log_cfg=cfg, client_overcommitted=[("hermes", 262144, "file")]
            )
            self.assertNotIn("client:ctx_overcommit:hermes", ids)


class OvercommitTest(unittest.TestCase):
    def test_declared_window_over_served(self):
        clients = [{"name": "hermes", "models": [], "context_length": 262144}]
        self.assertEqual(
            overcommitted_clients(clients, 65536), [("hermes", 262144, "declared")]
        )

    def test_client_within_window_is_silent(self):
        clients = [{"name": "hermes", "models": [], "context_length": 65536}]
        self.assertEqual(overcommitted_clients(clients, 65536), [])

    def test_unprovable_without_a_served_window(self):
        clients = [{"name": "hermes", "models": [], "context_length": 262144}]
        self.assertEqual(overcommitted_clients(clients, None), [])
        self.assertEqual(overcommitted_clients(clients, 0), [])

    def test_config_loader_parses_and_rejects(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "clients.json"
            p.write_text(
                '{"clients": [{"name": "hermes", "models": ["m"], "context_length": 65536},'
                ' {"name": "plain", "models": ["m"]},'
                ' {"name": "bad", "models": ["m"], "context_length": "65K"},'
                ' {"name": "probe", "models": [], "context_length_file": "x.yaml",'
                '  "context_length_key": "context_lengths",'
                '  "context_length_match": "localhost:11434"}]}',
                encoding="utf-8",
            )
            by_name = {c["name"]: c for c in load_client_config(p)}
        self.assertEqual(by_name["hermes"]["context_length"], 65536)
        self.assertNotIn("context_length", by_name["plain"])
        # Unparseable is dropped rather than guessed at.
        self.assertNotIn("context_length", by_name["bad"])
        self.assertEqual(by_name["probe"]["context_length_key"], "context_lengths")
        self.assertEqual(by_name["probe"]["context_length_match"], "localhost:11434")


class ProbeTest(unittest.TestCase):
    """Prevention has to survive drift: read the client's own config, not a claim."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, text: str) -> str:
        p = Path(self._tmp.name) / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_reads_hermes_style_cache(self):
        # Hermes rewrites this file on every re-probe, so a hand-fix silently reverts.
        path = self.write(
            "context_length_cache.yaml",
            "context_lengths:\n"
            "  qwen3.8:27b-heretic@http://localhost:11434/v1: 262144\n"
            "  other:model@http://localhost:11434/v1: 8192\n",
        )
        client = {
            "name": "hermes",
            "context_length_file": path,
            "context_length_key": "context_lengths",
        }
        self.assertEqual(resolve_client_context(client), (262144, "file"))
        self.assertEqual(
            overcommitted_clients([client], 65536), [("hermes", 262144, "file")]
        )

    def test_goes_quiet_once_the_file_is_corrected(self):
        path = self.write("cache.yaml", "context_lengths:\n  m@u: 65536\n")
        client = {
            "name": "hermes",
            "context_length_file": path,
            "context_length_key": "context_lengths",
        }
        self.assertEqual(overcommitted_clients([client], 65536), [])

    def test_json_with_dotted_key(self):
        path = self.write("settings.json", '{"model": {"context_length": 131072}}')
        client = {
            "name": "app",
            "context_length_file": path,
            "context_length_key": "model.context_length",
        }
        self.assertEqual(resolve_client_context(client), (131072, "file"))

    def test_file_beats_a_stale_declared_value(self):
        path = self.write("settings.json", '{"context_length": 262144}')
        client = {"name": "app", "context_length": 65536, "context_length_file": path}
        self.assertEqual(resolve_client_context(client), (262144, "file"))

    def test_falls_back_to_declared_when_file_is_missing(self):
        client = {
            "name": "app",
            "context_length": 65536,
            "context_length_file": str(Path(self._tmp.name) / "gone.json"),
        }
        self.assertEqual(resolve_client_context(client), (65536, "declared"))

    def test_unreadable_file_is_unknown_not_agreement(self):
        path = self.write("broken.json", "{not json")
        client = {"name": "app", "context_length_file": path}
        self.assertEqual(resolve_client_context(client), (None, "unknown"))
        self.assertEqual(overcommitted_clients([client], 65536), [])

    def test_booleans_are_not_context_windows(self):
        path = self.write("s.json", '{"enabled": true, "verbose": true}')
        client = {"name": "a", "context_length_file": path}
        self.assertEqual(resolve_client_context(client), (None, "unknown"))

    def test_missing_key_is_unknown(self):
        path = self.write("s.json", '{"model": {"name": "x"}}')
        client = {
            "name": "a",
            "context_length_file": path,
            "context_length_key": "model.ctx",
        }
        self.assertEqual(resolve_client_context(client), (None, "unknown"))

    def test_match_filter_prevents_cross_provider_false_positive(self):
        """Hermes caches every provider in one file; a cloud model's 400k window
        must not be compared against a 65k local server."""
        path = self.write(
            "mixed.yaml",
            "context_lengths:\n"
            "  qwen3.8:27b@http://localhost:11434/v1: 65536\n"
            "  gpt-5@https://api.openai.com/v1: 400000\n",
        )
        base = {
            "name": "hermes",
            "context_length_file": path,
            "context_length_key": "context_lengths",
        }
        # Without the filter the cloud entry wins and reports a problem that is not real.
        self.assertEqual(resolve_client_context(base), (400000, "file"))
        scoped = dict(base, context_length_match="localhost:11434")
        self.assertEqual(resolve_client_context(scoped), (65536, "file"))
        self.assertEqual(overcommitted_clients([scoped], 65536), [])

    def test_match_filter_with_no_hits_is_unknown(self):
        path = self.write(
            "c.yaml", "context_lengths:\n  m@http://elsewhere/v1: 262144\n"
        )
        client = {
            "name": "hermes",
            "context_length_file": path,
            "context_length_key": "context_lengths",
            "context_length_match": "localhost:11434",
        }
        self.assertEqual(resolve_client_context(client), (None, "unknown"))

    def test_no_probe_configured(self):
        self.assertEqual(resolve_client_context({"name": "a"}), (None, "unknown"))


class YamlFallbackTest(unittest.TestCase):
    """PyYAML is not a project dependency, so the fallback must handle the real shape."""

    def test_keys_containing_colons_and_urls(self):
        # A key like model@http://host:11434/v1 means splitting on the first colon fails.
        got = _parse_simple_yaml_ints(
            "context_lengths:\n  qwen3.8:27b-heretic@http://localhost:11434/v1: 262144\n"
        )
        self.assertEqual(
            got["context_lengths"]["qwen3.8:27b-heretic@http://localhost:11434/v1"], 262144
        )

    def test_top_level_scalars_and_comments(self):
        got = _parse_simple_yaml_ints("# a comment\nctx: 4096\n\nother: 12\n")
        self.assertEqual(got["ctx"], 4096)
        self.assertEqual(got["other"], 12)

    def test_non_integer_values_are_ignored(self):
        got = _parse_simple_yaml_ints("model:\n  name: qwen\n  context_length: 65536\n")
        self.assertEqual(got["model"], {"context_length": 65536})

    @unittest.skipUnless(HAS_YAML, "PyYAML not installed")
    def test_agrees_with_pyyaml_on_the_real_shape(self):
        import yaml

        text = (
            "context_lengths:\n"
            "  qwen3.8:27b-heretic@http://localhost:11434/v1: 262144\n"
            "  nomic-embed-text@http://localhost:11434/v1: 8192\n"
        )
        self.assertEqual(_parse_simple_yaml_ints(text), yaml.safe_load(text))


if __name__ == "__main__":
    unittest.main()
