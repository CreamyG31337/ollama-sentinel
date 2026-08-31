# ollama-sentinel

A cross-platform companion for Ollama on NVIDIA GPUs: spill / paging / VRAM alarms,
installed-vs-loaded library view, Hugging Face discover, and opt-in pull. Not a chat UI.

If `AGENTS.local.md` exists in this folder, read it for host-specific facts (GPU, Python path,
Ollama env). That file is gitignored.

See `PROMPT.md` for the original alarm spec.

## Two traps to respect

1. **`OLLAMA_HOST` is often a bind address (`0.0.0.0:11434`).** Never treat it as a connect
   address — always dial `http://127.0.0.1:11434` unless the user overrides in `.env`.
2. **`nvidia-smi` returns `[N/A]` for per-process VRAM on Windows.** Use whole-GPU queries only.
