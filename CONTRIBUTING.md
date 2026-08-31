# Contributing

- Run tests: `python -m unittest discover -s tests`
- No GPU, live Ollama, or Hugging Face token required for tests
- Keep alarm logic pure and testable in `ollama_sentinel/alarms.py`
- Never commit `.env`, `servers.json`, or `AGENTS.local.md`
