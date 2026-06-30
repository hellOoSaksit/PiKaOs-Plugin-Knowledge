# PiKaOs Plugin — Knowledge

RAG / document-knowledge plugin for [PiKaOs](https://github.com/hellOoSaksit/PiKaOs). Ingests documents,
chunks + embeds them, and answers grounded questions (the `knowledge.Retriever` contract other plugins
consume).

- `backend/` — FastAPI router + services (ingestion, chunking, retrieval, answer), `manifest.json`,
  `config.schema.json`. Plugin id: **`knowledge`**.
- `frontend/` — React screens (Codex, Recall) + per-plugin i18n.

## Install

This is a PiKaOs **plugin**, not a standalone app — it is dropped into a PiKaOs Core install (its code lives
outside Core in the external plugin root, never inside Core's source). It depends on Core contracts (e.g.
`ai.LLM`); enable it from the PiKaOs Modules / install page. See PiKaOs Core docs for the plugin lifecycle.
