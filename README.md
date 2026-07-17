# PiKaOs Plugin — Knowledge

RAG / document-knowledge plugin for [PiKaOs](https://github.com/hellOoSaksit/PiKaOs). Ingests documents,
chunks + embeds them, and answers grounded questions (the `knowledge.Retriever` contract other plugins
consume).

- `backend/` — FastAPI router + services (ingestion, chunking, retrieval, answer), `manifest.json`,
  `config.schema.json`. Plugin id: **`knowledge`**.
- `frontend/` — **retired 2026-07-17, backend-only for now.** The Codex/Recall screens were a
  pre-plugin-era prototype: they merged mock seed data into live API results, and they imported two Core
  modules that no longer exist (`screens/screens-builder.jsx`, deleted when the game screens went; the
  `KNOWLEDGE`/`byId` seed exports, deleted with them). Because Core's frontend registry globs every linked
  plugin, those dead imports took Core's whole dev server down with them — the screens weren't merely
  stale, they were a landmine for anyone linking this plugin. The UI gets rebuilt on the U1 primitive
  standard when the RAG v2 design lands; until then `git show HEAD~1:frontend/codex.jsx` is the reference.

## Install

This is a PiKaOs **plugin**, not a standalone app — it is dropped into a PiKaOs Core install (its code lives
outside Core in the external plugin root, never inside Core's source). It depends on Core contracts (e.g.
`ai.LLM`); enable it from the PiKaOs Modules / install page. See PiKaOs Core docs for the plugin lifecycle.
