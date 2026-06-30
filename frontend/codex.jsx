/* PiKaOs — CODEX (knowledge base): type metadata, seed bodies, the codex
   drawer + add-note modal, and the Codex screen. */
import React from 'react';
const { useState, useEffect, useRef } = React;
import { Btn, Empty, HelpNote, PageHead, Panel } from '../../components/components.jsx';
import { deleteDocument, getDocument, listDocuments, reindexKnowledge, searchKnowledge, uploadDocument } from '../../lib/api.js';
import { KNOWLEDGE, byId } from '../../data/data.jsx';
import { Field, Segmented, TagInput } from '../../screens/screens-builder.jsx';
import { RichBody } from '../../components/doc-editor.jsx';

/* ---------------- CODEX (knowledge) — fully working ---------------- */
const KTYPE = { diagram: "🗺️", research: "🔬", doc: "📄", decision: "⚖️", note: "📝" };
const KTYPE_TH = { diagram: "แผนภาพ", research: "งานวิจัย", doc: "เอกสาร", decision: "การตัดสินใจ", note: "บันทึก" };
const KTYPE_EN = { diagram: "Diagram", research: "Research", doc: "Document", decision: "Decision", note: "Note" };
const KTYPE_OPTS = ["doc", "research", "diagram", "decision", "note"];
let _ct = (k) => k;
const ct = (k, v) => _ct(k, v);
function ktypeLabel(type) { return ct("ktype." + type); }
const KBODY = {
  k1: "สถาปัตยกรรมของ auth-service แบ่งเป็น 3 ชั้น: API gateway, token service และ user store. ใช้ rotating refresh token อายุ 7 วัน และ access token อายุ 15 นาที",
  k2: "จากการทดลอง hybrid search (BM25 + vector) ร่วมกับ reranking พบว่าให้ผลแม่นยำกว่า ~14% บนชุดข้อมูลขององค์กร และควรใช้กับเอกสารที่ยาว",
  k3: "ขั้นตอนเริ่มต้นสำหรับเอเจนต์ใหม่: สร้างตัวละคร → รับงานแรก → เข้าร่วมห้องประชุมกลาง",
  k4: "มาตรฐานการเขียน test: ครอบคลุม edge case, ตั้งชื่อทดสอบให้สื่อความ, รายงานความล้มเหลวพร้อมขั้นตอนทำซ้ำ",
  k5: "บันทึกการตัดสินใจ: เลือก rotating refresh token แทน long-lived token เพื่อลดความเสี่ยงหาก token รั่วไหล",
  k6: "รายการ dependency ที่มีช่องโหว่ระดับสูง ควรอัปเดตก่อนปล่อยเวอร์ชันถัดไป",
};

const KCODEX_KEY = "guild-codex-v1";
function loadCodex() {
  try { const raw = localStorage.getItem(KCODEX_KEY); return raw ? JSON.parse(raw) : []; } catch { return []; }
}
function saveCodex(arr) { try { localStorage.setItem(KCODEX_KEY, JSON.stringify(arr)); } catch {} }

function CodexDrawer({ k, onClose }) {
  const by = byId(k.by);
  return (
    <div className="drawer-overlay" onClick={onClose}>
      <div className="drawer" onClick={e => e.stopPropagation()}>
        <div className="drawer-head">
          <span className="codex-type" style={{ width: 48, height: 48, flexBasis: 48, fontSize: 22 }}>{KTYPE[k.type] || "📝"}</span>
          <div style={{ flex: 1 }}>
            <div className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>{ktypeLabel(k.type)} · {ct("codex.updated")} {k.updated}</div>
            <h2 style={{ fontFamily: "var(--font-head)", fontSize: 19, margin: "5px 0 0", color: "var(--ink)", lineHeight: 1.3 }}>{k.title}</h2>
          </div>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>
        <div className="drawer-body">
          {k.bodyHtml
            ? <div style={{ margin: 0, color: "var(--ink-2)", fontSize: 14, lineHeight: 1.7 }} dangerouslySetInnerHTML={{ __html: k.bodyHtml }} />
            : <p style={{ margin: 0, color: "var(--ink-2)", fontSize: 14, lineHeight: 1.7 }}>{k.body || KBODY[k.id] || "— ยังไม่มีรายละเอียดเพิ่มเติม —"}</p>}
          <div className="kv">
            <div className="kv-item"><div className="kv-label">บันทึกโดย</div><div className="kv-val" style={{ fontSize: 14 }}>{by ? by.name : "ศูนย์ควบคุมกลาง"}</div></div>
            <div className="kv-item"><div className="kv-label">การอ้างอิง</div><div className="kv-val">{k.refs ?? 0}</div></div>
          </div>
          <div>
            <div className="kicker" style={{ marginBottom: 10 }}>ป้ายกำกับ</div>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>{(k.tags || []).map(t => <span key={t} className="tag">{t}</span>)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function AddNoteModal({ onSave, onClose }) {
  const [f, setF] = useState({ title: "", type: "doc", body: "", tags: [] });
  const set = (k, v) => setF(p => ({ ...p, [k]: v }));
  const can = f.title.trim().length > 0;
  return (
    <div className="drawer-overlay" onClick={onClose} style={{ justifyContent: "center", alignItems: "center", padding: 24 }}>
      <div className="builder ornate" style={{ width: 520 }} onClick={e => e.stopPropagation()}>
        <div className="builder-head">
          <span className="ph-icon" style={{ fontSize: 18 }}>📚</span>
          <div><div className="kicker">{ct("codex.addKicker")}</div>
            <h2 style={{ fontFamily: "var(--font-head)", fontSize: 18, margin: "2px 0 0", color: "var(--ink)" }}>{ct("codex.addTitle")}</h2></div>
          <button className="drawer-close" onClick={onClose} style={{ marginLeft: "auto" }}>✕</button>
        </div>
        <div style={{ padding: 22, display: "flex", flexDirection: "column", gap: 16 }}>
          <Field label={ct("codex.f.title")}><input className="bf-input" value={f.title} onChange={e => set("title", e.target.value)} placeholder={ct("codex.f.titlePh")} /></Field>
          <Field label={ct("codex.f.type")}><Segmented value={f.type} onChange={v => set("type", v)} options={KTYPE_OPTS.map(ty => ({ key: ty, label: ktypeLabel(ty) }))} /></Field>
          <Field label={ct("codex.f.body")} hint={ct("codex.f.bodyHint")}><RichBody value={f.bodyHtml || f.body} onChange={(text, html) => { set("body", text); set("bodyHtml", html); }} placeholder={ct("codex.f.bodyPh")} /></Field>
          <Field label={ct("codex.f.tags")} hint={ct("codex.f.tagsHint")}><TagInput tags={f.tags} onChange={v => set("tags", v)} suggest={["backend","security","docs","research","qa","rag"]} placeholder={ct("codex.f.tagsPh")} /></Field>
        </div>
        <div className="builder-foot">
          <Btn kind="ghost" onClick={onClose}>{ct("common.cancel")}</Btn>
          <Btn kind="gold" icon="✓" style={{ opacity: can ? 1 : .5, pointerEvents: can ? "auto" : "none" }}
            onClick={() => onSave({ id: "ku" + Date.now(), title: f.title.trim(), type: f.type, body: f.body.trim(), bodyHtml: f.bodyHtml || "", tags: f.tags, by: (window.__chars || [])[0]?.id, updated: ct("codex.justNow"), refs: 0 })}>{ct("codex.saveBtn")}</Btn>
        </div>
      </div>
    </div>
  );
}

/* ---------------- CODEX · เอกสาร (live) — backed by /api/knowledge (E4) ----------------
   The markdown-as-truth document store + RAG search. Files live in MinIO; the backend
   chunks + embeds them in the background (ingest_status). Upload/reindex gate on codex.manage,
   delete on codex.delete; reading is open to any user holding codex.view. */
const DOC_ICON = { md: "📝", image: "🖼️", pdf: "📕", log: "📄", other: "📦" };
const INGEST_ICON = { pending: "⏳", done: "✓", failed: "✕", skipped: "—" };

function fmtSize(n) {
  n = n || 0;
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(1) + " MB";
}

function CodexDocs({ t, can }) {
  const tx = (typeof t === "function") ? t : ((k) => k);
  const mayManage = !can || can("codex.manage");   // upload + reindex
  const mayDelete = !can || can("codex.delete");
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");             // transient success line (e.g. reindex result)
  const [q, setQ] = useState("");
  const [results, setResults] = useState(null);     // null = browse docs · array = search hits
  const fileRef = useRef(null);

  const load = async () => {
    setLoading(true); setErr("");
    try { const r = await listDocuments({ limit: 100 }); setDocs(r.items || []); }
    catch (e) { setErr(e.message || tx("codex.docs.err")); }
    finally { setLoading(false); }
  };
  useEffect(() => { load(); }, []);

  const onPick = async (e) => {
    const file = e.target.files && e.target.files[0];
    e.target.value = "";                            // let the same file be re-picked
    if (!file) return;
    setBusy(true); setErr("");
    try { await uploadDocument(file); await load(); }
    catch (e2) { setErr(e2.message || tx("codex.docs.err")); }
    finally { setBusy(false); }
  };

  const del = async (d) => {
    const ok = window.uiConfirm
      ? await window.uiConfirm({ title: tx("codex.docs.delConfirm"), message: d.name, danger: true, confirmText: tx("common.delete") })
      : window.confirm(tx("codex.docs.delConfirm"));
    if (!ok) return;
    setErr("");
    try { await deleteDocument(d.id); await load(); }
    catch (e) { setErr(e.message || tx("codex.docs.err")); }
  };

  const open = async (d) => {
    try { const full = await getDocument(d.id); if (full.url) window.open(full.url, "_blank", "noopener"); }
    catch (e) { setErr(e.message || tx("codex.docs.err")); }
  };

  const reindex = async () => {
    setBusy(true); setErr(""); setNote("");
    try {
      const r = await reindexKnowledge(true);       // only re-embed docs not on the current model
      setNote(tx("codex.docs.reindexDone").replace("{n}", r.queued).replace("{model}", r.model));
      await load();                                  // ingest runs in the worker; statuses settle shortly
    } catch (e) { setErr(e.message || tx("codex.docs.err")); }
    finally { setBusy(false); }
  };

  const doSearch = async () => {
    const query = q.trim();
    if (!query) { setResults(null); return; }
    setBusy(true); setErr("");
    try { const r = await searchKnowledge(query); setResults(r.items || []); }
    catch (e) { setErr(e.message || tx("codex.docs.err")); }
    finally { setBusy(false); }
  };

  return (
    <div>
      <div className="search-bar" style={{ margin: "16px 0" }}>
        <span>🔍</span>
        <input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => { if (e.key === "Enter") doSearch(); }}
          placeholder={tx("codex.docs.searchPh")} />
        <Btn sm onClick={doSearch} disabled={busy || !q.trim()}>{tx("codex.docs.searchBtn")}</Btn>
        {results != null && <Btn sm kind="ghost" onClick={() => { setQ(""); setResults(null); }}>{tx("codex.docs.clear")}</Btn>}
      </div>

      {mayManage && (
        <div style={{ marginBottom: 12, display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
          <input ref={fileRef} type="file" hidden onChange={onPick} />
          <Btn kind="gold" sm icon="⬆️" disabled={busy} onClick={() => fileRef.current && fileRef.current.click()}>
            {busy ? tx("codex.docs.uploading") : tx("codex.docs.upload")}</Btn>
          <Btn kind="ghost" sm icon="🔄" disabled={busy} onClick={reindex} title={tx("codex.docs.reindexHint")}>
            {tx("codex.docs.reindex")}</Btn>
          <span className="mono faint" style={{ fontSize: 11 }}>{tx("codex.docs.uploadHint")}</span>
        </div>
      )}

      {err && <div className="muted" style={{ color: "var(--danger,#c0392b)", fontSize: 12.5, padding: "6px 2px" }} data-no-lex>{err}</div>}
      {note && <div className="muted" style={{ color: "var(--gold,#a87f2e)", fontSize: 12.5, padding: "6px 2px" }} data-no-lex>{note}</div>}

      {results != null ? (
        results.length === 0 ? <Panel><Empty icon="🔍" title={tx("codex.docs.noHit")} /></Panel> : (
          <div className="list-rows stagger">
            {results.map(r => (
              <div key={r.id} className="codex-row" style={{ cursor: "default" }}>
                <span className="codex-type">🔎</span>
                <div className="codex-main">
                  <div className="codex-title">{r.document_name}{r.heading ? ` — ${r.heading}` : ""}</div>
                  <div className="codex-meta" style={{ whiteSpace: "normal" }}>{r.content}</div>
                </div>
                <span className="chip" data-no-lex>{Math.round((r.score || 0) * 100)}%</span>
              </div>
            ))}
          </div>
        )
      ) : loading ? (
        <div className="muted" style={{ fontSize: 13, padding: "10px 2px" }}>{tx("codex.docs.loading")}</div>
      ) : docs.length === 0 ? (
        <Panel><Empty icon="📚" title={tx("codex.docs.empty")} sub={mayManage ? tx("codex.docs.emptySub") : ""} /></Panel>
      ) : (
        <div className="list-rows stagger">
          {docs.map(d => (
            <div key={d.id} className="codex-row" style={{ cursor: "default" }}>
              <span className="codex-type">{DOC_ICON[d.kind] || "📦"}</span>
              <button className="codex-main" style={{ textAlign: "left", background: "none", border: 0, padding: 0, cursor: "pointer" }} onClick={() => open(d)}>
                <div className="codex-title">{d.name}</div>
                <div className="codex-meta">{d.kind} · {fmtSize(d.size)} · <span className="chip" data-no-lex>{INGEST_ICON[d.ingest_status] || "⏳"} {tx("codex.docs.ingest." + d.ingest_status)}</span></div>
              </button>
              {mayDelete && <button type="button" className="chip-act danger" title={tx("common.delete")} onClick={() => del(d)}>✕</button>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Codex({ t, can }) {
  _ct = (typeof t === "function") ? t : ((k) => k);
  const [mode, setMode] = useState("notes");        // notes (local) | docs (live · API)
  const [extra, setExtra] = useState(() => loadCodex());
  const [q, setQ] = useState("");
  const [filter, setFilter] = useState("all");
  const [sel, setSel] = useState(null);
  const [adding, setAdding] = useState(false);
  const all = [...extra, ...KNOWLEDGE];
  const ql = q.trim().toLowerCase();
  const list = all.filter(k => {
    if (filter !== "all" && k.type !== filter) return false;
    if (!ql) return true;
    return (k.title || "").toLowerCase().includes(ql) || (k.tags || []).some(t => t.toLowerCase().includes(ql)) || (KBODY[k.id] || k.body || "").toLowerCase().includes(ql);
  });
  const addNote = (note) => { const next = [note, ...extra]; setExtra(next); saveCodex(next); setAdding(false); };
  const tabs = [["all", ct("codex.allTab")], ...KTYPE_OPTS.map(ty => [ty, ktypeLabel(ty)])];

  const isDocs = mode === "docs";
  return (
    <div className="content-pad fade-in">
      <PageHead kicker={ct("codex.kicker")} title={ct("codex.title")} tag={isDocs ? "live" : "local"}
        desc={ct("codex.desc")}
        actions={isDocs ? undefined : <Btn kind="gold" sm icon="➕" onClick={() => setAdding(true)}>{ct("codex.add")}</Btn>} />
      <HelpNote tag={isDocs ? "live" : "local"}>{ct(isDocs ? "codex.docs.help" : "codex.help")}</HelpNote>
      <div style={{ margin: "14px 0 4px" }}>
        <Segmented value={mode} onChange={setMode}
          options={[{ key: "notes", label: ct("codex.mode.notes") }, { key: "docs", label: ct("codex.mode.docs") }]} />
      </div>
      {isDocs ? (
        <CodexDocs t={t} can={can} />
      ) : (
        <>
          <div className="search-bar" style={{ margin: "16px 0" }}>
            <span>🔍</span><input value={q} onChange={e => setQ(e.target.value)} placeholder={ct("codex.searchPh")} />
            <span className="mono faint" style={{ fontSize: 11 }}>{list.length}/{all.length} {ct("codex.items")}</span>
          </div>
          <div className="tabs" style={{ marginBottom: 16 }}>{tabs.map(([k, l]) => <button key={k} className={`tab ${filter === k ? "active" : ""}`} onClick={() => setFilter(k)}>{l}</button>)}</div>
          {list.length === 0 ? (
            <Panel><Empty icon="🔍" title={ct("codex.noMatch")} sub={ct("codex.noMatchSub")} /></Panel>
          ) : (
            <div className="list-rows stagger">
              {list.map(k => {
                const by = byId(k.by);
                return (
                  <button key={k.id} className="codex-row" onClick={() => setSel(k)}>
                    <span className="codex-type">{KTYPE[k.type] || "📝"}</span>
                    <div className="codex-main">
                      <div className="codex-title">{k.title}</div>
                      <div className="codex-meta">{ktypeLabel(k.type)} · {ct("codex.by")} {by ? by.name.split(" ")[0] : ct("codex.guild")} · {ct("codex.updated")} {k.updated} · {k.refs ?? 0} {ct("codex.refs")}</div>
                    </div>
                    <div className="row" style={{ gap: 6 }}>{(k.tags || []).slice(0, 2).map(t => <span key={t} className="tag">{t}</span>)}</div>
                  </button>
                );
              })}
            </div>
          )}
        </>
      )}
      {sel && <CodexDrawer k={sel} onClose={() => setSel(null)} />}
      {adding && <AddNoteModal onSave={addNote} onClose={() => setAdding(false)} />}
    </div>
  );
}

export { KTYPE, KTYPE_TH, KTYPE_EN, KTYPE_OPTS, ktypeLabel, KBODY, KCODEX_KEY, loadCodex, saveCodex, CodexDrawer, AddNoteModal, Codex };
