/* PiKaOs — RECALL: hybrid retrieval (keyword + meaning) over the Codex,
   plus a cited Q&A answer panel grounded in retrieved docs. */
import React from 'react';
const { useState, useRef } = React;
import { Btn, Empty, FeatureTag, HelpNote, PageHead, Panel } from '../../components/components.jsx';
import { KNOWLEDGE, byId } from '../../data/data.jsx';
import { KBODY, KTYPE, KTYPE_EN, KTYPE_TH, KTYPE_OPTS, loadCodex, CodexDrawer } from './codex.jsx';

/* ---------------- RECALL — hybrid retrieval + cited Q&A ---------------- */
/* semantic concept map: lets near-meaning queries still find the right docs */
const RECALL_CONCEPTS = [
  ["security", ["token", "auth", "login", "เข้าสู่ระบบ", "ความปลอดภัย", "รั่ว", "refresh", "ปลอดภัย", "credential"]],
  ["rag",      ["retrieval", "ค้นหา", "ค้นคืน", "vector", "embedding", "hybrid", "rerank", "semantic", "ความหมาย"]],
  ["test",     ["test", "ทดสอบ", "qa", "คุณภาพ", "bug", "บั๊ก", "regression", "edge case"]],
  ["onboard",  ["onboarding", "เริ่มต้น", "เอเจนต์ใหม่", "สมาชิกใหม่", "เข้าร่วม", "มือใหม่"]],
  ["deps",     ["dependency", "อัปเดต", "ช่องโหว่", "เวอร์ชัน", "package", "ล้าสมัย", "vulnerab"]],
];
function recallDocText(d) { return (d.title + " " + (KBODY[d.id] || d.body || "") + " " + (d.tags || []).join(" ")); }
function recallScore(doc, query) {
  const q = query.toLowerCase().trim();
  const title = (doc.title || "").toLowerCase();
  const text = recallDocText(doc).toLowerCase();
  let score = 0, hits = 0;
  if (q && title.includes(q)) score += 5;
  else if (q && text.includes(q)) score += 3;
  const words = q.split(/[\s,?.!“”"’'()/]+/).filter(w => w.length >= 2);
  for (const w of words) {
    if (title.includes(w)) { score += 2; hits++; }
    else if (text.includes(w)) { score += 1; hits++; }
  }
  for (const t of (doc.tags || [])) if (q.includes(t.toLowerCase())) score += 1.5;
  for (const [, terms] of RECALL_CONCEPTS) {
    const qHit = terms.some(t => q.includes(t.toLowerCase()));
    const dHit = terms.some(t => text.includes(t.toLowerCase()));
    if (qHit && dHit) { score += 1.4; hits++; }
  }
  return { score, hits, words };
}
function recallSnippet(doc, words) {
  const body = KBODY[doc.id] || doc.body || doc.title || "";
  const lower = body.toLowerCase();
  let idx = -1;
  for (const w of words) { const i = lower.indexOf(w); if (i !== -1 && (idx === -1 || i < idx)) idx = i; }
  const start = idx > 50 ? idx - 40 : 0;
  let snip = body.slice(start, start + 170);
  if (start > 0) snip = "…" + snip;
  if (start + 170 < body.length) snip = snip + "…";
  return snip;
}
function recallHighlight(text, words) {
  const terms = words.filter(w => w.length >= 2);
  if (!terms.length) return text;
  const esc = terms.map(t => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp("(" + esc.join("|") + ")", "gi");
  const lowerTerms = terms.map(t => t.toLowerCase());
  return text.split(re).map((p, i) =>
    lowerTerms.includes(p.toLowerCase())
      ? <mark key={i} className="hl">{p}</mark>
      : <React.Fragment key={i}>{p}</React.Fragment>);
}

/* mock GET /recall?q=&type= — hybrid retrieval → ranked documents (§6.2 contract) */
function recallSearch(all, query, typeFilter) {
  const ranked = all
    .filter(d => !typeFilter || typeFilter === "all" || d.type === typeFilter)
    .map(d => { const s = recallScore(d, query); return { doc: d, ...s }; })
    .filter(r => r.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 6);
  const top = ranked[0] ? ranked[0].score : 1;
  ranked.forEach(r => { r.sim = Math.min(0.97, 0.5 + 0.45 * (r.score / top)); r.matched = Math.max(1, r.hits); });
  return ranked;
}

/* mock POST /ask — answer grounded in retrieved docs, cited inline [n] (§6.3 contract).
   Uses the real model when available, else a deterministic context-grounded reply. */
async function askHermes(query, ranked, lang) {
  const T = (en, th) => lang === "en" ? en : th;
  const context = ranked.map((r, i) => `[${i + 1}] ${r.doc.title}: ${KBODY[r.doc.id] || r.doc.body || ""}`).join("\n");
  try {
    if (ranked.length && window.claude && window.claude.complete) {
      const langLine = lang === "en"
        ? "Answer in English, concise (max 4 sentences). Cite sources inline like [1], [2] referring to the numbered context."
        : "ตอบเป็นภาษาไทยกระชับ ไม่เกิน 4 ประโยค อ้างอิงแหล่งแบบ [1], [2] ตามหมายเลขบริบทที่ให้";
      const sys = "You are Orchestrator, the guild librarian. Only use the provided context. If nothing matches, say so. " + langLine + "\n\nContext:\n" + context;
      const reply = await window.claude.complete(sys + "\n\nQuestion: " + query);
      if (reply && reply.trim()) return reply.trim();
    }
  } catch (e) { /* fall through to deterministic reply */ }
  if (!ranked.length) return T("I couldn't find anything matching in the codex. Try different words, or add a note first.", "ไม่พบเอกสารที่ตรงในคลังความรู้ ลองใช้คำอื่น หรือเพิ่มบันทึกก่อน");
  const b0 = KBODY[ranked[0].doc.id] || ranked[0].doc.body || "";
  const b1 = ranked[1] ? (KBODY[ranked[1].doc.id] || ranked[1].doc.body || "") : "";
  return T(
    `Based on the codex, the most relevant source is “${ranked[0].doc.title}” [1]. ${b0}${b1 ? ` This aligns with “${ranked[1].doc.title}” [2].` : ""}`,
    `จากคลังความรู้ แหล่งที่เกี่ยวข้องที่สุดคือ “${ranked[0].doc.title}” [1] · ${b0}${b1 ? ` และสอดคล้องกับ “${ranked[1].doc.title}” [2]` : ""}`
  );
}

/* answer text with clickable inline [n] citations that jump back to the source doc */
function AnswerBody({ text, results, onCite, streaming }) {
  const parts = (text || "").split(/(\[\d+\])/g);
  return (
    <span>
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (m) {
          const r = results && results[+m[1] - 1];
          if (r) return <button key={i} className="cite-inline" onClick={() => onCite(r.doc)} title={r.doc.title}>{m[1]}</button>;
        }
        return <React.Fragment key={i}>{p}</React.Fragment>;
      })}
      {streaming && <span className="stream-caret" />}
    </span>
  );
}

function RecallResult({ rank, r, T, onOpen }) {
  const { doc, sim, matched, words } = r;
  const by = byId(doc.by);
  const tone = sim >= 0.85 ? "hi" : sim >= 0.7 ? "mid" : "lo";
  return (
    <button className="recall-result" onClick={() => onOpen(doc)} data-no-lex>
      <span className="rr-rank mono">{rank}</span>
      <span className="codex-type" style={{ flex: "none" }}>{KTYPE[doc.type] || "📝"}</span>
      <div className="rr-main">
        <div className="rr-top">
          <span className="rr-title">{doc.title}</span>
          <span className="rr-scorewrap" title={T("relevance score", "คะแนนความเกี่ยวข้อง")}>
            <span className={`rr-scorebar t-${tone}`}><i style={{ width: Math.round(sim * 100) + "%" }} /></span>
            <span className="rr-score mono">{sim.toFixed(2)}</span>
          </span>
        </div>
        <div className="rr-snippet">{recallHighlight(recallSnippet(doc, words), words)}</div>
        <div className="rr-meta mono">
          <span>{T(KTYPE_EN[doc.type] || "Note", KTYPE_TH[doc.type] || "บันทึก")}</span>
          <span>·</span><span>{by ? by.name.split(" ")[0] : T("Guild", "องค์กร")}</span>
          <span>·</span><span>{matched} {T("matched chunks", "ส่วนที่ตรง")}</span>
          <span>·</span><span>{doc.refs ?? 0} {T("refs", "อ้างอิง")}</span>
        </div>
      </div>
      <span className="rr-open mono">{T("open →", "เปิด →")}</span>
    </button>
  );
}

function Recall({ lang }) {
  const T = (en, th) => lang === "en" ? en : th;
  const [q, setQ] = useState("");
  const [results, setResults] = useState(null);   // null | [] | [{doc,sim,...}]
  const [answer, setAnswer] = useState("");       // progressively built while streaming
  const [phase, setPhase] = useState("idle");     // idle | retrieving | streaming | done
  const [docType, setDocType] = useState("all");
  const [sel, setSel] = useState(null);
  const runRef = useRef(0);
  const extra = (typeof loadCodex === "function") ? loadCodex() : [];
  const all = [...extra, ...KNOWLEDGE];
  const busy = phase === "retrieving" || phase === "streaming";

  const ask = async (text, typeOverride) => {
    const query = (text ?? q).trim();
    if (!query || busy) return;
    const type = typeOverride ?? docType;
    const run = ++runRef.current;
    setQ(query); setAnswer(""); setResults(null); setPhase("retrieving");

    // ---- GET /recall?q=&type= : hybrid retrieval (keyword + meaning), ranked ----
    await new Promise(r => setTimeout(r, 460));            // simulate retrieval latency
    if (run !== runRef.current) return;
    const ranked = recallSearch(all, query, type);
    setResults(ranked);

    // ---- POST /ask : answer grounded in retrieved docs ----
    const full = await askHermes(query, ranked, lang);
    if (run !== runRef.current) return;

    // ---- stream the answer token-by-token ----
    setPhase("streaming");
    const toks = full.split(/(\s+)/);
    let acc = "";
    for (let i = 0; i < toks.length; i++) {
      if (run !== runRef.current) return;
      acc += toks[i];
      setAnswer(acc);
      if (toks[i].trim()) await new Promise(r => setTimeout(r, 20));
    }
    if (run !== runRef.current) return;
    setPhase("done");
  };

  const suggestions = [
    T("How did we decide on refresh tokens?", "เราตัดสินใจเรื่อง refresh token อย่างไร?"),
    T("Which retrieval approach did we pick?", "เราเลือก retrieval แบบไหน?"),
    T("What are our testing standards?", "มาตรฐานการเขียน test ของเรา"),
  ];
  const typeTabs = [["all", T("all types", "ทุกประเภท")], ...KTYPE_OPTS.map(t => [t, T(KTYPE_EN[t], KTYPE_TH[t])])];

  return (
    <div className="content-pad fade-in">
      <PageHead kicker={T("Knowledge · Recall", "ความรู้ · Recall")} title={T("Recall", "ค้นหาความรู้")} tag="live"
        desc={T("Ask the guild's knowledge base in plain language — Orchestrator retrieves the most relevant documents and answers with citations.",
                "ถามคลังความรู้ด้วยภาษาธรรมดา — ผู้ควบคุมกลาง จะค้นเอกสารที่เกี่ยวข้องที่สุดแล้วตอบพร้อมอ้างอิง")} />
      <HelpNote tag="live">{T("It runs a hybrid search over your Codex notes (keyword + meaning), then synthesizes an answer. Click any [number] in the answer to jump to its source document.",
        "ระบบค้นแบบ hybrid จากบันทึกในหน้า Codex (คำสำคัญ + ความหมาย) แล้วสรุปคำตอบ · กดเลข [n] ในคำตอบเพื่อไปยังเอกสารต้นทาง")}</HelpNote>

      <div className="search-bar" style={{ margin: "16px 0 12px", padding: "14px 18px" }}>
        <span style={{ fontSize: 16 }}>🔮</span>
        <input value={q} onChange={e => setQ(e.target.value)} onKeyDown={e => e.key === "Enter" && ask()}
          placeholder={T("Ask anything about the guild's knowledge…", "ถามอะไรก็ได้เกี่ยวกับคลังความรู้…")} />
        <Btn kind="gold" sm onClick={() => ask()}>{T("Search", "สืบค้น")}</Btn>
      </div>

      <div className="recall-filters" style={{ marginBottom: 14 }}>
        <span className="mono faint" style={{ fontSize: 11 }}>{T("filter", "กรอง")}</span>
        {typeTabs.map(([k, label]) => (
          <button key={k} className={`tab-pill ${docType === k ? "on" : ""}`}
            onClick={() => { setDocType(k); if (results || busy) ask(q, k); }}>{label}</button>
        ))}
      </div>

      {!results && !busy && (
        <div className="grid cols-3" style={{ marginBottom: 18 }}>
          {suggestions.map((s, i) => (
            <button key={i} className="codex-row" style={{ justifyContent: "flex-start" }} onClick={() => ask(s)}>
              <span className="codex-type">💡</span>
              <div className="codex-main"><div className="codex-title" style={{ fontSize: 13 }}>{s}</div><div className="codex-meta">{T("suggested question", "คำถามแนะนำ")}</div></div>
            </button>
          ))}
        </div>
      )}

      <div className="grid" style={{ gridTemplateColumns: "1fr 360px", gap: 16, alignItems: "start" }}>
        {/* ---- answer + citations ---- */}
        <Panel title={T("Orchestrator answer", "คำตอบจาก ผู้ควบคุมกลาง")} en="SYNTHESIS" icon="⚜" right={<FeatureTag kind="live" />}>
          {phase === "retrieving" ? (
            <div className="row" style={{ gap: 10, color: "var(--ink-2)" }}>
              <span className="typing-bubble" style={{ display: "inline-flex" }}><span /><span /><span /></span>
              {T("Orchestrator is retrieving and reading the codex…", "ผู้ควบคุมกลาง กำลังค้นและอ่านคลังความรู้…")}
            </div>
          ) : (phase === "streaming" || phase === "done") ? (
            <div style={{ fontSize: 14, lineHeight: 1.8, color: "var(--ink)" }}>
              <div className="row" style={{ gap: 9, marginBottom: 10 }}><span className="wchat-crest">⚜</span><span className="mono gold-text" style={{ fontSize: 12 }}>ผู้ควบคุมกลาง</span></div>
              <AnswerBody text={answer} results={results} onCite={setSel} streaming={phase === "streaming"} />
              {phase === "done" && results && results.length > 0 && (
                <div className="citations">
                  <div className="cite-label mono">{T("Sources", "แหล่งอ้างอิง")}</div>
                  <div className="cite-chips">
                    {results.map((r, i) => (
                      <button key={r.doc.id} className="cite-chip" onClick={() => setSel(r.doc)} title={r.doc.title}>
                        <span className="cite-num">{i + 1}</span>
                        <span className="cite-title">{r.doc.title}</span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <Empty icon="🔮" title={T("No query yet", "ยังไม่มีการสืบค้น")} sub={T("Ask a question above to search the codex", "พิมพ์คำถามด้านบนเพื่อค้นคลังความรู้")} />
          )}
        </Panel>

        {/* ---- retrieved documents ---- */}
        <Panel title={T("Documents found", "เอกสารที่เจอ")} en="RETRIEVAL" icon="🔍" bodyPad={false}
          right={results ? <span className="mono faint" style={{ fontSize: 11 }}>{results.length}</span> : null}>
          <div style={{ padding: 8 }}>
            {!results && busy ? (
              <div className="muted" style={{ fontSize: 13, padding: "10px 8px" }}>{T("ranking documents…", "กำลังจัดอันดับเอกสาร…")}</div>
            ) : !results ? (
              <div className="muted" style={{ fontSize: 12.5, padding: "10px 8px", lineHeight: 1.6 }}>{T("Results appear here, ranked by relevance — keyword and meaning combined.", "ผลลัพธ์จะแสดงที่นี่ เรียงตามความเกี่ยวข้อง (คำสำคัญ + ความหมาย)")}</div>
            ) : results.length === 0 ? (
              <Empty icon="🔍" title={T("No strong matches", "ไม่พบที่ตรงพอ")} sub={T("Try different words, or add a note to the Codex", "ลองคำอื่น หรือเพิ่มบันทึกในคลัง")} />
            ) : (
              <div className="col" style={{ gap: 8 }}>
                {results.map((r, i) => <RecallResult key={r.doc.id} rank={i + 1} r={r} T={T} onOpen={setSel} />)}
              </div>
            )}
          </div>
        </Panel>
      </div>

      {sel && <CodexDrawer k={sel} onClose={() => setSel(null)} />}
    </div>
  );
}

export { RECALL_CONCEPTS, recallDocText, recallScore, recallSnippet, recallHighlight, recallSearch, askHermes, AnswerBody, RecallResult, Recall };
