/* Knowledge / RAG — the frontend half of the `knowledge` plugin (plugin-architecture.md §12, Phase 6).

   The descriptor is the frontend mirror of the backend manifest: it tells the Core shell which routes
   this feature owns, how to render them, their topbar metadata, and its sidebar entries — so Core never
   hardcodes Codex/Recall into App.jsx or data.jsx. Each `render(ctx)` is handed only the Core seams the
   screen needs (t · can · language); the plugin owns the prop wiring.

   The screens now physically live in this folder (codex.jsx · recall.jsx), relocated out of
   screens/extra/ in Phase 6b. The dead Codex/Recall imports the Base screens (MyDashboard, RBAC)
   used to carry were dropped at the same time, so nothing in Core reaches into this plugin. A
   per-plugin i18n pack + lazy code-split remain as later refinements. */
import React from 'react';

import { Codex } from './codex.jsx';
import { Recall } from './recall.jsx';

export default {
  id: 'knowledge',
  routes: [
    {
      id: 'codex',
      meta: { icon: '📚', title: 'บันทึกความรู้', en: 'Codex' },
      render: (ctx) => <Codex t={ctx.t} can={ctx.can} />,
    },
    {
      id: 'search',
      meta: { icon: '🔍', title: 'ค้นหาความรู้', en: 'Recall' },
      render: (ctx) => <Recall lang={ctx.language} />,
    },
  ],
  // sidebar entries (i18n label resolves from `nav.<id>`, same as Base items — §11 config-driven).
  nav: [
    {
      group: 'ความรู้และความทรงจำ',
      items: [
        { id: 'codex', icon: '📚' },
        { id: 'search', icon: '🔍' },
      ],
    },
  ],
  // RBAC permissions this plugin OWNS — contributed to the catalog dynamically (plugin-architecture §0,
  // "permissions are manifest/descriptor-contributed"). Install ⇒ they appear in the RBAC screen + admin
  // gets them; remove ⇒ they disappear. Keys match what the backend router enforces (require_perm
  // ("codex.*")) so one name gates both the API and the UI.
  permissions: [
    { key: 'knowledge.view',   group: 'Knowledge', th: 'ดู/ค้นหาคลังความรู้',            en: 'View & search codex' },
    { key: 'knowledge.manage', group: 'Knowledge', th: 'อัปโหลด/จัดการเนื้อหาคลังความรู้', en: 'Upload & manage codex content' },
    { key: 'knowledge.delete', group: 'Knowledge', th: 'ลบเอกสารในคลังความรู้',          en: 'Delete codex documents' },
  ],
};
