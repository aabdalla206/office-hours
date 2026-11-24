// Export a Canvas course to course_export.json using your existing Canvas login (no access token).
// Use when your school does not allow personal access tokens.
//
// How: open any page of the course in Canvas, open the browser console (Cmd+Option+J / Ctrl+Shift+J),
// paste this whole file, press Enter. A file named course_export.json downloads.
// Then: python ingest.py --from-export course_export.json
//
// It only reads what your account can already see, and only runs in your own browser.
(async () => {
  const m = location.pathname.match(/\/courses\/(\d+)/);
  if (!m) { alert("Open a page inside the course first."); return; }
  const id = m[1];
  const api = async (path) => {
    let url = `/api/v1/courses/${id}${path}${path.includes("?") ? "&" : "?"}per_page=100`;
    const out = [];
    while (url) {
      const r = await fetch(url, { headers: { Accept: "application/json" } });
      if (!r.ok) throw new Error(`${r.status} on ${url}`);
      const data = JSON.parse((await r.text()).replace(/^while\(1\);/, ""));
      Array.isArray(data) ? out.push(...data) : out.push(data);
      const next = (r.headers.get("Link") || "").split(",").find((l) => l.includes('rel="next"'));
      url = next ? next.slice(next.indexOf("<") + 1, next.indexOf(">")) : null;
    }
    return out;
  };

  const [course] = await api("?include[]=syllabus_body");
  // Many courses hide the Pages list from students (404), so reach pages through module items.
  const mods = await api("/modules?include[]=items");
  const pages = [], seen = new Set();
  for (const mod of mods) for (const it of mod.items || []) {
    if (it.type !== "Page" || !it.page_url || seen.has(it.page_url)) continue;
    seen.add(it.page_url);
    try {
      const [full] = await api(`/pages/${encodeURIComponent(it.page_url)}`);
      if (full.body) pages.push({ title: full.title, module: mod.name, html: full.body });
    } catch (e) { console.warn("skipped", it.title, e.message); }
  }
  const modules = mods.map((mod) => ({
    name: mod.name,
    items: (mod.items || []).map((i) => ({ type: i.type, title: i.title })),
  }));

  const blob = new Blob(
    [JSON.stringify({ course: course.name, exported_at: new Date().toISOString(),
                      syllabus_html: course.syllabus_body || "", pages, modules })],
    { type: "application/json" }
  );
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob), download: "course_export.json",
  });
  document.body.appendChild(a); a.click(); a.remove();
  console.log(`Exported syllabus, ${pages.length} pages, ${modules.length} modules.`);
})();
