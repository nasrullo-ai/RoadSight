// Shared header/footer, theme toggle and the small SVG/HTML charts used across pages.
(function () {
  const PAGES = [
    ["index.html", "Home"], ["approach.html", "Approach"], ["eda.html", "Data"], ["results.html", "Results"],
    ["demo.html", "Live demo"], ["report.html", "Report"], ["team.html", "Team"], ["links.html", "Links"],
  ];
  const here = location.pathname.split("/").pop() || "index.html";

  function header() {
    const el = document.getElementById("site-header");
    if (!el) return;
    const links = PAGES.map(([href, name]) =>
      `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${name}</a>`).join("");
    el.className = "site-header";
    el.innerHTML = `<div class="wrap">
      <a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">
        <svg viewBox="0 0 16 16"><path d="M8 1v14" stroke="#f2c200" stroke-width="2" stroke-dasharray="3 2"/></svg></span>RoadSight</a>
      <nav class="main" aria-label="Pages">${links}</nav>
      <button class="theme-toggle" type="button" id="theme-toggle">Theme</button></div>`;
    document.getElementById("theme-toggle").addEventListener("click", toggleTheme);
  }

  function footer() {
    const el = document.getElementById("site-footer");
    if (!el) return;
    el.className = "site-footer";
    el.innerHTML = `<div class="wrap">RoadSight, a traffic event detection entry. Runs offline on one GPU.
      Detector: Ultralytics YOLO11 (AGPL-3.0); tracker: ByteTrack. <a href="links.html">Code, weights and predictions</a>.</div>`;
  }

  function currentTheme() {
    const set = document.documentElement.dataset.theme;
    if (set) return set;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function toggleTheme() {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("rs-theme", next); } catch (e) { /* storage may be blocked */ }
  }
  try { const t = localStorage.getItem("rs-theme"); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pretty = (label) => label.replace(/_/g, " ");
  const fmt = (t) => { const m = Math.floor(t / 60); const s = (t - 60 * m).toFixed(1).padStart(4, "0"); return `${m}:${s}`; };

  async function getJSON(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url}: ${r.status}`);
    return r.json();
  }

  // Timeline: one lane per class; predicted segments on top, ground truth as a thin bar below.
  function timeline(el, { events = [], gt = [], duration = 1, onSeek = null }) {
    const labels = [...new Set([...events, ...gt].map((e) => e[2]))].sort();
    if (!labels.length) {
      el.innerHTML = `<p class="muted ui">No events detected in this video.</p>`;
      return { setTime() {} };
    }
    const pct = (t) => `${Math.max(0, Math.min(100, (100 * t) / duration))}%`;
    const rows = labels.map((lab) => {
      const pred = events.filter((e) => e[2] === lab).map((e) =>
        `<button class="seg" style="left:${pct(e[0])};width:calc(${pct(e[1] - e[0])})" data-t="${e[0]}"
          title="${esc(pretty(lab))}: ${fmt(e[0])} to ${fmt(e[1])}" aria-label="Seek to ${esc(pretty(lab))} at ${fmt(e[0])}"></button>`).join("");
      const g = gt.filter((e) => e[2] === lab).map((e) =>
        `<span class="seg gt" style="left:${pct(e[0])};width:calc(${pct(e[1] - e[0])})"></span>`).join("");
      return `<div class="lane-row"><div class="lane-name">${esc(pretty(lab))}</div><div class="track">${g}${pred}<span class="playhead" style="left:0"></span></div></div>`;
    }).join("");
    el.innerHTML = `<div class="timeline">${rows}<div class="axis"><div></div><div><span>0:00.0</span><span>${fmt(duration)}</span></div></div></div>`;
    if (onSeek) el.querySelectorAll("button.seg").forEach((b) => b.addEventListener("click", () => onSeek(parseFloat(b.dataset.t))));
    const heads = el.querySelectorAll(".playhead");
    return { setTime(t) { heads.forEach((h) => { h.style.left = pct(t); }); } };
  }

  // Risk curve as an SVG polyline with the 0.5 alarm threshold.
  function riskChart(el, risk, { duration = null, onSeek = null } = {}) {
    if (!risk || !risk.length) { el.innerHTML = `<p class="muted ui">No risk scores.</p>`; return { setTime() {} }; }
    const W = 1000, H = 150, P = 22;
    const T = duration || risk[risk.length - 1][0] || 1;
    const step = Math.max(1, Math.floor(risk.length / 1500));
    const pts = [];
    for (let i = 0; i < risk.length; i += step) pts.push([P + ((W - 2 * P) * risk[i][0]) / T, H - P - (H - 2 * P) * risk[i][1]]);
    const line = pts.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ");
    const area = `${P},${H - P} ${line} ${pts[pts.length - 1][0].toFixed(1)},${H - P}`;
    const yThr = H - P - (H - 2 * P) * 0.5;
    el.innerHTML = `<svg class="risk-chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Accident risk over time">
      <polygon class="area" points="${area}"/><polyline class="curve" points="${line}"/>
      <line class="thr" x1="${P}" x2="${W - P}" y1="${yThr}" y2="${yThr}"/>
      <line class="head" x1="${P}" x2="${P}" y1="${P / 2}" y2="${H - P}" stroke="currentColor" stroke-width="1.5"/></svg>
      <div class="legend"><span><i style="background:var(--red)"></i>risk of an accident starting within 5 s</span>
      <span>dashed line: alarm threshold 0.5</span><span>click to jump</span></div>`;
    const svg = el.querySelector("svg");
    const head = el.querySelector("line.head");
    if (onSeek) svg.addEventListener("click", (ev) => {
      const r = svg.getBoundingClientRect();
      onSeek(Math.max(0, ((ev.clientX - r.left) / r.width * W - P) / (W - 2 * P) * T));
    });
    return { setTime(t) { const x = P + ((W - 2 * P) * t) / T; head.setAttribute("x1", x); head.setAttribute("x2", x); } };
  }

  function eventsTable(el, events, onSeek) {
    if (!events.length) { el.innerHTML = `<p class="muted ui">No events.</p>`; return; }
    el.innerHTML = `<div class="table-scroll"><table><thead><tr><th>Event</th><th class="num">Start</th><th class="num">End</th><th class="num">Length</th></tr></thead><tbody>
      ${events.map((e) => `<tr><td><a href="#" data-t="${e[0]}">${esc(pretty(e[2]))}</a></td><td class="num">${fmt(e[0])}</td>
        <td class="num">${fmt(e[1])}</td><td class="num">${(e[1] - e[0]).toFixed(1)} s</td></tr>`).join("")}</tbody></table></div>`;
    el.querySelectorAll("a[data-t]").forEach((a) => a.addEventListener("click", (ev) => { ev.preventDefault(); onSeek && onSeek(parseFloat(a.dataset.t)); }));
  }

  // Small multi-series line chart for counts over time.
  function lineChart(el, series, { xLabel = "seconds", yLabel = "" } = {}) {
    const names = Object.keys(series);
    if (!names.length) { el.innerHTML = `<p class="muted ui">No data.</p>`; return; }
    const W = 1000, H = 220, P = 30;
    const xs = names.flatMap((n) => series[n].map((p) => p[0]));
    const ys = names.flatMap((n) => series[n].map((p) => p[1]));
    const xMax = Math.max(1, ...xs), yMax = Math.max(1, ...ys);
    const colors = ["var(--sign)", "var(--pred)", "var(--amber)", "var(--red)", "var(--muted)", "var(--go)"];
    const lines = names.map((n, i) => `<polyline fill="none" stroke="${colors[i % colors.length]}" stroke-width="2" points="${series[n]
      .map((p) => `${(P + ((W - 2 * P) * p[0]) / xMax).toFixed(1)},${(H - P - ((H - 2 * P) * p[1]) / yMax).toFixed(1)}`).join(" ")}"/>`).join("");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="risk-chart" style="height:220px" role="img" aria-label="${esc(yLabel)} over time">
      ${lines}<text x="${P}" y="${P - 10}">${esc(yLabel)} (max ${yMax})</text><text x="${W - P - 60}" y="${H - 6}">${esc(xLabel)}</text></svg>
      <div class="legend">${names.map((n, i) => `<span><i style="background:${colors[i % colors.length]}"></i>${esc(n)}</span>`).join("")}</div>`;
  }

  window.RS = { getJSON, timeline, riskChart, eventsTable, lineChart, esc, pretty, fmt };
  document.addEventListener("DOMContentLoaded", () => { header(); footer(); });
})();
