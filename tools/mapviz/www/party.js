// Party page: a snapshot of every env's party (/api/party/snapshot) and an on-demand behaviour sample
// (/api/party/sample), both read by tools/mapviz/party_probe.py from the trainer's shared memory. Filtering and
// aggregation happen here.
"use strict";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pretty = (s) => String(s).toLowerCase().split("_").map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w)).join(" ");
const pct = (a, b) => (b > 0 ? `${Math.round((100 * a) / b)}%` : "–");
const pct1 = (a, b) => (b > 0 ? `${((100 * a) / b).toFixed(1)}%` : "–");
const fmt = (x) => (x >= 1e4 ? `${(x / 1e3).toFixed(0)}k` : x >= 1e3 ? `${(x / 1e3).toFixed(1)}k` : String(Math.round(x)));
function quant(a, q) {
  if (!a.length) return NaN;
  const s = [...a].sort((x, y) => x - y);
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
const med = (a) => quant(a, 0.5);
const ICON_COLS = 32, SPECIES_EGG = 412;  // world/icons.png: 32 x 32 px per species, ICON_COLS per row (build_world.py build_icons)
const ico = (sp) => `<span class="ico" style="background-position:${-(sp % ICON_COLS) * 32}px ${-Math.floor(sp / ICON_COLS) * 32}px"></span>`;
const STORE = "fireredsim-party";
let SNAP = null, SAMPLE = null, showAll = false, openSp = new Set(), autoTimer = null;
const sel = { badges: new Set(), scope: "all", where: "all", hm: "all", q: "" };
try {
  const s = JSON.parse(localStorage.getItem(STORE) || "{}");
  if (typeof s.scope === "string") sel.scope = s.scope;
  if (typeof s.where === "string") sel.where = s.where;
  if (typeof s.hm === "string") sel.hm = s.hm;
  if (Array.isArray(s.badges)) sel.badges = new Set(s.badges);
} catch (e) { /* defaults */ }
function save() {
  try { localStorage.setItem(STORE, JSON.stringify({ badges: [...sel.badges], scope: sel.scope, where: sel.where, hm: sel.hm })); } catch (e) { /* ignore */ }
}
function setState(text, cls) { const s = $("state"); s.textContent = text; s.className = cls || "dim"; }
function showErr(t) { $("err").innerHTML = t ? `<div class="msg err">${esc(t)}</div>` : ""; }
async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  return r.json();
}

// ---- snapshot ------------------------------------------------------------------------------------------------
async function refresh(fresh) {
  setState("reading…", "busy");
  try {
    const d = await getJSON("/api/party/snapshot" + (fresh ? "?fresh=1" : ""));
    if (d.error) { showErr(d.error); setState("no training", "down"); return; }
    showErr("");
    SNAP = d;
    $("run").textContent = `${d.run} · ${d.n} envs`;
    setState(`snapshot ${new Date(d.t * 1000).toLocaleTimeString()}`, "live");
    buildFilters();
    render();
  } catch (e) { showErr(String(e)); setState("server down", "down"); }
}
function envObj(r) {
  if (r[1] < 0) return null;
  const [e, b, bits, map, scope, battle, money, boxed, party, hm, bag, stats] = r;
  const h = hm || [0, 0, 0];
  return { e, b, bits, map, scope, battle, money, boxed, party, hm: h, gap: h[0] & ~h[1], bag: bag || [], stats };
}
function buildFilters() {
  const N = SNAP.names;
  const have = new Set(SNAP.envs.filter((r) => r[1] >= 0).map((r) => r[1]));
  $("badges").innerHTML = [`<button data-b="all" aria-pressed="${sel.badges.size === 0}">all</button>`]
    .concat([...Array(9).keys()].map((b) => `<button data-b="${b}" aria-pressed="${sel.badges.has(b)}"
      ${have.has(b) ? "" : 'class="muted"'} title="${have.has(b) ? "" : "no env has this many now"}">${b}</button>`)).join("");
  const sc = $("scope");
  const opts = [["all", "any"], ["none", "none"]].concat(N.scopes.map((s, i) => [String(i), s]));
  sc.innerHTML = opts.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
  sc.value = opts.some((o) => o[0] === sel.scope) ? sel.scope : "all";
  $("where").value = sel.where;
  $("hm").value = sel.hm;
}
function hmOk(gap) { return sel.hm === "all" || (sel.hm === "gap") === !!gap; }
function match(o, useWhere = true) {
  if (!o) return false;
  if (sel.badges.size && !sel.badges.has(o.b)) return false;
  if (sel.scope === "none" ? o.scope !== -1 : sel.scope !== "all" && o.scope !== +sel.scope) return false;
  if (!hmOk(o.gap)) return false;
  if (useWhere) {
    if (sel.where === "field" && o.battle) return false;
    if (sel.where === "battle" && !o.battle) return false;
    if (sel.where === "trainer" && o.battle !== 2) return false;
  }
  if (sel.q) {
    const q = sel.q.toUpperCase().replace(/\s+/g, "_");
    if (!o.party.some((m) => SNAP.names.species[m[0]].includes(q))) return false;
  }
  return true;
}
function mons(o) { return o.party.filter((m) => m[0] && !m[6]); }
function tile(t, v, s) { return `<div class="tile"><div class="t">${t}</div><div class="v">${v}</div><div class="s">${s || ""}</div></div>`; }

function render() {
  if (!SNAP) return;
  const N = SNAP.names;
  const all = SNAP.envs.map(envObj);
  const E = all.filter((o) => match(o));
  const ing = all.filter(Boolean).length;
  // tiles
  const best = E.map((o) => Math.max(0, ...mons(o).map((m) => m[1])));
  const size = E.map((o) => mons(o).length);
  const weakLead = E.filter((o) => { const m = mons(o); return m.length > 1 && o.party[0] && m.length && o.party[0][1] < Math.max(...m.map((x) => x[1])); }).length;
  const inb = E.filter((o) => o.battle).length;
  $("tiles").innerHTML = [
    tile("Envs shown", `${E.length}`, `of ${ing} in game`),
    tile("Best level", isNaN(med(best)) ? "–" : med(best).toFixed(0), `median · q90 ${isNaN(quant(best, 0.9)) ? "–" : quant(best, 0.9).toFixed(0)}`),
    tile("Party size", E.length ? (size.reduce((a, b) => a + b, 0) / E.length).toFixed(1) : "–", "mean mons (no eggs)"),
    tile("Lead not the strongest", pct(weakLead, E.length), "slot 1 below the best level"),
    tile("In battle now", pct(inb, E.length), `${E.filter((o) => o.battle === 2).length} trainer`),
    tile("Mons in the PC", E.length ? med(E.map((o) => o.boxed)).toFixed(0) : "–", `median · ${pct(E.filter((o) => o.boxed > 0).length, E.length)} have any`),
    tile("Money", E.length ? `₽${fmt(med(E.map((o) => o.money)))}` : "–", "median"),
    tile("HM in bag, not taught", pct(E.filter((o) => o.gap).length, E.length),
      `${pct(E.filter((o) => o.gap & o.hm[2]).length, E.length)} have a mon that can learn it`),
  ].concat(E.length && E[0].stats ? [
    tile("Ever shopped", pct(E.filter((o) => o.stats[0] > 0).length, E.length), "game counter, whole save lineage"),
    tile("Used a nurse", pct(E.filter((o) => o.stats[1] > 0).length, E.length), "Pokémon Center heals, lineage"),
  ] : []).join("");
  renderBag(E);
  // levels by rank
  const ranks = [[], [], [], [], [], []];
  for (const o of E) mons(o).map((m) => m[1]).sort((a, b) => b - a).forEach((l, i) => ranks[i].push(l));
  $("ranks").innerHTML = `<tr><th>Rank</th><th class="n">Envs</th><th class="n">Median</th><th class="n">q10–q90</th><th>Spread</th></tr>` +
    ranks.map((a, i) => {
      if (!a.length) return `<tr><td>#${i + 1}</td><td class="n">0</td><td class="n">–</td><td class="n">–</td><td></td></tr>`;
      const lo = quant(a, 0.1), hi = quant(a, 0.9), m = med(a);
      return `<tr><td>#${i + 1}</td><td class="n">${a.length}</td><td class="n">${m.toFixed(0)}</td><td class="n">${lo.toFixed(0)}–${hi.toFixed(0)}</td>
        <td><span class="bar" style="margin-left:${lo}px;width:${Math.max(2, hi - lo)}px" title="q10 ${lo.toFixed(0)} – q90 ${hi.toFixed(0)}"></span></td></tr>`;
    }).join("");
  // leads
  const lead = new Map();
  for (const o of E) {
    const m = o.party[0];
    if (!m || !m[0]) continue;
    const ms = mons(o), rank = 1 + ms.filter((x) => x[1] > m[1]).length;
    const L = lead.get(m[0]) || { n: 0, lv: [], top: 0 };
    L.n++; L.lv.push(m[1]); L.top += rank === 1;
    lead.set(m[0], L);
  }
  const leads = [...lead.entries()].sort((a, b) => b[1].n - a[1].n).slice(0, 10);
  $("leads").innerHTML = `<tr><th>Lead</th><th class="n">Envs</th><th class="n">Level</th><th class="n">Is the best</th></tr>` +
    (leads.length ? leads.map(([sp, L]) => `<tr><td class="name">${ico(sp)}${esc(pretty(N.species[sp]))}</td><td class="n">${pct(L.n, E.length)}</td>
      <td class="n">${med(L.lv).toFixed(0)}</td><td class="n">${pct(L.top, L.n)}</td></tr>`).join("") : `<tr><td colspan="4" class="dim">no env</td></tr>`);
  renderSpecies(E);
  renderEnvs(E);
  renderSample();
}

function renderSpecies(E) {
  const N = SNAP.names;
  const sp = new Map();
  for (const o of E) {
    const seen = new Set();
    for (const m of mons(o)) {
      const S = sp.get(m[0]) || { envs: 0, mons: 0, lv: [], moves: new Map(), hp: 0 };
      if (!seen.has(m[0])) { S.envs++; seen.add(m[0]); }
      S.mons++; S.lv.push(m[1]); S.hp += m[3] ? m[2] / m[3] : 0;
      for (const mv of m[5]) if (mv) S.moves.set(mv, (S.moves.get(mv) || 0) + 1);
      sp.set(m[0], S);
    }
  }
  const rows = [...sp.entries()].sort((a, b) => b[1].envs - a[1].envs || b[1].mons - a[1].mons);
  const tm = new Set(SNAP.names.tm_moves || []);
  const chips = (S, k) => [...S.moves.entries()].sort((a, b) => b[1] - a[1]).slice(0, k)
    .map(([mv, c]) => `<span class="chip${tm.has(mv) ? " tm" : ""}">${esc(pretty(N.moves[mv]))} <b>${pct(c, S.mons)}</b></span>`).join("");
  $("species").innerHTML = `<tr><th>Species</th><th class="n">In party</th><th class="n">Mons</th><th class="n">Level</th>
    <th class="n">q10–q90</th><th class="n">HP</th><th>Top moves <span class="muted">(share of these mons that know it)</span></th></tr>` +
    (rows.length ? rows.slice(0, 40).map(([id, S]) => {
      const open = openSp.has(id);
      let h = `<tr class="click" data-sp="${id}"><td class="name">${open ? "▾" : "▸"} ${ico(id)}${esc(pretty(N.species[id]))}</td>
        <td class="n">${pct(S.envs, E.length)}</td><td class="n">${S.mons}</td><td class="n">${med(S.lv).toFixed(0)}</td>
        <td class="n">${quant(S.lv, 0.1).toFixed(0)}–${quant(S.lv, 0.9).toFixed(0)}</td><td class="n">${pct(S.hp, S.mons)}</td>
        <td><div class="chips">${chips(S, 6)}</div></td></tr>`;
      if (open) {
        const lv = new Map();
        for (const l of S.lv) { const b = Math.floor(l / 5) * 5; lv.set(b, (lv.get(b) || 0) + 1); }
        const hist = [...lv.entries()].sort((a, b) => a[0] - b[0]).map(([b, c]) => `<span class="chip">L${b}–${b + 4} <b>${c}</b></span>`).join("");
        h += `<tr class="detail"><td colspan="7"><div class="dim" style="margin-bottom:4px">Every move (${S.moves.size})</div>
          <div class="chips">${chips(S, 999)}</div><div class="dim" style="margin:8px 0 4px">Levels</div><div class="chips">${hist}</div></td></tr>`;
      }
      return h;
    }).join("") : `<tr><td colspan="7" class="dim">no env matches the filters</td></tr>`);
}

function renderEnvs(E) {
  const N = SNAP.names;
  const best = (o) => Math.max(0, ...mons(o).map((m) => m[1]));
  const rows = [...E].sort((a, b) => b.b - a.b || best(b) - best(a));
  const shown = showAll ? rows : rows.slice(0, 40);
  $("envnote").textContent = `${shown.length} of ${rows.length}, most badges and highest level first`;
  $("more").style.display = rows.length > shown.length || showAll ? "" : "none";
  $("more").textContent = showAll ? "Show 40" : `Show all ${rows.length}`;
  const where = (o) => (o.battle === 2 ? "trainer battle" : o.battle ? "wild battle" : "");
  $("envs").innerHTML = `<tr><th class="n">Env</th><th class="n">Badges</th><th>Map</th><th>Scope</th><th>Party</th></tr>` +
    shown.map((o) => `<tr><td class="n">${o.e}</td><td class="n">${o.b}</td>
      <td>${esc(pretty(N.maps[o.map] || "?"))}<div class="dim">${where(o)}</div>${o.gap ? `<div class="dim" title="HM in the bag, no party mon knows the move">untaught: ${esc((N.hm_moves || []).map((_, i) => i).filter((i) => o.gap & (1 << i)).map(hmName).join(", "))}</div>` : ""}</td>
      <td>${o.scope >= 0 ? esc(N.scopes[o.scope]) : '<span class="muted">none</span>'}</td>
      <td><div class="party">${o.party.map(monCell).join("")}</div></td></tr>`).join("");
}
function monCell(m) {
  const N = SNAP.names;
  if (!m[0]) return "";
  if (m[6]) return `<div class="mon">${ico(SPECIES_EGG)}<div><span class="sp">Egg</span></div></div>`;
  const f = m[3] ? m[2] / m[3] : 0;
  const c = f > 0.5 ? "var(--good)" : f > 0.2 ? "var(--warn)" : "var(--critical)";
  const mv = m[5].filter(Boolean).map((x) => pretty(N.moves[x])).join(" · ");
  return `<div class="mon">${ico(m[0])}<div><span class="sp">${esc(pretty(N.species[m[0]]))}</span> <span class="lv">L${m[1]}</span>
    <span class="hp" title="${m[2]}/${m[3]} HP"><i style="width:${Math.round(100 * f)}%;background:${c}"></i></span>
    <span class="mv">${esc(mv)}</span></div></div>`;
}

function hmName(i) { const m = (SNAP.names.hm_moves || [])[i]; return m ? pretty(SNAP.names.moves[m]) : `HM0${i + 1}`; }
function itemName(names, i) { return pretty((names.items || {})[i] || `item ${i}`); }
function renderBag(E) {
  const N = SNAP.names, n = E.length;
  const hms = (N.hm_moves || []).map((_, i) => {
    const b = 1 << i, bag = E.filter((o) => o.hm[0] & b), gap = bag.filter((o) => !(o.hm[1] & b));
    return [i, bag.length, E.filter((o) => o.hm[1] & b).length, gap.length, gap.filter((o) => o.hm[2] & b).length];
  }).filter((r) => r[1] || r[2]);
  $("hms").innerHTML = `<tr><th>HM</th><th class="n">In bag</th><th class="n">Taught</th><th class="n" title="in the bag, no party mon knows the move">In bag, not taught</th>
    <th class="n" title="of those, a party mon could learn it">… a mon can learn it</th></tr>` +
    (hms.length ? hms.map(([i, bag, taught, gap, can]) => `<tr><td class="name">${esc(hmName(i))}</td><td class="n">${pct(bag, n)}</td>
      <td class="n">${pct(taught, n)}</td><td class="n">${pct(gap, n)}</td><td class="n">${pct(can, gap)}</td></tr>`).join("")
      : `<tr><td colspan="5" class="dim">no HM in any shown env</td></tr>`);
  const it = new Map();
  for (const o of E) for (const [i, q] of o.bag) { const r = it.get(i) || []; r.push(q); it.set(i, r); }
  const rows = [...it.entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 20);
  $("bag").innerHTML = `<tr><th>Item <span class="muted">(items, balls, berries pockets)</span></th><th class="n">Envs holding</th><th class="n">Qty</th></tr>` +
    (rows.length ? rows.map(([i, q]) => `<tr><td>${esc(itemName(N, i))}</td><td class="n">${pct(q.length, n)}</td>
      <td class="n" title="median of the envs holding it">${med(q).toFixed(0)}</td></tr>`).join("") : `<tr><td colspan="3" class="dim">no items</td></tr>`);
  const st = E.length && E[0].stats ? (N.stats || []).map((k, j) => {
    const v = E.map((o) => o.stats[j]);
    return `<tr><td>${esc(pretty(k))}</td><td class="n">${pct(v.filter((x) => x > 0).length, n)}</td><td class="n">${med(v).toFixed(0)}</td>
      <td class="n">${(v.reduce((a, b) => a + b, 0) / n).toFixed(1)}</td></tr>`;
  }).join("") : "";
  $("stats").innerHTML = st ? `<tr><th>Game counter</th><th class="n">Envs &gt; 0</th><th class="n">Median</th><th class="n">Mean</th></tr>${st}`
    : `<tr><td class="dim">the trainer's records have no game counters</td></tr>`;
}

// ---- behaviour sample ----------------------------------------------------------------------------------------
let sampleTimer = null;
async function pollSample(start) {
  try {
    const d = await getJSON("/api/party/sample" + (start ? `?secs=${$("secs").value}` : ""));
    if (d.error) { $("sstate").textContent = d.error; return; }
    if (d.state === "running") {
      $("start").disabled = true;
      $("sstate").textContent = `sampling… ${Math.round(d.elapsed)} / ${Math.round(d.secs)} s`;
      clearTimeout(sampleTimer);
      sampleTimer = setTimeout(() => pollSample(false), 2000);
      return;
    }
    $("start").disabled = false;
    if (d.state === "done" && d.result) {
      if (d.result.error) { $("sstate").textContent = d.result.error; return; }
      SAMPLE = d.result;
      $("sstate").textContent = `sample of ${Math.round(SAMPLE.secs)} s at ${new Date(SAMPLE.t * 1000).toLocaleTimeString()}`;
      renderSample();
    } else $("sstate").textContent = "no sample yet";
  } catch (e) { $("sstate").textContent = String(e); $("start").disabled = false; }
}
function sumGroups() {
  const c = {}, wm = {};
  for (const g of SAMPLE.groups) {
    if (sel.badges.size && !sel.badges.has(g.b)) continue;
    if (sel.scope === "none" ? g.s !== -1 : sel.scope !== "all" && g.s !== +sel.scope) continue;
    if (!hmOk(g.g)) continue;
    for (const [k, v] of Object.entries(g.c)) c[k] = (c[k] || 0) + v;
    for (const [k, v] of Object.entries(g.wild_maps || {})) wm[k] = (wm[k] || 0) + v;
    for (const f of ["items", "dwell", "dwell_battle"]) {
      const t = (c["@" + f] = c["@" + f] || {});
      for (const [k, v] of Object.entries(g[f] || {})) t[k] = (t[k] || 0) + v;
    }
  }
  return [c, wm];
}
const OUTS = [["ko", "KO", "var(--good)"], ["ran", "Ran", "var(--warn)"], ["caught", "Caught", "var(--bar)"], ["lost", "Lost", "var(--critical)"]];
function stack(o, n) {  // a 100% bar of the wild outcomes
  return `<span style="display:inline-flex;width:140px;height:9px;border-radius:2px;overflow:hidden;background:#232a33;vertical-align:0">` +
    OUTS.map(([k, t, col]) => `<i title="${t} ${pct(o[k] || 0, n)}" style="width:${(100 * (o[k] || 0)) / Math.max(n, 1)}%;background:${col}"></i>`).join("") + "</span>";
}
function wildBlock(c, wm, envMin) {
  const g = (k) => c[k] || 0;
  const n = g("battles_wild");
  if (!n) return "";
  const tot = { ko: g("wild_ko"), ran: g("wild_ran"), caught: g("wild_caught"), lost: g("wild_lost") };
  const maps = {};
  for (const [k, v] of Object.entries(wm)) {
    const [m, o] = k.split(":");
    (maps[m] = maps[m] || { n: 0 })[o] = (maps[m][o] || 0) + v;
    maps[m].n += v;
  }
  const names = SAMPLE.names.maps || [];
  const rows = Object.entries(maps).sort((a, b) => b[1].n - a[1].n).slice(0, 12).map(([m, o]) =>
    `<tr><td>${esc(pretty(names[+m] || "?"))}</td><td class="n">${fmt(o.n)}</td><td class="n">${pct(o.n, n)}</td>
      <td>${stack(o, o.n)}</td>${OUTS.map(([k]) => `<td class="n">${pct(o[k] || 0, o.n)}</td>`).join("")}</tr>`).join("");
  return `<h2 style="margin-top:14px">Wild battles <span class="dim">is it farming? one outcome per battle</span></h2>
    <div class="tiles">
      ${tile("Wild battles", (n / envMin).toFixed(2), "per env-minute")}
      ${tile("KO'd wild mons", (tot.ko / envMin).toFixed(2), `per env-minute · ${pct(tot.ko, n)} of battles`)}
      ${tile("Ran away", pct(tot.ran, n), "of wild battles (or another escape)")}
      ${tile("Caught", pct(tot.caught, n), `${fmt(tot.caught)} mons`)}
      ${tile("Lost", pct(tot.lost, n), "every mon fainted")}
    </div>
    <div class="tablewrap"><table><tr><th>Map</th><th class="n">Wild battles</th><th class="n">Share</th><th>Outcomes</th>
      ${OUTS.map(([, t, col]) => `<th class="n"><span class="bar" style="width:8px;background:${col}"></span> ${t}</th>`).join("")}</tr>
      <tr><td class="name">All maps</td><td class="n">${fmt(n)}</td><td class="n">100%</td><td>${stack(tot, n)}</td>
      ${OUTS.map(([k]) => `<td class="n">${pct(tot[k], n)}</td>`).join("")}</tr>${rows}</table></div>
    <p class="note">KO: a party mon gained EXP (only a KO gives EXP). Caught: a new party mon or a box count increase
      at the end. Lost: every mon fainted. Ran: anything else (running, or Teleport / Roar ending it).</p>`;
}
function itemsBlock(c, envMin) {
  const g = (k) => c[k] || 0, names = SAMPLE.names, items = c["@items"] || {};
  const per = new Map(), KINDS = ["buy", "throw", "use_battle", "use_field", "give", "store", "toss", "sell"];
  for (const [k, v] of Object.entries(items)) {
    const [kind, i] = k.split(":");
    const r = per.get(+i) || {};
    r[kind] = (r[kind] || 0) + v;
    per.set(+i, r);
  }
  const rows = [...per.entries()].sort((a, b) => KINDS.reduce((s, k) => s + (b[1][k] || 0), 0) - KINDS.reduce((s, k) => s + (a[1][k] || 0), 0)).slice(0, 15);
  const sk = (names.skills || []).map((k) => [k, g("sk_" + k)]).filter((r) => r[1]);
  const rate = (k) => `${(g(k) / envMin).toFixed(2)} per env-minute`;
  return `<h2 style="margin-top:14px">Items and shopping <span class="dim">bag changes between copies</span></h2>
    <div class="tiles">
      ${tile("Purchases", fmt(g("buy_events")), `${rate("buy_events")} · ${g("envs_buy")} envs`)}
      ${tile("Balls thrown", fmt(g("throws")), `${rate("throws")} · ${g("envs_throw")} envs`)}
      ${tile("Items used in battle", fmt(g("use_battle")), `not balls · ${g("envs_use_battle")} envs`)}
      ${tile("Items used outside battle", fmt(g("use_field")), `from the bag / party menu · ${g("envs_use_field")} envs`)}
      ${tile("Given / stored / tossed", `${fmt(g("give"))} / ${fmt(g("store"))} / ${fmt(g("toss"))}`, "held item (GIVE) / item PC / thrown away")}
      ${tile("Sales", fmt(g("sell_events")), `${g("envs_sell")} envs`)}
      ${tile("Catches sent to the box", fmt(g("box_auto")), "box count rose outside the PC")}
    </div>
    <div class="grid2"><div class="tablewrap"><table><tr><th>Item</th><th class="n">Bought</th><th class="n">Thrown</th>
      <th class="n">Used in battle</th><th class="n">Used outside</th><th class="n">Given</th><th class="n">Stored</th>
      <th class="n">Tossed</th><th class="n">Sold</th></tr>
      ${rows.length ? rows.map(([i, r]) => `<tr><td>${esc(itemName(names, i))}</td>${KINDS.map((k) => `<td class="n">${r[k] ? fmt(r[k]) : '<span class="muted">0</span>'}</td>`).join("")}</tr>`).join("")
        : `<tr><td colspan="9" class="dim">no bag change seen</td></tr>`}</table></div>
    <div class="tile"><div class="t" title="polls whose one-step skill event bit was set: the sampler sees about 1 in (steps per second / Hz) of them, so compare events with each other, not with the counts above">Skill events seen in the copies (sampled pulses)</div>
      <div class="chips" style="margin-top:4px">${sk.length ? sk.sort((a, b) => b[1] - a[1]).map(([k, v]) => `<span class="chip">${esc(pretty(k))} <b>${fmt(v)}</b></span>`).join("") : '<span class="dim">none</span>'}</div></div></div>
    <p class="note">Bought / sold: money and bag moved in opposite directions on the mart screen. Used: one item fell
      by exactly 1 with money and party unchanged, in battle or on the bag / party menu screen. Outside battle that drop
      is split: given (a party mon now holds one more of it), stored (item PC), tossed (the copy before waited on the
      bag's quantity / yes-no box; a quick toss can still land in Used outside). Units, not purchases.</p>`;
}
function dwellBlock(c, P) {
  const d = c["@dwell"] || {}, db = c["@dwell_battle"] || {}, names = SAMPLE.names.maps || [];
  const rows = Object.entries(d).sort((a, b) => b[1] - a[1]).slice(0, 15);
  if (!rows.length) return "";
  return `<h2 style="margin-top:14px">Where the time goes <span class="dim">share of sampled env time per map</span></h2>
    <div class="tablewrap"><table><tr><th>Map</th><th class="n">Time</th><th></th><th class="n">In battle there</th></tr>
    ${rows.map(([m, v]) => `<tr><td>${esc(pretty(names[+m] || "?"))}</td><td class="n">${pct1(v, P)}</td>
      <td><span class="bar" style="width:${Math.max(2, Math.round((300 * v) / P))}px"></span></td><td class="n">${pct(db[m] || 0, v)}</td></tr>`).join("")}
    </table></div>`;
}
function renderSample() {
  if (!SAMPLE) { $("beh").innerHTML = ""; return; }
  const [c, wm] = sumGroups(), g = (k) => c[k] || 0;
  const P = g("polls");
  if (!P) { $("beh").innerHTML = `<div class="msg">No sampled env matches the Badges / Scope filters.</div>`; return; }
  const envMin = P / SAMPLE.hz / 60;
  const time = [["Battle", "battle"], ["Trainer battle", "trainer_battle"], ["Free on the field", "field"],
    ["Party menu", "party_menu"], ["Start menu", "start_menu"], ["Bag", "bag"], ["TM case", "tm_case"], ["PC box", "pc_box"]];
  const tiles = time.map(([t, k]) => tile(t, pct1(g(k), P), "of the time")).join("");
  const bt = (k) => ["trainer", "wild"].map((t) => g(`${k}_${t}`));
  const row = (label, vals, title) => `<tr title="${esc(title || "")}"><td>${label}</td>${vals.map((v) => `<td class="n">${v}</td>`).join("")}</tr>`;
  const [bT, bW] = bt("battles"), [wT, wW] = bt("won"), [lT, lW] = bt("lead_weaker"), [uT, uW] = bt("switched_up");
  const [xT, xW] = bt("lead_exp"), [sT, sW] = bt("switch_train"), [eT, eW] = bt("exp_split");
  const [vT, vW] = bt("switch_vol"), [fT, fW] = bt("switch_forced");
  const fl = g("trainer_flag_trainer");
  const battles = `<div class="tablewrap"><table><tr><th>Battles (start and end seen)</th><th class="n">Trainer</th><th class="n">Wild</th></tr>
    ${row("Battles", [fmt(bT), fmt(bW)])}
    ${row("Won (wild: KO)", [pct(wT, fl), pct(wW, bW)], "trainer: its defeated flag got set (first-time trainers only); wild: a mon gained EXP (a KO)")}
    ${row("Lost", [pct(g("lost_trainer"), bT), pct(g("wild_lost"), bW)], "every mon fainted")}
    ${row("Lead below the party's best level", [pct(lT, bT), pct(lW, bW)], "the first mon sent out")}
    ${row("… then switched to a stronger mon", [pct(uT, lT), pct(uW, lW)], "while the weaker lead still had HP")}
    ${row("… and the lead still got EXP", [pct(sT, lT), pct(sW, lW)], "switch training: weaker lead out, stronger mon finishes, both share the EXP")}
    ${row("Weaker lead got EXP at all", [pct(xT, lT), pct(xW, lW)])}
    ${row("EXP shared by 2+ mons", [pct(eT, bT), pct(eW, bW)])}
    ${row("Voluntary switches per battle", [(vT / Math.max(bT, 1)).toFixed(2), (vW / Math.max(bW, 1)).toFixed(2)], "the active mon changed while it still had HP")}
    ${row("Switches after a faint per battle", [(fT / Math.max(bT, 1)).toFixed(2), (fW / Math.max(bW, 1)).toFixed(2)])}
    </table></div>`;
  const kv = (k, v, t) => `<dt title="${esc(t || "")}">${k}</dt><dd>${v}</dd>`;
  const party = `<div class="tile"><dl class="kv">
    ${kv("Reorders outside battle", `${fmt(g("reorder"))} · ${(g("reorder") / envMin).toFixed(2)} per env-minute`)}
    ${kv("Envs that reordered", `${g("envs_reorder")}`)}
    ${kv("… leaving a weaker lead", pct(g("reorder_weak_lead"), g("reorder")), "after the reorder, slot 1 is below the best level")}
    ${kv("Field time with a weaker lead", pct(g("lead_weaker_field"), P), "outside battle, share of the time")}
    ${kv("PC deposits / withdrawals", `${g("pc_deposit")} / ${g("pc_withdraw")} · ${g("envs_pc")} envs`)}
    ${kv("Mons added (catch, gift)", fmt(g("added_outside_pc")))}
    ${kv("Moves learned", `${fmt(g("moves_learned"))} · ${g("moves_learned_tm")} are TM/HM moves`, "a TM/HM move can also come from a level-up")}
    ${kv("Evolutions", fmt(g("evolutions")))}
    ${kv("Shift prompt seen", `${g("shift_prompt")} polls`, "waiting on the battle YES/NO box")}
    </dl></div>`;
  $("beh").innerHTML = `<div class="tiles">${tiles}</div><div class="grid2">${battles}${party}</div>${wildBlock(c, wm, envMin)}
    ${itemsBlock(c, envMin)}${dwellBlock(c, P)}
    <p class="note">${fmt(envMin)} env-minutes sampled · ${SAMPLE.sweeps} copies at ${SAMPLE.hz.toFixed(1)} Hz ·
    sampler CPU ${SAMPLE.cpu_s.toFixed(1)} s (of one core) · ${fmt(g("restarts"))} restored states skipped</p>`;
}

// ---- events --------------------------------------------------------------------------------------------------
$("badges").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  if (b.dataset.b === "all") sel.badges.clear();
  else { const v = +b.dataset.b; sel.badges.has(v) ? sel.badges.delete(v) : sel.badges.add(v); }
  save(); buildFilters(); render();
});
$("scope").addEventListener("change", (ev) => { sel.scope = ev.target.value; save(); render(); });
$("where").addEventListener("change", (ev) => { sel.where = ev.target.value; save(); render(); });
$("hm").addEventListener("change", (ev) => { sel.hm = ev.target.value; save(); render(); });
$("q").addEventListener("input", (ev) => { sel.q = ev.target.value.trim(); render(); });
$("refresh").addEventListener("click", () => refresh(true));
$("more").addEventListener("click", () => { showAll = !showAll; render(); });
$("start").addEventListener("click", () => pollSample(true));
$("species").addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr[data-sp]");
  if (!tr) return;
  const id = +tr.dataset.sp;
  openSp.has(id) ? openSp.delete(id) : openSp.add(id);
  render();
});
$("auto").addEventListener("click", () => {
  const on = $("auto").getAttribute("aria-pressed") !== "true";
  $("auto").setAttribute("aria-pressed", String(on));
  clearInterval(autoTimer);
  if (on) autoTimer = setInterval(() => { if (document.visibilityState === "visible") refresh(false); }, 60000);
});
refresh(false);
pollSample(false);
