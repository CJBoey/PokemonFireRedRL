// fireredsim map visualizer page.
//
// Adapted from PWhiddy/pokerl-map-viz visualizer_live.js (MIT License, Copyright (c) 2024 Peter Whidden;
// LICENSE-pokerl-map-viz.txt): a PixiJS stage with wheel/drag/pinch navigation, one sprite per env walking
// that env's batch of positions over the interval between batches. Changed for fireredsim: the FireRed
// world of build_world.py (an overview image plus full-resolution chunks loaded only when zoomed in and in
// view; story and full variants), serve.py's batch format, the FireRed player sprite, and activity
// reporting: the server reads the training only while some page is visible, not paused and not idle; and a
// dwell heatmap of the server's position samples (serve.py Heat), one texture pixel per world tile; and the
// novelty view of the run-long tile counts behind the tile-count bonus (serve.py Novelty, from checkpoints).
"use strict";

const TILE = 16;
const DETAIL_SCALE = 0.45;   // zoom from which the full-resolution chunks are drawn
const MAX_CHUNKS = 16;       // chunk textures kept loaded (2048 px: 16 MB of GPU memory each)
const STALE_MS = 30000;      // an env without updates for this long is removed
const WARP_TILES = 3;        // consecutive positions further apart than this are a warp: jump, no glide
const DOT_SCALE = 0.5;       // below this zoom an env is a dot of fixed screen size (badge colour), not a sprite
const DOT_PX = 9;            // dot diameter on screen
const params = new URLSearchParams(location.search);
const IDLE_MS = 60000 * Number(params.get("idle") ?? 10);  // no input for this long pauses (0 = never)
const HEAT_MAGIC = 0x54414548;   // "HEAT": serve.py Heat.frame
const HEAT_TOP = 0.995;          // the colour scale tops out at this quantile of the visited tiles (log scale)
const HEAT_STOPS = [[0, 0x3b0f70], [0.25, 0x8c2981], [0.5, 0xde4968], [0.75, 0xfe9f6d], [1, 0xfcfdbf]];  // magma
const NOV_MAGIC = 0x4C564F4E;    // "NOVL": serve.py Novelty.frame
const NOV_STOPS = [[0, 0x440154], [0.25, 0x3b528b], [0.5, 0x21918c], [0.75, 0x5ec962], [1, 0xfde725]];  // viridis
const WALK_LAND = 1, WALK_WATER = 2;  // build_world.py walk.bin (0 no map, 3 blocked)
const FRONTIER_LAND = 0xff4d8d, FRONTIER_WATER = 0x00d8ff, NEW_TILE = 0xffffff, STALE_TILE = 0x3a4450;
const UNREACHED_LAND = 0x9aa3ad, UNREACHED_WATER = 0x6e8fb0;
const FRONTIER_PX = 6;        // frontier markers are at least this many screen pixels (a tile when zoomed in)
const BADGE_COLORS = [0xffffff, 0xff7b72, 0x58a6ff, 0xffcb05, 0x7ee787, 0xff9bce, 0xffa657, 0xd2a8ff, 0x39d0d8];  // by badge count

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private window */ } },
};

const app = new PIXI.Application({ resizeTo: window, background: 0x0b0d10, antialias: false });
document.body.prepend(app.view);
const world = new PIXI.Container();
const bgLayer = new PIXI.Container();
const chunkLayer = new PIXI.Container();
const heatLayer = new PIXI.Container();
const novLayer = new PIXI.Container();
const agentLayer = new PIXI.Container();
agentLayer.sortableChildren = true;
world.addChild(bgLayer, chunkLayer, heatLayer, novLayer, agentLayer);
app.stage.addChild(world);

const $ = (id) => document.getElementById(id);
let meta = null;              // world.json of the shown variant
let regions = new Map();      // map_group * 256 + map_num -> [x, y] of the map's top-left metatile
let regionSize = new Map();   // the same ids -> [width, height] in metatiles
let tileRegion = null;        // Int16Array [world tiles]: index in meta.regions of the map drawn there, -1 none
let chunks = [];              // world.json chunks + {sprite, used}
let variant = params.get("map") || store.get("mapviz.variant") || "story";
let showLabels = store.get("mapviz.labels") === "1";
let showEnvs = store.get("mapviz.envs") !== "0";
let heatOn = store.get("mapviz.heat") === "1";
let heatBattles = store.get("mapviz.heatBattles") !== "0";
let novOn = store.get("mapviz.nov") === "1" && !heatOn;
let novMode = store.get("mapviz.novMode") === "recent" ? "recent" : "left";
let novFrontier = store.get("mapviz.novFrontier") !== "0";
let filter = null;
let frames = null;            // player textures: down, up, left (right = left mirrored)
let dotTex = null;
const agents = new Map();     // env id -> state

// ------------------------------------------------------------------ world

async function loadWorld(v) {
  const res = await fetch(`world/${v}/world.json`, { cache: "no-cache" });
  if (!res.ok) throw new Error(`world/${v}/world.json: ${res.status} (run tools/mapviz/build_world.py)`);
  const m = await res.json();
  for (const c of chunks) if (c.sprite) c.sprite.destroy({ texture: true, baseTexture: true });
  for (const s of bgLayer.removeChildren()) s.destroy({ texture: true, baseTexture: true });
  for (const a of agents.values()) a.box.destroy({ children: true });
  agents.clear();
  meta = m;
  variant = v;
  store.set("mapviz.variant", v);
  regions = new Map(m.regions.map((r) => [Number(r.id), r.coordinates]));
  regionSize = new Map(m.regions.map((r) => [Number(r.id), r.tileSize]));
  const tw = Math.ceil(m.width / TILE), th = Math.ceil(m.height / TILE);
  tileRegion = new Int16Array(tw * th).fill(-1);
  m.regions.forEach((r, i) => {
    const [x0, y0] = r.coordinates, [w, h] = r.tileSize;
    for (let y = Math.max(0, y0); y < Math.min(th, y0 + h); y++) tileRegion.fill(i, y * tw + Math.max(0, x0), y * tw + Math.min(tw, x0 + w));
  });
  resetHeatLayer(tw, th);
  resetNovLayer(tw, th);
  walk = null;
  if (m.walk) {
    try {
      const r = await fetch(`world/${v}/${m.walk}?v=${m.version}`);
      const b = r.ok ? new Uint8Array(await r.arrayBuffer()) : null;
      walk = b && b.length === tw * th ? b : null;
    } catch (e) { walk = null; }
  }
  const ov = PIXI.Sprite.from(`world/${v}/${m.overview.file}?v=${m.version}`,
    { scaleMode: PIXI.SCALE_MODES.LINEAR, mipmap: PIXI.MIPMAP_MODES.ON });
  ov.scale.set(m.overview.scale);
  bgLayer.addChild(ov);
  chunks = m.chunks.map((c) => ({ ...c, sprite: null, used: 0 }));
  if (heat) drawHeat();
  if (nov) drawNov();
  $("v-story").setAttribute("aria-pressed", v === "story");
  $("v-full").setAttribute("aria-pressed", v === "full");
}

function updateChunks(now) {
  const s = world.scale.x;
  chunkLayer.visible = s >= DETAIL_SCALE;
  if (chunkLayer.visible) {
    const x0 = -world.x / s, y0 = -world.y / s;
    const x1 = x0 + app.screen.width / s, y1 = y0 + app.screen.height / s;
    for (const c of chunks) {
      if (c.x >= x1 || c.x + c.w <= x0 || c.y >= y1 || c.y + c.h <= y0) continue;
      c.used = now;
      if (!c.sprite) {
        c.sprite = PIXI.Sprite.from(`world/${variant}/${c.file}?v=${meta.version}`,
          { scaleMode: PIXI.SCALE_MODES.NEAREST, mipmap: PIXI.MIPMAP_MODES.OFF });
        c.sprite.position.set(c.x, c.y);
        chunkLayer.addChild(c.sprite);
      }
    }
  }
  const loaded = chunks.filter((c) => c.sprite);
  if (loaded.length > MAX_CHUNKS) {
    loaded.sort((a, b) => a.used - b.used);
    for (const c of loaded.slice(0, loaded.length - MAX_CHUNKS)) {
      if (now - c.used < 1000) break;
      c.sprite.destroy({ texture: true, baseTexture: true });
      c.sprite = null;
    }
  }
}

// ------------------------------------------------------------------ dwell heatmap

let heat = null;              // the server's table: {keys, field, battle, seconds, ticks, samples, since}
let heatGrid = null;          // Float32Array [world tiles]: shown env samples per tile
let heatCanvas = null, heatTex = null, heatSprite = null, heatDim = null, heatTop = 1;

function makeLut(stops) {     // 256 RGB entries interpolated between the stops
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1], f = (t - t0) / (t1 - t0);
    for (let ch = 0; ch < 3; ch++) {
      const a = (c0 >> (16 - 8 * ch)) & 255, b = (c1 >> (16 - 8 * ch)) & 255;
      lut[3 * i + ch] = Math.round(a + (b - a) * f);
    }
  }
  return lut;
}
const HEAT_LUT = makeLut(HEAT_STOPS), NOV_LUT = makeLut(NOV_STOPS);
const gradient = (stops) => `linear-gradient(90deg, ${stops.map(([t, c]) => `${hex(c)} ${100 * t}%`).join(", ")})`;
const hex = (c) => `#${c.toString(16).padStart(6, "0")}`;

function resetHeatLayer(tw, th) {
  for (const c of heatLayer.removeChildren()) c.destroy({ texture: true, baseTexture: true });
  heatDim = new PIXI.Graphics().beginFill(0x000000, 0.55).drawRect(0, 0, tw * TILE, th * TILE).endFill();
  heatCanvas = document.createElement("canvas");
  heatCanvas.width = tw;
  heatCanvas.height = th;
  heatTex = PIXI.Texture.from(heatCanvas, { scaleMode: PIXI.SCALE_MODES.NEAREST, mipmap: PIXI.MIPMAP_MODES.OFF });
  heatSprite = new PIXI.Sprite(heatTex);
  heatSprite.scale.set(TILE);
  heatLayer.addChild(heatDim, heatSprite);
  heatGrid = new Float32Array(tw * th);
  heatLayer.visible = heatOn;
}

function onHeat(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== HEAT_MAGIC || dv.getUint32(4, true) !== 1) return;
  const n = dv.getUint32(8, true);
  heat = { seconds: dv.getFloat64(16, true), ticks: dv.getFloat64(24, true), samples: dv.getFloat64(32, true),
           since: dv.getFloat64(40, true), keys: new Uint32Array(buf, 48, n),
           field: new Float32Array(buf, 48 + 4 * n, n), battle: new Float32Array(buf, 48 + 8 * n, n) };
  if (heatOn && meta) drawHeat();
}

function drawHeat() {
  if (!heat || !heatCanvas) return;
  const tw = heatCanvas.width, th = heatCanvas.height;
  heatGrid.fill(0);
  const { keys, field, battle } = heat;
  const vals = [];
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i], o = regions.get(k >>> 16), sz = regionSize.get(k >>> 16);
    if (!o) continue;  // a map the shown variant lacks
    const x = k & 255, y = (k >>> 8) & 255;
    if (x >= sz[0] || y >= sz[1]) continue;
    const gx = o[0] + x, gy = o[1] + y;
    if (gx < 0 || gy < 0 || gx >= tw || gy >= th) continue;
    const v = field[i] + (heatBattles ? battle[i] : 0);
    if (v > 0) { heatGrid[gy * tw + gx] += v; vals.push(v); }
  }
  vals.sort((a, b) => a - b);
  heatTop = vals.length ? vals[Math.min(vals.length - 1, Math.floor(HEAT_TOP * vals.length))] : 1;
  const ctx = heatCanvas.getContext("2d");
  const img = ctx.createImageData(tw, th), px = img.data, norm = Math.log1p(Math.max(heatTop, 1));
  for (let i = 0; i < heatGrid.length; i++) {
    const v = heatGrid[i];
    if (!v) continue;
    const t = Math.min(1, Math.log1p(v) / norm), c = Math.round(t * 255);
    px[4 * i] = HEAT_LUT[3 * c]; px[4 * i + 1] = HEAT_LUT[3 * c + 1]; px[4 * i + 2] = HEAT_LUT[3 * c + 2];
    px[4 * i + 3] = Math.round(255 * (0.55 + 0.45 * t));
  }
  ctx.putImageData(img, 0, 0);
  heatTex.baseTexture.update();
  renderHeatInfo();
}

function fmtSec(sec) {
  return sec < 90 ? `${sec.toFixed(sec < 10 ? 1 : 0)} s` : sec < 5400 ? `${(sec / 60).toFixed(1)} min` : `${(sec / 3600).toFixed(1)} h`;
}

function envTime(samples) {  // env samples -> env time (one sample of one env = the window's wall time / its samples)
  return fmtSec(heat && heat.ticks ? samples * heat.seconds / heat.ticks : 0);
}

function renderHeatInfo() {
  if (!heatOn) return;
  if (!heat) { $("heat-sum").textContent = "waiting for the server…"; return; }
  const since = new Date(heat.since * 1000);
  const f = heat.field.reduce((a, b) => a + b, 0), b = heat.battle.reduce((a, c) => a + c, 0);
  $("heat-sum").textContent = `${heat.keys.length} tiles · ${fmtSec(heat.seconds)} sampled since ` +
    `${since.toLocaleDateString()} ${since.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })} · ` +
    `battle ${f + b ? (100 * b / (f + b)).toFixed(1) : 0}%`;
  $("heat-lo").textContent = "1 sample";
  $("heat-hi").textContent = `≥ ${envTime(heatTop)} env time`;
}

function hoverHeat(sx, sy) {
  if (!heatOn || !heat || !heatCanvas) return;
  const s = world.scale.x, tw = heatCanvas.width;
  const gx = Math.floor((sx - world.x) / s / TILE), gy = Math.floor((sy - world.y) / s / TILE);
  const el = $("heat-at");
  if (gx < 0 || gy < 0 || gx >= tw || gy >= heatCanvas.height || tileRegion[gy * tw + gx] < 0) { el.textContent = ""; return; }
  const r = meta.regions[tileRegion[gy * tw + gx]], v = heatGrid[gy * tw + gx];
  el.textContent = `${r.name} (${gx - r.coordinates[0]}, ${gy - r.coordinates[1]}): ` +
    (v ? `${envTime(v)} env time · ${(100 * v / heat.samples).toPrecision(2)}% of all` : "not visited");
}

// ------------------------------------------------------------------ novelty (the tile-count bonus)
//
// A first visit (in an episode) to a tile that N earlier episodes first-visited pays W * share, share =
// min(1, sqrt(S / (N + 1))) (reward.py; S = --tile-count-scale, W = --tile-count-reward). "Left" colours each
// reached tile by that share; "Recent" by the first visits between two checkpoints about --novelty-window apart,
// new tiles (no count in the older one) white. Never-reached passable tiles are a grey (land) or blue (water)
// checker; the frontier (never-reached passable tiles next to a reached one) is solid pink (land) or cyan (water,
// needs Surf). walk.bin (build_world.py) gives passable / water / blocked.

let nov = null;               // the server's counts: {scale, weight, t1, s1, u1, t0, s0, u0, keys, count, delta}
let walk = null;              // Uint8Array [world tiles] of walk.bin, null for a world built without it
let novCount = null, novDelta = null, novFront = null;  // per world tile: count (-1 never reached), first visits, frontier 1 land 2 water
let novCanvas = null, novTex = null, novTop = 1, novStats = null;
let novMarks = null;          // PIXI.Container of the frontier markers (they keep a minimum screen size)

function resetNovLayer(tw, th) {
  for (const c of novLayer.removeChildren()) c.destroy({ texture: true, baseTexture: true });
  const dim = new PIXI.Graphics().beginFill(0x000000, 0.6).drawRect(0, 0, tw * TILE, th * TILE).endFill();
  novCanvas = document.createElement("canvas");
  novCanvas.width = tw;
  novCanvas.height = th;
  novTex = PIXI.Texture.from(novCanvas, { scaleMode: PIXI.SCALE_MODES.NEAREST, mipmap: PIXI.MIPMAP_MODES.OFF });
  const sp = new PIXI.Sprite(novTex);
  sp.scale.set(TILE);
  novMarks = new PIXI.Container();
  novLayer.addChild(dim, sp, novMarks);
  novCount = new Int32Array(tw * th);
  novDelta = new Uint32Array(tw * th);
  novFront = new Uint8Array(tw * th);
  novLayer.visible = novOn;
}

function onNovelty(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(4, true) !== 1) return;
  const n = dv.getUint32(8, true), f = (i) => dv.getFloat64(16 + 8 * i, true);
  nov = { scale: f(0), weight: f(1), t1: f(2), s1: f(3), u1: f(4), t0: f(5), s0: f(6), u0: f(7),
          keys: new Uint32Array(buf, 80, n), count: new Uint32Array(buf, 80 + 4 * n, n),
          delta: new Uint32Array(buf, 80 + 8 * n, n) };
  if (novOn && meta) drawNov();
}

const novShare = (c) => Math.min(1, Math.sqrt(nov.scale / (c + 1)));

function drawNov() {
  if (!nov || !novCanvas) return;
  const tw = novCanvas.width, th = novCanvas.height, N = tw * th;
  novCount.fill(-1);
  novDelta.fill(0);
  novFront.fill(0);
  const { keys, count, delta } = nov;
  const reached = [], deltas = [];
  let full = 0, visits = 0, newTiles = 0, paid = 0;
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i], o = regions.get(k >>> 16), sz = regionSize.get(k >>> 16);
    if (!o) continue;
    const x = k & 255, y = (k >>> 8) & 255;
    if (x >= sz[0] || y >= sz[1]) continue;
    const gx = o[0] + x, gy = o[1] + y;
    if (gx < 0 || gy < 0 || gx >= tw || gy >= th) continue;
    const g = gy * tw + gx, c = count[i], d = delta[i];
    novCount[g] = c;
    novDelta[g] = d;
    reached.push(g);
    if (c + 1 <= nov.scale) full++;
    if (d) { deltas.push(d); visits += d; paid += d * novShare(c - d / 2); if (d === c) newTiles++; }
  }
  let frontLand = 0, frontWater = 0;
  if (walk) {
    for (const g of reached) {
      const gx = g % tw;
      for (const j of [g - tw, g + tw, gx > 0 ? g - 1 : -1, gx < tw - 1 ? g + 1 : -1]) {
        if (j < 0 || j >= N || novCount[j] >= 0 || novFront[j]) continue;
        if (walk[j] === WALK_LAND) { novFront[j] = 1; frontLand++; }
        else if (walk[j] === WALK_WATER) { novFront[j] = 2; frontWater++; }
      }
    }
  }
  deltas.sort((a, b) => a - b);
  novTop = deltas.length ? Math.max(2, deltas[Math.min(deltas.length - 1, Math.floor(HEAT_TOP * deltas.length))]) : 2;
  const shares = reached.map((g) => novShare(novCount[g])).sort((a, b) => a - b);
  novStats = { reached: reached.length, full, median: shares.length ? shares[shares.length >> 1] : 0, visits, newTiles,
               paid, frontLand, frontWater };

  const ctx = novCanvas.getContext("2d");
  const img = ctx.createImageData(tw, th), px = img.data, norm = Math.log1p(novTop);
  const put = (i, c, a) => { px[4 * i] = c >> 16; px[4 * i + 1] = (c >> 8) & 255; px[4 * i + 2] = c & 255; px[4 * i + 3] = a; };
  const putLut = (i, lut, t, a) => {
    const c = Math.round(Math.min(1, Math.max(0, t)) * 255);
    px[4 * i] = lut[3 * c]; px[4 * i + 1] = lut[3 * c + 1]; px[4 * i + 2] = lut[3 * c + 2]; px[4 * i + 3] = a;
  };
  if (walk) {
    for (let i = 0; i < N; i++) {
      const w = walk[i];
      if ((w === WALK_LAND || w === WALK_WATER) && tileRegion[i] >= 0) {
        const odd = ((i % tw) + ((i / tw) | 0)) & 1;
        put(i, w === WALK_LAND ? UNREACHED_LAND : UNREACHED_WATER, odd ? 110 : 45);
      }
    }
  }
  for (const g of reached) {
    if (novMode === "left") putLut(g, NOV_LUT, novShare(novCount[g]), 235);
    else if (!novDelta[g]) put(g, STALE_TILE, 170);
    else if (novDelta[g] === novCount[g] && nov.u0 !== nov.u1) put(g, NEW_TILE, 255);
    else putLut(g, HEAT_LUT, Math.log1p(novDelta[g]) / norm, 235);
  }
  for (const c of novMarks.removeChildren()) c.destroy();
  if (novFrontier) {
    for (let i = 0; i < N; i++) {
      if (!novFront[i]) continue;
      const m = new PIXI.Sprite(PIXI.Texture.WHITE);
      m.anchor.set(0.5);
      m.position.set(((i % tw) + 0.5) * TILE, (((i / tw) | 0) + 0.5) * TILE);
      m.tint = novFront[i] === 1 ? FRONTIER_LAND : FRONTIER_WATER;
      novMarks.addChild(m);
    }
    scaleNovMarks();
  }
  ctx.putImageData(img, 0, 0);
  novTex.baseTexture.update();
  renderNovInfo();
}

function scaleNovMarks() {
  if (!novMarks || !novMarks.children.length) return;
  const px = Math.max(TILE, FRONTIER_PX / world.scale.x);
  if (novMarks.children[0].width === px) return;
  for (const m of novMarks.children) m.width = m.height = px;
}

const fmtN = (n) => Math.round(n).toLocaleString();

function renderNovInfo() {
  if (!novOn) return;
  $("nov-left").setAttribute("aria-pressed", novMode === "left");
  $("nov-recent").setAttribute("aria-pressed", novMode === "recent");
  $("nov-frontier").setAttribute("aria-pressed", novFrontier);
  const err = lastStatus && lastStatus.nov_error;
  if (!nov || !novStats) { $("nov-sum").textContent = err || "waiting for a checkpoint…"; return; }
  const st = novStats, W = nov.weight, hasW = Number.isFinite(W);
  const age = Math.max(0, Date.now() / 1000 - nov.t1), span = nov.t1 - nov.t0, steps = nov.s1 - nov.s0;
  if (novMode === "left") {
    $("nov-lo").textContent = "spent 0%";
    $("nov-bar").style.background = gradient(NOV_STOPS);
    $("nov-hi").textContent = `100%${hasW ? ` = ${W}` : ""} per first visit (capped at 100% up to ${fmtN(nov.scale)} visits, ` +
      `50% at ${fmtN(4 * nov.scale)})`;
  } else {
    $("nov-lo").textContent = "1";
    $("nov-bar").style.background = gradient(HEAT_STOPS);
    $("nov-hi").textContent = `≥ ${fmtN(novTop)} first visits`;
  }
  const sw = (c, label, checker) => `<span class="sw${checker ? " ck" : ""}" style="--c:${hex(c)}"></span>${label}`;
  $("nov-keys").innerHTML = [
    novFrontier && walk ? sw(FRONTIER_LAND, `frontier ${fmtN(st.frontLand)}`) : "",
    novFrontier && walk ? sw(FRONTIER_WATER, `water frontier ${fmtN(st.frontWater)}`) : "",
    walk ? sw(UNREACHED_LAND, "never reached", true) : "rebuild the world (build_world.py) for the frontier",
    novMode === "recent" ? sw(NEW_TILE, `new ${fmtN(st.newTiles)}`) + sw(STALE_TILE, "none recently") : "",
  ].filter(Boolean).map((t) => `<span>${t}</span>`).join("");
  const head = `ckpt ${nov.u1} (${fmtSec(age)} old)`;
  if (novMode === "left") {
    $("nov-sum").textContent = `${head} · ${fmtN(st.reached)} tiles reached · ${fmtN(st.full)} at full pay · ` +
      `median pays ${(100 * st.median).toFixed(0)}%`;
  } else if (nov.u0 === nov.u1) {
    $("nov-sum").textContent = `${head} · needs a second checkpoint (one so far)`;
  } else {
    const per = steps > 0 ? 1 / steps : 0;
    $("nov-sum").textContent = `ckpt ${nov.u0}→${nov.u1} (${fmtSec(span)}, ${(steps / 1e6).toFixed(1)}M steps) · ` +
      `${fmtN(st.visits)} first visits · ${fmtN(st.newTiles)} new tiles · ` +
      (hasW ? `bonus ≈ ${(st.paid * W * per).toPrecision(2)} per env-step` : `${(st.paid * per).toPrecision(2)} full-pay visits per env-step`);
  }
}

function hoverNov(sx, sy) {
  if (!novOn || !nov || !novCanvas) return;
  const s = world.scale.x, tw = novCanvas.width;
  const gx = Math.floor((sx - world.x) / s / TILE), gy = Math.floor((sy - world.y) / s / TILE);
  const el = $("nov-at");
  if (gx < 0 || gy < 0 || gx >= tw || gy >= novCanvas.height || tileRegion[gy * tw + gx] < 0) { el.textContent = ""; return; }
  const g = gy * tw + gx, r = meta.regions[tileRegion[g]], c = novCount[g];
  const where = `${r.name} (${gx - r.coordinates[0]}, ${gy - r.coordinates[1]}): `;
  if (c >= 0) {
    const sh = novShare(c), W = nov.weight;
    el.textContent = where + `${fmtN(c)} first visits · next pays ${(100 * sh).toFixed(0)}%` +
      (Number.isFinite(W) ? ` (${(W * sh).toPrecision(2)})` : "") +
      (nov.u0 !== nov.u1 ? ` · +${fmtN(novDelta[g])} in ${fmtSec(nov.t1 - nov.t0)}` : "");
  } else {
    const w = walk ? walk[g] : -1;
    el.textContent = where + (w === WALK_LAND ? "never reached" : w === WALK_WATER ? "never reached (water, Surf)" :
      w === 3 ? "blocked" : "never reached") + (novFront[g] ? " · frontier" : "");
  }
}

// ------------------------------------------------------------------ navigation

function zoomAt(sx, sy, factor) {
  const s = Math.min(8, Math.max(0.02, world.scale.x * factor));
  const k = s / world.scale.x;
  world.x = sx - (sx - world.x) * k;
  world.y = sy - (sy - world.y) * k;
  world.scale.set(s);
}

function fit(x, y, w, h) {
  const s = Math.min(app.screen.width / w, app.screen.height / h) * 0.92;
  world.scale.set(s);
  world.x = (app.screen.width - w * s) / 2 - x * s;
  world.y = (app.screen.height - h * s) / 2 - y * s;
}

const home = () => meta && fit(...meta.kanto);
const view = app.view;
const pointers = new Map();
let drag = null, pinch = null;
view.addEventListener("pointerdown", (e) => {
  view.setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1) drag = { x: e.clientX - world.x, y: e.clientY - world.y };
  else { drag = null; const [a, b] = [...pointers.values()]; pinch = Math.hypot(a.x - b.x, a.y - b.y); }
});
view.addEventListener("pointermove", (e) => {
  hoverHeat(e.clientX, e.clientY);
  hoverNov(e.clientX, e.clientY);
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size === 1 && drag) {
    world.x = e.clientX - drag.x;
    world.y = e.clientY - drag.y;
  } else if (pointers.size === 2 && pinch) {
    const [a, b] = [...pointers.values()];
    const d = Math.hypot(a.x - b.x, a.y - b.y);
    zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, d / pinch);
    pinch = d;
  }
});
const release = (e) => {
  pointers.delete(e.pointerId);
  pinch = null;
  drag = null;
  if (pointers.size === 1) {
    const p = [...pointers.values()][0];
    drag = { x: p.x - world.x, y: p.y - world.y };
  }
};
view.addEventListener("pointerup", release);
view.addEventListener("pointercancel", release);
view.addEventListener("wheel", (e) => {
  e.preventDefault();
  zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0015));
}, { passive: false });

// ------------------------------------------------------------------ envs

function labelText(a) { return `e${a.env} b${a.badges}`; }

function applyLook(a) {
  const shown = !filter || filter.test(labelText(a));
  a.box.visible = shown && a.path !== null;
  a.box.alpha = a.battle ? 0.45 : 1;  // in battle: dimmed
  a.dot.tint = BADGE_COLORS[Math.min(a.badges, 8)];
  if (showLabels && !a.label) {
    a.label = new PIXI.Text("", { fontFamily: "system-ui, sans-serif", fontSize: 11, fill: 0xffffff,
      stroke: 0x000000, strokeThickness: 3 });
    a.label.anchor.set(0.5, 1);
    a.box.addChild(a.label);
  }
  if (a.label) {
    a.label.visible = showLabels;
    a.label.text = labelText(a);
    a.label.style.fill = BADGE_COLORS[Math.min(a.badges, 8)];
  }
}

function onBatch(msg, now) {
  if (!frames || !meta) return;
  for (const [env, badges, battle, flat] of msg.envs) {
    const path = [];
    for (let i = 0; i < flat.length; i += 3) {
      const o = regions.get(flat[i + 2]);  // (a map of the other variant only: no point)
      if (o) path.push(o[0] + flat[i] + 0.5, o[1] + flat[i + 1] + 0.5);
    }
    let a = agents.get(env);
    if (!a) {
      const box = new PIXI.Container();
      const sprite = new PIXI.Sprite(frames.down);
      sprite.anchor.set(0.5, 0.75);  // the frame's lower half covers the tile
      const dot = new PIXI.Sprite(dotTex);
      dot.anchor.set(0.5);
      box.addChild(sprite, dot);
      agentLayer.addChild(box);
      a = { env, box, sprite, dot, label: null, path: null, start: 0, duration: 1000, interval: null, last: null,
            updated: now, badges: 0, battle: 0 };
      agents.set(env, a);
    }
    if (a.last !== null) {
      const dt = now - a.last;
      a.interval = a.interval === null ? dt : 0.7 * a.interval + 0.3 * dt;
    }
    a.last = a.updated = now;
    a.path = path.length ? path : null;
    a.start = now;
    a.duration = Math.max(250, 0.95 * (a.interval ?? msg.dt * 1000));
    a.badges = badges;
    a.battle = battle;
    applyLook(a);
  }
}

function animate(now) {
  const s = world.scale.x, far = s < DOT_SCALE;
  const dotScale = DOT_PX / 16 / s, labelY = -(far ? DOT_PX / 2 + 1 : 26 * s) / s;
  for (const [env, a] of agents) {
    if (now - a.updated > STALE_MS) {
      a.box.destroy({ children: true });
      agents.delete(env);
      continue;
    }
    if (!a.path) continue;
    a.sprite.visible = !far;
    a.dot.visible = far;
    if (far) a.dot.scale.set(dotScale);
    if (a.label && showLabels) {  // labels keep their screen size
      a.label.scale.set(1 / s);
      a.label.y = labelY;
    }
    const n = a.path.length / 2;
    const p = Math.min((now - a.start) / a.duration, 1) * (n - 1);
    const i = Math.floor(p), j = Math.min(i + 1, n - 1), f = p - i;
    const x0 = a.path[2 * i], y0 = a.path[2 * i + 1];
    const dx = a.path[2 * j] - x0, dy = a.path[2 * j + 1] - y0;
    const glide = Math.abs(dx) + Math.abs(dy) <= WARP_TILES ? f : 0;
    a.box.x = (x0 + dx * glide) * TILE;
    a.box.y = (y0 + dy * glide) * TILE;
    a.box.zIndex = a.box.y;
    if (dx || dy) {
      if (Math.abs(dx) > Math.abs(dy)) {
        a.sprite.texture = frames.left;
        a.sprite.scale.x = dx > 0 ? -1 : 1;
      } else {
        a.sprite.texture = dy > 0 ? frames.down : frames.up;
        a.sprite.scale.x = 1;
      }
    }
  }
}

function refreshBadges() {
  const counts = new Array(9).fill(0);
  for (const a of agents.values()) counts[Math.min(a.badges, 8)]++;
  const parts = counts.map((c, b) => (c ? `<span style="color:#${BADGE_COLORS[b].toString(16).padStart(6, "0")}">` +
    `${b} badge${b === 1 ? "" : "s"}: ${c}</span>` : "")).join("");
  $("badges").innerHTML = parts || '<span class="dim">no envs yet</span>';
}

// ------------------------------------------------------------------ server connection and activity

let ws = null, userPaused = false, lastInput = performance.now(), sentActive = null, lastStatus = null;

function isActive() {
  return document.visibilityState === "visible" && !userPaused &&
    (IDLE_MS <= 0 || performance.now() - lastInput < IDLE_MS);
}

function reportActivity(force) {
  const active = isActive();
  if (ws && ws.readyState === WebSocket.OPEN && (force || active !== sentActive)) {
    ws.send(JSON.stringify({ active, heat: heatOn, nov: novOn }));
    sentActive = active;
  }
  renderState();
}

function renderState() {
  const el = $("state");
  const s = lastStatus;
  if (!ws || ws.readyState !== WebSocket.OPEN) { el.textContent = "server unreachable, retrying…"; el.className = "down"; return; }
  if (!isActive()) {
    el.textContent = userPaused ? "paused" : document.visibilityState !== "visible" ? "paused (tab hidden)" :
      "paused (idle) - move the mouse to resume";
    el.className = "paused";
    return;
  }
  if (s && s.error) { el.textContent = s.error; el.className = "down"; return; }
  el.textContent = s && s.run ? `live · ${s.hz} Hz` : "waiting for the server…";
  el.className = s && s.run ? "live" : "paused";
}

function onStatus(s) {
  lastStatus = s;
  $("run").textContent = s.run || "-";
  $("envs").textContent = s.run ? `${s.envs} envs` : "";
  $("cost").textContent = s.run ? `server ${s.cpu.toFixed(1)}% of a core · ${s.viewers} viewer${s.viewers === 1 ? "" : "s"}` : "";
  renderState();
  if (!nov) renderNovInfo();
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/receive`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => reportActivity(true);
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) {
      if (e.data.byteLength >= 80 && new DataView(e.data).getUint32(0, true) === NOV_MAGIC) onNovelty(e.data);
      else onHeat(e.data);
      return;
    }
    const m = JSON.parse(e.data);
    if (m.t === "batch") onBatch(m, performance.now());
    else if (m.t === "status") onStatus(m);
  };
  ws.onclose = () => { sentActive = null; renderState(); setTimeout(connect, 2000); };
}

for (const ev of ["pointermove", "pointerdown", "wheel", "keydown", "touchstart"]) {
  window.addEventListener(ev, () => {
    const was = isActive();
    lastInput = performance.now();
    if (!was) reportActivity();
  }, { passive: true });
}
document.addEventListener("visibilitychange", () => reportActivity());
setInterval(() => reportActivity(), 5000);  // idle timeout

// ------------------------------------------------------------------ controls

function setVariant(v) {
  if (v === variant && meta) return;
  loadWorld(v).then(home).catch((e) => { $("state").textContent = e.message; $("state").className = "down"; });
}

function setLabels(on) {
  showLabels = on;
  store.set("mapviz.labels", on ? "1" : "0");
  $("labels").setAttribute("aria-pressed", on);
  for (const a of agents.values()) applyLook(a);
}

function setEnvs(on) {
  showEnvs = on;
  store.set("mapviz.envs", on ? "1" : "0");
  $("envs-btn").setAttribute("aria-pressed", on);
  agentLayer.visible = on;
}

function setHeat(on) {
  if (on && novOn) setNov(false);  // (one overlay at a time)
  heatOn = on;
  store.set("mapviz.heat", on ? "1" : "0");
  $("heat").setAttribute("aria-pressed", on);
  $("hud").classList.toggle("heat-on", on);
  if (heatLayer) heatLayer.visible = on;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ heat: on }));
  if (on) drawHeat();
  renderHeatInfo();
}

function setNov(on) {
  if (on && heatOn) setHeat(false);
  novOn = on;
  store.set("mapviz.nov", on ? "1" : "0");
  $("nov").setAttribute("aria-pressed", on);
  $("hud").classList.toggle("nov-on", on);
  novLayer.visible = on;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ nov: on }));
  if (on) drawNov();
  renderNovInfo();
}

function setNovMode(m) {
  novMode = m;
  store.set("mapviz.novMode", m);
  if (!novOn) setNov(true);
  else drawNov();
  renderNovInfo();
}

function setNovFrontier(on) {
  novFrontier = on;
  store.set("mapviz.novFrontier", on ? "1" : "0");
  if (novOn) drawNov();
  renderNovInfo();
}

function setHeatBattles(on) {
  heatBattles = on;
  store.set("mapviz.heatBattles", on ? "1" : "0");
  $("heat-battles").setAttribute("aria-pressed", on);
  if (heatOn) drawHeat();
}

function setPaused(on) {
  userPaused = on;
  $("pause").setAttribute("aria-pressed", on);
  $("pause").textContent = on ? "Resume" : "Pause";
  reportActivity();
}

$("v-story").onclick = () => setVariant("story");
$("v-full").onclick = () => setVariant("full");
$("labels").onclick = () => setLabels(!showLabels);
$("pause").onclick = () => setPaused(!userPaused);
$("envs-btn").onclick = () => setEnvs(!showEnvs);
$("heat").onclick = () => setHeat(!heatOn);
$("heat-battles").onclick = () => setHeatBattles(!heatBattles);
$("nov").onclick = () => setNov(!novOn);
$("nov-left").onclick = () => setNovMode("left");
$("nov-recent").onclick = () => setNovMode("recent");
$("nov-frontier").onclick = () => setNovFrontier(!novFrontier);
$("heat-reset").onclick = () => {
  if (ws && ws.readyState === WebSocket.OPEN && confirm("Clear this run's dwell heatmap on the server (for every viewer)?"))
    ws.send(JSON.stringify({ heat_reset: true }));
};
$("collapse").onclick = () => {
  const c = $("hud").classList.toggle("collapsed");
  $("collapse").textContent = c ? "+" : "–";
};
$("filter").addEventListener("input", (e) => {
  try { filter = e.target.value ? new RegExp(e.target.value) : null; e.target.style.borderColor = ""; }
  catch (err) { e.target.style.borderColor = "#ff7b72"; return; }
  for (const a of agents.values()) applyLook(a);
});
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") { if (e.key === "Escape") e.target.blur(); return; }
  const k = e.key.toLowerCase();
  if (k === "h") home();
  else if (k === "l") setLabels(!showLabels);
  else if (k === "p") setVariant(variant === "story" ? "full" : "story");
  else if (k === "m") setHeat(!heatOn);
  else if (k === "b") setHeatBattles(!heatBattles);
  else if (k === "n") setNov(!novOn);
  else if (k === "r") setNovMode(novMode === "left" ? "recent" : "left");
  else if (k === "o") setNovFrontier(!novFrontier);
  else if (k === "e") setEnvs(!showEnvs);
  else if (k === "f") { e.preventDefault(); $("filter").focus(); }
  else if (k === " ") { e.preventDefault(); setPaused(!userPaused); }
});

// ------------------------------------------------------------------ start

PIXI.Assets.load("world/player.png").then((tex) => {
  tex.baseTexture.scaleMode = PIXI.SCALE_MODES.NEAREST;
  const fr = (i) => new PIXI.Texture(tex.baseTexture, new PIXI.Rectangle(16 * i, 0, 16, 32));
  frames = { down: fr(0), up: fr(1), left: fr(2) };
});
{
  const g = new PIXI.Graphics();
  g.lineStyle(2, 0x000000, 0.85).beginFill(0xffffff).drawCircle(8, 8, 7).endFill();
  dotTex = app.renderer.generateTexture(g, { resolution: 2, region: new PIXI.Rectangle(0, 0, 16, 16) });
}
$("labels").setAttribute("aria-pressed", showLabels);
$("heat-bar").style.background = gradient(HEAT_STOPS);
setEnvs(showEnvs);
setHeatBattles(heatBattles);
setHeat(heatOn);
setNov(novOn);
loadWorld(variant === "full" ? "full" : "story").then(home)
  .catch((e) => { $("state").textContent = e.message; $("state").className = "down"; });
app.ticker.add(() => {
  const now = performance.now();
  updateChunks(now);
  animate(now);
  if (novOn) scaleNovMarks();
});
setInterval(refreshBadges, 1000);
setInterval(renderNovInfo, 5000);  // the checkpoint's age
connect();
