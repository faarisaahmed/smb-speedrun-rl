// Renders the page from data.json (made by make_site.py). The words live here; the numbers come from the run.

const FPS = 60.0988;

const HIGHLIGHTS = [
  { video: "pipe_shortcut", feature: true, tag: "1-1 · Shortcut",
    title: "It found the tool-assisted run's pipe shortcut",
    desc: "Shown only where the TAS's shortcut is (practice starts by the pipe, none of the TAS's inputs), it learned to drop into the coin room under 1-1 and come out by the end of the level, all the way from the start. Its record fell from 21.23 s to 20.00 s." },
  { video: "ghost_7-1", tag: "7-1 · 300 attempts",
    title: "Through the Hammer Bros",
    desc: "300 attempts at once. The hammers were invisible to it until they were added to its view; its record now reaches the flag 1.5 s before the TAS's." },
  { video: "ghost_6-2_wall", tag: "6-2 · Before",
    title: "The Buzzy Beetle shell trap",
    desc: "A shell bouncing between two pipes stopped every one of these 300 attempts. 6-2 sat at 0% for hundreds of millions of steps." },
  { video: "ghost_6-2", tag: "6-2 · After",
    title: "Getting through",
    desc: "460M steps later the pack gets past the trap and 5 of 300 reach the flag, with no scripted fix. It now clears about 1 in 5." },
  { video: "ghost_8-1", tag: "8-1 · 300 attempts",
    title: "The longest outdoor level",
    desc: "135 of these 300 attempts clear 8-1's 16+ screens, up from 64 of 300 at 482M steps." },
  { video: "race_8-1_vs_tas_small", tag: "8-1 · Fair race",
    title: "Beating the power-up-less TAS by 11.6 s",
    desc: "From the TAS's own save state, both small Mario, timed to the next level's start. The white ghost is SysDataSoft's tool-assisted run." },
  { video: "race_6-1_vs_tas", tag: "6-1 · Never trained on",
    title: "Clearing a level it has never seen",
    desc: "6-1 was held out of training entirely. Dropped into it from the TAS's save state, it still found its way to the flag." },
  { video: "progression_1-1", tag: "1-1 · Every record",
    title: "Every 1-1 record, in order",
    desc: "All of its 1-1 records back to back, from the first clumsy clear at 33.8 s down to the shortcut route at 20.00 s." },
  { video: "secret_pipe", tag: "1-1 · Discovery",
    title: "Finding the bonus pipe on its own",
    desc: "Early in training, before any hints, it found the pipe down to 1-1's bonus room by itself." },
  { video: "ghost_1-3_jump", tag: "1-3 · Breakthrough",
    title: "Learning the tall-tree jump",
    desc: "1-3 needs a full-height running jump from a 3-step window. For 75M steps it never once made it; within minutes of the explorer and self-imitation going in, it made the jump 97% of the time." },
];

const GHOSTS = [
  { video: "ghost_1-1_early", tag: "1-1 · Early", title: "1-1, early in training", desc: "Attempts piling up at the first pipes and Goombas." },
  { video: "ghost_1-1_late", tag: "1-1 · Later", title: "1-1, later", desc: "The same level once it has it down: one tight pack." },
  { video: "ghost_3-2", tag: "3-2 · 146M steps", title: "3-2 at 146M steps", desc: "300 attempts soon after it started on every outdoor level." },
  { video: "ghost_4-3", tag: "4-3 · 225M steps", title: "4-3 at 225M steps", desc: "Moving platforms and the long drops beneath them." },
  { video: "ghost_8-1_early", tag: "8-1 · 482M steps", title: "8-1 at 482M steps", desc: "64 of 300 attempts clear." },
  { video: "ghost_8-1", tag: "8-1 · 1.26B steps", title: "8-1 at 1.26B steps", desc: "135 of 300 attempts clear." },
];

// The project's problems and fixes, oldest first.
const TIMELINE = [
  { when: "Before", title: "A first attempt on a remake",
    problem: "An earlier version of this project rebuilt the game from scratch in pygame and trained on the copy. It got about 21% of the way through 1-1.",
    fix: "Start over on the real cartridge, in an emulator, with the screen read straight from the console's memory." },
  { when: "252k steps", title: "First clear of 1-1", big: true, video: "first_clear_1-1",
    fix: "33.8 s, from random button presses. By 12M steps it was down to 21.5 s." },
  { when: "1-3 · stuck for 75M steps", title: "The tall-tree jump", big: true, video: "first_clear_1-3",
    problem: "1-3 needs a full-height running jump onto a tall tree from a 3-step take-off window, holding A for about 15 steps. Its chance of jumping in that window was about zero, and in the air it let go of A at random, so it never saw a single success to learn from.",
    fix: "Go-Explore-style restart points, a background explorer that tries random held-button sequences at walls until one gets through, and self-imitation of those finds. Within minutes its chance of making the jump went from about 0 to 97%." },
  { when: "Reward", title: "Shortcuts were being punished",
    problem: "Taking 1-1's bonus pipe resets Mario's x position inside the bonus room, so the run looked stuck and was cut short. Pay-per-pixel reward also paid less for routes that skip ahead.",
    fix: "Pay progress on the furthest point reached, judge being stuck per area, and give bonus rooms their own restart points. Any route to the same place now earns the same." },
  { when: "Reward", title: "A bonus room 1,800 pixels ahead",
    problem: "6-2's bonus room sits at an x position far beyond its entrance pipe, so going in looked like a huge leap forward.",
    fix: "Progress is only paid in a level's main area." },
  { when: "Practice", title: "A bad practice start wrecked 8-1",
    problem: "The explorer saved a run through an awkward state as a practice start, and 8-1's clear rate fell from 34% to 3%.",
    fix: "A spot only counts as a wall if runs from the level's start also fail there." },
  { when: "Reward", title: "Waiting out the piranha plants",
    problem: "Standing still until the attempt timed out cost less than risking a piranha plant, so it learned to wait.",
    fix: "Stalling now ends the attempt with the same penalty as dying." },
  { when: "Observation", title: "Pressing a button that's already held",
    problem: "In NES Mario a held A can't start a new jump. The agent couldn't tell whether A was still held, and kept bouncing off 6-2's pipes.",
    fix: "It now sees which buttons it held on the last step." },
  { when: "Observation", title: "Invisible hammers",
    problem: "7-1's two worst death spots were both Hammer Bros pairs. The hammers live in a different part of memory from enemies, so they weren't in its view at all.",
    fix: "Hammers are now drawn into its view as hostile objects." },
  { when: "6-2 · 0% for ~900M steps", title: "The Buzzy Beetle shell wall", big: true, video: "ghost_6-2_wall",
    problem: "A Buzzy Beetle walks between two close pipes; stomp it and the shell ricochets between them. Nothing was hidden from the agent: it was just a hard spot, and the level is hard all the way through.",
    fix: "No scripted fix, by design: more practice at each section. 0 of 300 attempts at 430M, 5 of 300 at 890M, about 1 in 5 now." },
  { when: "Benchmark", title: "A ghost to race",
    problem: "The first ghost traced a human world-record video by matching sprites frame by frame, drawn as Luigi. Too jittery to trust.",
    fix: "Replay HappyLee & Mars608's TAS input file in the same emulator and read Mario's position straight from memory, frame-exact." },
  { when: "1.37B steps", title: "Comparing fairly",
    problem: "Beating the TAS to the flag isn't beating the TAS: it plays as big Mario, and a real run is timed by when the next level starts, which moves in 21-frame steps.",
    fix: "Race from the TAS's own save state at each level start, clocked to the next level. Against the power-up-less TAS, which stays small like the agent, it wins 1-1, 4-1 and 8-1 by 3.9 to 11.6 seconds. The warpless world record still wins every level that counts, by 0.68 s on 8-3 at its closest." },
  { when: "Bug", title: "A 9.80 s 6-2 that wasn't",
    problem: "The TAS seemed to finish 6-2 in 9.80 s. Climbing a vine sets the same memory flag as sliding down the flagpole.",
    fix: "A flagpole check that tells the two apart. The real time is 24.18 s." },
  { when: "Decision", title: "An input no controller can make",
    problem: "The TAS skips most of 6-2 by holding Left and Right at the same time on a vine. A real D-pad can't press both, and none of the agent's 12 actions do. A brute-force search over 14,641 legal input sequences found no other way.",
    fix: "Left out on purpose: every run here can be played on a real controller." },
  { when: "Late in training", title: "Hints: shown where, never how", big: true, video: "record_1-1",
    problem: "Some shortcuts are nearly impossible to stumble on, like 1-1's coin-room pipe route.",
    fix: "Practice starts just before each shortcut the TAS takes, as small Mario, with none of its inputs. It learned the 1-1 route from the start, 21.23 s → 20.00 s." },
  { when: "1.2B – 1.64B steps", title: "A plateau",
    problem: "The average clear rate had stalled around 55%.",
    fix: "Halving the learning rate took it to about 67%." },
  { when: "Throughout", title: "Collapses that heal",
    problem: "Levels sometimes collapse: 4-3 fell to 0% twice, and 8-1 to 4% near the end.",
    fix: "Left alone, 4-3 recovered both times. Records are kept no matter what, so a collapse never loses a best run." },
  { when: "2.38B steps", title: "Where it stopped", big: true, video: "race_6-1_vs_tas",
    fix: "All 17 outdoor training levels cleared, 281 records set, and 3 fair wins over the power-up-less TAS. On 5-3 and 6-1, two levels it never trained on, it reached the flag from the TAS's save state too." },
];

const RACES = {
  race_small: {
    blurb: "SysDataSoft's power-up-less TAS stays small the whole game, like the agent, so this is the like-for-like race. It takes warp zones, so the outdoor levels it plays are 1-1, 4-1 and 8-1 to 8-3.",
  },
  race: {
    blurb: "HappyLee & Mars608's warpless TAS, the current standard. It picks up a mushroom in 1-3, so that race doesn't count. From 2-1 on, the agent starts from the TAS's own save state, so both play as big Mario.",
  },
};

// ---------- helpers ----------

const $ = (s, el = document) => el.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const kid of kids) if (kid != null) n.append(kid);
  return n;
};
const svg = (tag, attrs = {}) => {
  const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
};
const steps = (t) => t >= 1e9 ? `${(t / 1e9).toFixed(t >= 1e10 ? 0 : 2)}B` : t >= 1e6 ? `${Math.round(t / 1e6)}M` : `${Math.round(t / 1e3)}k`;
const secs = (s) => s == null ? "—" : `${s.toFixed(2)} s`;
const mmss = (d) => `${Math.floor(d / 60)}:${String(Math.round(d % 60)).padStart(2, "0")}`;
const levelOrder = (a, b) => a.localeCompare(b, undefined, { numeric: true });

let DATA;

// ---------- player ----------

const player = $("#player");
const playerVideo = $("#player-video");
function openVideo(key, title, desc) {
  const v = DATA.videos[key];
  if (!v) return;
  $("#player-title").textContent = title;
  $("#player-desc").textContent = desc || "";
  playerVideo.poster = v.poster;
  playerVideo.src = v.src;
  player.showModal();
  playerVideo.play().catch(() => {});
}
function closeVideo() {
  playerVideo.pause();
  playerVideo.removeAttribute("src");
  playerVideo.load();
  player.close();
}
$("#player-close").addEventListener("click", closeVideo);
player.addEventListener("click", (e) => { if (e.target === player) closeVideo(); });
player.addEventListener("cancel", (e) => { e.preventDefault(); closeVideo(); });

const canHover = matchMedia("(hover: hover) and (pointer: fine)").matches;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

function card(item) {
  const v = DATA.videos[item.video];
  if (!v) return null;
  const thumb = el("div", { class: "thumb" },
    el("img", { src: v.poster, alt: "", loading: "lazy" }),
    el("span", { class: "play", "aria-hidden": "true" }),
    el("span", { class: "dur" }, mmss(v.duration)));
  const c = el("button", { class: "card" + (item.feature ? " feature" : ""), type: "button",
    onclick: () => openVideo(item.video, item.title, item.desc) },
    thumb,
    el("div", { class: "card-body" },
      el("p", { class: "card-tag" }, item.tag),
      el("h3", {}, item.title),
      el("p", { class: "desc" }, item.desc)));
  if (canHover && !reduceMotion) {
    let preview;
    c.addEventListener("mouseenter", () => {
      if (!preview) {
        preview = el("video", { muted: "", loop: "", playsinline: "", preload: "auto" });
        preview.muted = true;
        preview.src = v.src;
        thumb.insertBefore(preview, thumb.children[1]);
      }
      preview.play().then(() => c.classList.add("previewing")).catch(() => {});
    });
    c.addEventListener("mouseleave", () => { if (preview) { preview.pause(); c.classList.remove("previewing"); } });
  }
  return c;
}

// ---------- sections ----------

function heroStats() {
  const levels = DATA.levels.filter((l) => DATA.best[l] != null).length;
  const records = Object.values(DATA.records).reduce((n, r) => n + r.length, 0);
  const smallWins = Object.values(DATA.race_small).filter((r) => r.agent_next != null && r.tas_next != null && r.agent_next < r.tas_next).length;
  const stats = [
    [`${levels} / ${DATA.levels.length}`, "outdoor levels cleared"],
    [steps(DATA.timesteps), "steps of practice"],
    [String(records), "records set along the way"],
    [String(smallWins), "fair wins vs the power-up-less TAS"],
  ];
  const dl = $("#hero-stats");
  for (const [v, k] of stats) dl.append(el("div", {}, el("dd", {}, v), el("dt", {}, k)));
  document.querySelectorAll('[data-stat="steps"]').forEach((n) => {
    n.textContent = `${(DATA.timesteps / 1e9).toFixed(1)} billion`;
  });
  const hero = $("#hero-video"), hv = DATA.videos["ghost_7-1"];
  if (hv && !reduceMotion) { hero.src = hv.src; hero.play().catch(() => {}); }
}

function resultsTable() {
  const tbody = $("#results-table tbody");
  for (const level of [...DATA.levels].sort(levelOrder)) {
    const best = DATA.best[level];
    const tas = DATA.race[level] && DATA.race[level].tas_flag;
    const d = best != null && tas != null ? best - tas : null;
    const rate = DATA.success[level] || 0;
    const key = `record_${level}`;
    tbody.append(el("tr", {},
      el("td", { class: "level" }, level),
      el("td", { class: "num" }, secs(best)),
      el("td", { class: "num" }, secs(tas)),
      el("td", { class: "num " + (d == null ? "" : d < 0 ? "delta-win" : "delta-loss") },
        d == null ? "—" : `${d < 0 ? "−" : "+"}${Math.abs(d).toFixed(2)} s`),
      el("td", {}, el("div", { class: "rate" },
        el("div", { class: "rate-bar" }, el("span", { style: `width:${(rate * 100).toFixed(1)}%` })),
        el("span", { class: "rate-val" }, `${Math.round(rate * 100)}%`))),
      el("td", {}, DATA.videos[key] ? el("button", { class: "watch", type: "button",
        onclick: () => openVideo(key, `${level} record · ${secs(best)}`,
          "The agent's fastest run from the level start, with the TAS ghost in white.") }, "Watch") : "")));
  }
}

function raceTable(which) {
  $("#race-blurb").textContent = RACES[which].blurb;
  const tbody = $("#race-table tbody");
  tbody.replaceChildren();
  const rows = DATA[which];
  for (const level of Object.keys(rows).sort(levelOrder)) {
    const r = rows[level];
    let verdict;
    if (which === "race" && level === "1-3") verdict = el("span", { class: "pill none" }, "not comparable: the TAS detours for a mushroom");
    else if (r.agent_next == null) verdict = el("span", { class: "pill none" }, "no timed clear");
    else if (r.tas_next == null) verdict = el("span", { class: "pill none" }, "not timed");
    else {
      const d = r.agent_next - r.tas_next, frames = Math.round(Math.abs(d) * FPS);
      verdict = d < 0
        ? el("span", { class: "pill win" }, `Agent by ${Math.abs(d).toFixed(2)} s (${frames} frames)`)
        : el("span", { class: "pill loss" }, `TAS by ${d.toFixed(2)} s (${frames} frames)`);
    }
    const tasName = which === "race_small" ? "tas_small" : "tas";
    const key = `race_${level}_vs_${tasName}`;
    tbody.append(el("tr", {},
      el("td", { class: "level" }, level, DATA.holdout.includes(level) ? el("span", { class: "holdout" }, "never trained on") : null),
      el("td", { class: "num" }, secs(r.tas_next)),
      el("td", { class: "num" }, secs(r.agent_next)),
      el("td", {}, verdict),
      el("td", {}, DATA.videos[key] ? el("button", { class: "watch", type: "button",
        onclick: () => openVideo(key, `${level} · race vs the ${which === "race_small" ? "power-up-less" : "warpless"} TAS`,
          "Both start from the TAS's save state at the level start. The white ghost is the TAS; times run to the next level's start.") }, "Watch") : "")));
  }
}

function raceTabs() {
  const tabs = document.querySelectorAll("#race-tabs button");
  tabs.forEach((b) => b.addEventListener("click", () => {
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t === b)));
    raceTable(b.dataset.race);
  }));
  raceTable("race_small");
}

// ---------- charts ----------

function chartFrame(container, height) {
  const w = container.clientWidth, h = height;
  const m = { top: 12, right: 16, bottom: 30, left: 48 };
  const root = svg("svg", { viewBox: `0 0 ${w} ${h}`, role: "img" });
  const defs = svg("defs");
  defs.innerHTML = '<linearGradient id="gold-fade" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#f2b33d" stop-opacity="0.22"/><stop offset="1" stop-color="#f2b33d" stop-opacity="0"/></linearGradient>';
  root.append(defs);
  container.replaceChildren(root);
  const tip = el("div", { class: "tooltip" });
  container.append(tip);
  return { root, tip, w, h, m, iw: w - m.left - m.right, ih: h - m.top - m.bottom };
}

function placeTip(f, tip, x, y, html) {
  tip.innerHTML = html;
  tip.classList.add("on");
  const tw = tip.offsetWidth;
  const left = Math.min(Math.max(x + 14, 0), f.w - tw);
  tip.style.left = `${x + 14 + tw > f.w ? x - tw - 14 : left}px`;
  tip.style.top = `${Math.max(y - 40, 0)}px`;
}

function curveChart() {
  const box = $("#curve-chart");
  const pts = DATA.curve.map((c) => {
    const vals = Object.values(c.rates);
    return { t: c.t, v: vals.reduce((a, b) => a + b, 0) / vals.length, n: vals.length };
  });
  const f = chartFrame(box, box.clientHeight);
  const tMax = pts[pts.length - 1].t;
  const x = (t) => f.m.left + (t / tMax) * f.iw;
  const y = (v) => f.m.top + (1 - v) * f.ih;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    f.root.append(svg("line", { class: v === 0 ? "axis-line" : "grid-line", x1: f.m.left, x2: f.w - f.m.right, y1: y(v), y2: y(v) }));
    const t = svg("text", { class: "tick", x: f.m.left - 10, y: y(v) + 4, "text-anchor": "end" });
    t.textContent = `${v * 100}%`;
    f.root.append(t);
  }
  const tickStep = tMax > 2e9 ? 5e8 : 2.5e8;
  for (let t = 0; t <= tMax; t += tickStep) {
    const tx = svg("text", { class: "tick", x: x(t), y: f.h - 8, "text-anchor": "middle" });
    tx.textContent = t === 0 ? "0" : steps(t);
    f.root.append(tx);
  }
  const refs = [];
  const allLevels = DATA.curve.find((c) => c.stage === 2);
  if (allLevels) refs.push([allLevels.t, "all 17 levels"]);
  refs.push([1.643e9, "learning rate halved"]);
  for (const [t, label] of refs) {
    if (t > tMax) continue;
    f.root.append(svg("line", { class: "ref", x1: x(t), x2: x(t), y1: f.m.top, y2: f.m.top + f.ih }));
    const lx = svg("text", { class: "ref-label", x: x(t) + 6, y: f.m.top + 12 });
    lx.textContent = label;
    f.root.append(lx);
  }
  const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
  f.root.append(svg("path", { class: "area", d: `${d}L${x(tMax)},${y(0)}L${x(0)},${y(0)}Z` }));
  f.root.append(svg("path", { class: "series", d }));
  f.root.setAttribute("aria-label", `Average clear rate rising to ${Math.round(pts[pts.length - 1].v * 100)}% over ${steps(tMax)} steps`);

  const cross = svg("line", { class: "cross", y1: f.m.top, y2: f.m.top + f.ih, opacity: 0 });
  const dot = svg("circle", { class: "hover-dot", r: 4.5, opacity: 0 });
  f.root.append(cross, dot);
  const hit = svg("rect", { x: f.m.left, y: f.m.top, width: f.iw, height: f.ih, fill: "transparent" });
  f.root.append(hit);
  const move = (e) => {
    const r = f.root.getBoundingClientRect();
    const t = ((e.clientX - r.left) - f.m.left) / f.iw * tMax;
    let best = pts[0];
    for (const p of pts) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    cross.setAttribute("x1", x(best.t)); cross.setAttribute("x2", x(best.t)); cross.setAttribute("opacity", 1);
    dot.setAttribute("cx", x(best.t)); dot.setAttribute("cy", y(best.v)); dot.setAttribute("opacity", 1);
    placeTip(f, f.tip, x(best.t), y(best.v),
      `<b>${Math.round(best.v * 100)}%</b> of attempts clear<br><span class="t-muted">${steps(best.t)} steps · ${best.n} level${best.n > 1 ? "s" : ""}</span>`);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerleave", () => { cross.setAttribute("opacity", 0); dot.setAttribute("opacity", 0); f.tip.classList.remove("on"); });
}

let recordLevel = "1-1";
function recordChart() {
  const box = $("#record-chart");
  const recs = DATA.records[recordLevel] || [];
  if (!recs.length) return;
  const f = chartFrame(box, box.clientHeight);
  const tas = DATA.race[recordLevel] && DATA.race[recordLevel].tas_flag;
  const tMin = 1e5, tMax = Math.max(DATA.timesteps, recs[recs.length - 1][0]);
  const lx = (t) => Math.log10(Math.max(t, tMin));
  const x = (t) => f.m.left + (lx(t) - lx(tMin)) / (lx(tMax) - lx(tMin)) * f.iw;
  const times = recs.map((r) => r[1]).concat(tas ? [tas] : []);
  let lo = Math.min(...times), hi = Math.max(...times);
  const pad = (hi - lo) * 0.08 || 1;
  lo = Math.floor(lo - pad); hi = Math.ceil(hi + pad);
  const y = (s) => f.m.top + (hi - s) / (hi - lo) * f.ih;
  const span = hi - lo, stepS = span > 20 ? 5 : span > 8 ? 2 : 1;
  for (let s = Math.ceil(lo / stepS) * stepS; s <= hi; s += stepS) {
    f.root.append(svg("line", { class: "grid-line", x1: f.m.left, x2: f.w - f.m.right, y1: y(s), y2: y(s) }));
    const t = svg("text", { class: "tick", x: f.m.left - 10, y: y(s) + 4, "text-anchor": "end" });
    t.textContent = `${s} s`;
    f.root.append(t);
  }
  f.root.append(svg("line", { class: "axis-line", x1: f.m.left, x2: f.w - f.m.right, y1: f.m.top + f.ih, y2: f.m.top + f.ih }));
  for (const t of [1e5, 1e6, 1e7, 1e8, 1e9]) {
    if (t > tMax) continue;
    const tx = svg("text", { class: "tick", x: x(t), y: f.h - 8, "text-anchor": "middle" });
    tx.textContent = steps(t);
    f.root.append(tx);
  }
  if (tas) {
    f.root.append(svg("line", { class: "tas-line", x1: f.m.left, x2: f.w - f.m.right, y1: y(tas), y2: y(tas) }));
    const tl = svg("text", { class: "tas-label", x: f.m.left + 8, y: y(tas) + 16 });
    tl.textContent = `TAS flag time ${tas.toFixed(2)} s`;
    f.root.append(tl);
  }
  let d = `M${x(recs[0][0]).toFixed(1)},${y(recs[0][1]).toFixed(1)}`;
  for (let i = 1; i < recs.length; i++) d += `H${x(recs[i][0]).toFixed(1)}V${y(recs[i][1]).toFixed(1)}`;
  d += `H${x(tMax).toFixed(1)}`;
  f.root.append(svg("path", { class: "series", d }));
  const last = recs[recs.length - 1];
  for (const r of [recs[0], last]) f.root.append(svg("circle", { class: "dot", cx: x(r[0]), cy: y(r[1]), r: 4.5 }));
  const endLabel = svg("text", { class: "tas-label", x: x(last[0]) - 8, y: y(last[1]) - 10, "text-anchor": "end" });
  endLabel.textContent = `${last[1].toFixed(2)} s`;
  f.root.append(endLabel);
  f.root.setAttribute("aria-label", `${recordLevel}: ${recs.length} records, from ${recs[0][1]} s to ${last[1]} s`);

  const dot = svg("circle", { class: "hover-dot", r: 5, opacity: 0 });
  f.root.append(dot);
  const hit = svg("rect", { x: f.m.left, y: f.m.top, width: f.iw, height: f.ih, fill: "transparent" });
  f.root.append(hit);
  hit.addEventListener("pointermove", (e) => {
    const r = f.root.getBoundingClientRect(), px = e.clientX - r.left;
    let best = recs[0];
    for (const rec of recs) if (Math.abs(x(rec[0]) - px) < Math.abs(x(best[0]) - px)) best = rec;
    dot.setAttribute("cx", x(best[0])); dot.setAttribute("cy", y(best[1])); dot.setAttribute("opacity", 1);
    const i = recs.indexOf(best);
    placeTip(f, f.tip, x(best[0]), y(best[1]),
      `<b>${best[1].toFixed(2)} s</b> · record ${i + 1} of ${recs.length}<br><span class="t-muted">at ${steps(best[0])} steps</span>`);
  });
  hit.addEventListener("pointerleave", () => { dot.setAttribute("opacity", 0); f.tip.classList.remove("on"); });
}

function recordChips() {
  const row = $("#record-levels");
  for (const level of Object.keys(DATA.records).sort(levelOrder)) {
    row.append(el("button", { class: "chip", role: "tab", type: "button", "aria-selected": String(level === recordLevel),
      onclick: (e) => {
        recordLevel = level;
        row.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-selected", String(c === e.currentTarget)));
        recordChart();
      } }, level));
  }
}

function timeline() {
  const ol = $("#timeline");
  for (const t of TIMELINE) {
    const li = el("li", { class: t.big ? "big" : "" },
      el("p", { class: "tl-when" }, t.when),
      el("h3", {}, t.title));
    if (t.problem) li.append(el("p", { class: "tl-problem" }, el("span", { class: "tl-label" }, "Problem"), t.problem));
    if (t.fix) li.append(el("p", { class: "tl-fix" }, el("span", { class: "tl-label" }, t.problem ? "Fix" : "Result"), t.fix));
    if (t.video && DATA.videos[t.video]) {
      li.append(el("div", { class: "tl-watch" }, el("button", { class: "watch", type: "button",
        onclick: () => openVideo(t.video, t.title, t.fix) }, "Watch")));
    }
    ol.append(li);
  }
}

fetch("data.json").then((r) => r.json()).then((data) => {
  DATA = data;
  heroStats();
  $("#highlight-cards").append(...HIGHLIGHTS.map(card).filter(Boolean));
  $("#ghost-cards").append(...GHOSTS.map(card).filter(Boolean));
  resultsTable();
  raceTabs();
  curveChart();
  recordChips();
  recordChart();
  timeline();
  let resize;
  addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => { curveChart(); recordChart(); }, 150); });
});
