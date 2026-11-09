"""Build the side-by-side demo page (DESIGN.md §13) from demo_data.json and the frozen test files.

The page is one self-contained HTML file: the drawn problems, the four saved outputs of each, and a player that
reveals them at one constant token rate (not the recorded timing). Problem statements are joined from the frozen test
files, checked by SHA-256. No network access, no model, no API.

usage: python -I build_demo_page.py DEMO_DATA TEST_DIR OUT_HTML
  TEST_DIR holds tier1_variants.jsonl, tier2_aime2026.jsonl, tier2_hmmt_feb2026.jsonl and tier3_math500.jsonl
"""
import hashlib
import json
import sys
from pathlib import Path

TEST_FILES = {
    'tier1_variants.jsonl': 'c5e1eb3cea9ddcabdadd93952951cc1673968083c113662926c8c2245e16331b',
    'tier2_aime2026.jsonl': '67dbd42cc6327589a164a1f831b47690903af619124ea87e49e55863d27759da',
    'tier2_hmmt_feb2026.jsonl': '40f1a4430432c608f832d1e9cc2c66780e1c3f34d071c05da59cb138dea6fddc',
    'tier3_math500.jsonl': 'f9608716204039b465c396c7a3a753b2f3bb1976933237c5064d5a4e55cc385b'}
# DESIGN.md §10, accuracy at 32k with paired, clustered 95% intervals (conditions 1-4, tiers 1-3).
HEADLINE = {'1': ['0.567', '0.012', '0.505'], '2': ['0.564', '0.019', '0.516'],
            '3': ['0.948', '0.393', '0.907'], '4': ['0.928', '0.324', '0.886']}


def main(data_path, test_dir, out_html):
    data = json.loads(Path(data_path).read_text(encoding='utf-8'))
    problems = {}
    for name, digest in TEST_FILES.items():
        p = Path(test_dir) / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
        for line in p.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            problems[r['id']] = r['problem']
    for prob in data['problems']:
        prob['problem'] = problems[prob['id']]
    data['headline'] = HEADLINE
    blob = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    html = TEMPLATE.replace('__DATA__', blob)
    Path(out_html).write_text(html, encoding='utf-8')
    print('WROTE', out_html, round(len(html.encode()) / 2**20, 2), 'MB', len(data['problems']), 'problems')


TEMPLATE = r'''<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Four Conditions, Side by Side</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {
  --bg: #f6f5f1; --panel: #fffdf8; --ink: #1d2126; --muted: #5d636b; --line: #dedad0; --soft: #ece9e0;
  --c1: #6b7684; --c2: #b07a1e; --c3: #1f7f78; --c4: #6a52a8;
  --ok: #24794a; --bad: #b3402f; --warn: #9a6a00; --mark: #fff1c2; --after: #8b8f95;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15181c; --panel: #1c2026; --ink: #e6e3dc; --muted: #9aa0a8; --line: #2e333b; --soft: #252a31;
  --c1: #9aa6b4; --c2: #d9a54a; --c3: #4cb8ae; --c4: #a28ce0;
  --ok: #5cc28a; --bad: #ec7a68; --warn: #e0b04a; --mark: #3e3518; --after: #6f747b; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #15181c; --panel: #1c2026; --ink: #e6e3dc; --muted: #9aa0a8; --line: #2e333b; --soft: #252a31;
  --c1: #9aa6b4; --c2: #d9a54a; --c3: #4cb8ae; --c4: #a28ce0;
  --ok: #5cc28a; --bad: #ec7a68; --warn: #e0b04a; --mark: #3e3518; --after: #6f747b; color-scheme: dark }
* { box-sizing: border-box }
[hidden] { display: none !important }
body { background: var(--bg); color: var(--ink); font: 15px/1.5 "IBM Plex Sans", system-ui, sans-serif; }
.wrap { max-width: 1680px; margin: 0 auto; padding-block: 20px 40px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  display: grid; grid-template-columns: minmax(0, 1fr); gap: 16px }
.wrap > *, .foot > * { min-width: 0 }
h1 { font-size: 1.45rem; margin: 0; font-weight: 600; text-wrap: balance }
.sub { color: var(--muted); margin: 4px 0 0; max-width: 80ch; font-size: .92rem }
.label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); font-weight: 500 }
.picker { display: flex; flex-wrap: wrap; gap: 14px; align-items: center }
.tiergroup { display: flex; gap: 6px; align-items: center; flex-wrap: wrap }
.chip { font: 500 .82rem "IBM Plex Mono", ui-monospace, monospace; border: 1px solid var(--line); background: var(--panel);
  color: var(--ink); border-radius: 6px; padding: 5px 9px; cursor: pointer }
.chip[aria-pressed="true"] { background: var(--ink); color: var(--bg); border-color: var(--ink) }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; min-width: 0 }
.problem { display: grid; gap: 8px }
.ptext { font-size: 1rem; max-width: 110ch; overflow-wrap: anywhere }
.meta { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: .88rem; color: var(--muted) }
.meta b { color: var(--ink); font-weight: 500; font-family: "IBM Plex Mono", ui-monospace, monospace }
.controls { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center }
button.ctl { font: 500 .9rem "IBM Plex Sans", system-ui, sans-serif; border: 1px solid var(--ink); background: var(--ink);
  color: var(--bg); border-radius: 7px; padding: 6px 14px; cursor: pointer; min-width: 84px }
button.ghost { background: transparent; color: var(--ink); border-color: var(--line) }
.speeds { display: flex; gap: 4px }
.toggle { display: flex; gap: 6px; align-items: center; font-size: .9rem; cursor: pointer }
.clock { font: 500 .95rem "IBM Plex Mono", ui-monospace, monospace; margin-left: auto }
input[type=range] { width: 100%; accent-color: var(--ink) }
.grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px }
@media (max-width: 1180px) { .grid { grid-template-columns: repeat(2, minmax(0, 1fr)) } }
@media (max-width: 680px) { .grid { grid-template-columns: minmax(0, 1fr) } .clock { margin-left: 0 } }
.pane { background: var(--panel); border: 1px solid var(--line); border-top: 4px solid var(--cc); border-radius: 10px;
  display: grid; grid-template-rows: auto auto 1fr; min-width: 0; overflow: hidden }
.phead { padding: 10px 12px 4px; display: grid; gap: 2px }
.pname { font-weight: 600; color: var(--cc) }
.pdesc { font-size: .8rem; color: var(--muted) }
.pstat { padding: 4px 12px 8px; display: flex; flex-wrap: wrap; gap: 4px 10px; align-items: center; font-size: .82rem;
  border-bottom: 1px solid var(--line); min-height: 34px }
.tok { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--muted) }
.badge { font-weight: 600; border-radius: 5px; padding: 1px 7px; font-size: .78rem }
.b-ok { color: var(--ok); border: 1px solid var(--ok) } .b-bad { color: var(--bad); border: 1px solid var(--bad) }
.b-warn { color: var(--warn); border: 1px solid var(--warn) } .b-run { color: var(--muted); border: 1px dashed var(--line) }
.text { font: .86rem/1.55 "IBM Plex Sans", system-ui, sans-serif; overflow-wrap: anywhere;
  padding: 10px 12px; height: 56vh; min-height: 260px; overflow-y: auto; margin: 0 }
.text p, .ptext p { margin: 0 0 .6em }
.text p.tail { white-space: pre-line }
.text .after { color: var(--after) }
.dmath { display: block; overflow-x: auto; overflow-y: hidden; margin: .35em 0; text-align: center; scrollbar-width: thin }
.dmath mjx-container { margin: 0 !important }
.imath, .dmath { color: inherit }
.tag { display: block; font: 600 .7rem "IBM Plex Sans", system-ui, sans-serif; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); border-bottom: 1px solid var(--line); padding-bottom: 2px; margin: .2em 0 .6em }
.text .stopmark { display: block; background: var(--mark); color: var(--ink); font-family: "IBM Plex Sans", sans-serif;
  font-size: .8rem; padding: 4px 8px; margin: 6px 0; border-left: 3px solid var(--warn) }
.foot { display: grid; gap: 10px; font-size: .86rem; color: var(--muted) }
.tablewrap { overflow-x: auto }
table { border-collapse: collapse; font-size: .86rem; min-width: 440px }
th, td { text-align: left; padding: 4px 14px 4px 0; border-bottom: 1px solid var(--line) }
td.num { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--ink) }
</style>
<div class="wrap">
  <header>
    <h1>Four conditions, one problem</h1>
    <p class="sub">Saved outputs from the final evaluation of Qwen3.5-9B-Base and its three LoRA adapters (training seed 1,
      sample 0, same request seed for every model). The four texts are revealed at one constant token rate: the speed is
      for reading, not the recorded generation timing. Problems were drawn by a fixed rule before any output was read.</p>
  </header>
  <nav class="picker" id="picker" aria-label="Problems"></nav>
  <section class="card problem" id="problem"></section>
  <section class="card controls" aria-label="Playback">
    <button class="ctl" id="play">Play</button>
    <button class="ctl ghost" id="restart">Restart</button>
    <button class="ctl ghost" id="end">Show all</button>
    <div class="speeds" id="speeds"></div>
    <label class="toggle"><input type="checkbox" id="looprule"> Stop at the loop rule (§12)</label>
    <span class="clock" id="clock"></span>
    <input type="range" id="scrub" min="0" max="1000" value="0" aria-label="Token position">
  </section>
  <section class="grid" id="grid"></section>
  <section class="card foot">
    <div class="label">Accuracy at the 32k budget, all seeds and samples (final results)</div>
    <div class="tablewrap"><table id="headline"></table></div>
    <div>Tier 1: 89 rewritten MATH test variants. Tier 2: AIME and HMMT February 2026. Tier 3: MATH-500 (possibly
      contaminated). Conditions 2–4 must close their think block to be graded correct. The loop rule checks every
      1,000 characters whether the text has started repeating with a fixed period; its stop point here is computed
      offline on the saved text.</div>
  </section>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>window.MathJax = {startup: {typeset: false, pageReady() { if (window.__mathReady) window.__mathReady(); }},
  svg: {fontCache: 'local'}};</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('data').textContent);
  const COND = {
    1: ['Base model', 'no training, 4-shot prompt'],
    2: ['Human solutions', 'LoRA on MATH reference solutions'],
    3: ['Raw traces', 'LoRA on teacher reasoning traces'],
    4: ['Cleaned traces', 'LoRA on cleaned teacher traces']};
  const TIER = {tier1: 'Tier 1', tier2: 'Tier 2', tier3: 'Tier 3'};
  const SPEEDS = [[1, 250], [4, 1000], [16, 4000], [64, 16000]];
  const fmt = n => Math.round(n).toLocaleString('en-US');
  let cur = 0, clock = 0, rate = 1000, playing = false, last = 0, useRule = false, follow = {};
  const $ = id => document.getElementById(id);

  // problem picker, grouped by tier
  const picker = $('picker');
  for (const t of ['tier1', 'tier2', 'tier3']) {
    const g = document.createElement('div'); g.className = 'tiergroup';
    const l = document.createElement('span'); l.className = 'label'; l.textContent = TIER[t]; g.append(l);
    D.problems.forEach((p, i) => { if (p.tier !== t) return;
      const b = document.createElement('button'); b.className = 'chip'; b.dataset.i = i;
      b.textContent = p.id.replace(/^variant\//, '').replace(/^math500\/test\//, '').replace(/\.json$/, '');
      b.onclick = () => select(i); g.append(b); });
    picker.append(g);
  }
  // speed buttons
  for (const [m, r] of SPEEDS) { const b = document.createElement('button'); b.className = 'chip'; b.textContent = m + '×';
    b.title = fmt(r) + ' tokens per second'; b.onclick = () => { rate = r; markSpeed(); }; b.dataset.r = r; $('speeds').append(b); }
  const markSpeed = () => [...$('speeds').children].forEach(b => b.setAttribute('aria-pressed', +b.dataset.r === rate));
  markSpeed();
  // headline table
  const ht = $('headline');
  ht.innerHTML = '<tr><th>Condition</th><th>Tier 1</th><th>Tier 2</th><th>Tier 3</th></tr>';
  for (const c of ['1', '2', '3', '4']) { const tr = document.createElement('tr');
    const td = document.createElement('td'); td.textContent = c + ' · ' + COND[c][0]; tr.append(td);
    for (const v of D.headline[c]) { const x = document.createElement('td'); x.className = 'num'; x.textContent = v; tr.append(x); }
    ht.append(tr); }

  // Formatted text: blank lines start paragraphs, LaTeX is drawn by MathJax (SVG) once it scrolls into view, and the
  // think tags become labels. Model text only ever enters the page as text nodes or as TeX source for MathJax.
  const MATH = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)|\$[^$\n]{1,200}?\$)/;
  let mathReady = false; const pendingMath = new Set();
  const io = new IntersectionObserver(es => { for (const e of es) if (e.isIntersecting) typeset(e.target); },
                                      {rootMargin: '600px 0px'});
  function typeset(el) {
    if (el.dataset.done) return;
    if (!mathReady) { pendingMath.add(el); return; }
    el.dataset.done = '1'; io.unobserve(el); pendingMath.delete(el);
    MathJax.tex2svgPromise(el.dataset.tex, {display: el.classList.contains('dmath')})
      .then(svg => { el.textContent = ''; el.append(svg); }).catch(() => {});
  }
  window.__mathReady = () => {
    mathReady = true;
    try { document.head.append(MathJax.svgStylesheet()); } catch (e) {}
    for (const el of [...pendingMath]) if (el.isConnected) typeset(el);
  };
  function words(s, p) {  // **bold** and markdown headings; newlines inside a paragraph become line breaks
    s.split('\n').forEach((line, i) => {
      if (i) p.append(document.createElement('br'));
      const h = /^#{1,6}\s+/.exec(line); if (h) line = line.slice(h[0].length);
      const box = h ? p.appendChild(document.createElement('strong')) : p;
      line.split(/(\*\*[^*\n]+\*\*)/).forEach((b, j) => {
        if (j % 2) box.appendChild(document.createElement('strong')).textContent = b.slice(2, -2);
        else if (b) box.append(document.createTextNode(b));
      });
    });
  }
  function inline(s, p) {
    const bits = s.split(MATH), disp = b => b.startsWith('$$') || b.startsWith('\\[');
    bits.forEach((b, i) => {
      if (i % 2) {
        const d = disp(b), m = document.createElement('span');
        m.className = d ? 'dmath' : 'imath'; m.dataset.tex = b.startsWith('$') && !d ? b.slice(1, -1) : b.slice(2, -2);
        m.textContent = b; p.append(m); io.observe(m);
      } else {
        if (i > 0 && disp(bits[i - 1])) b = b.replace(/^[ \t]*\n/, '');
        if (i + 1 < bits.length && disp(bits[i + 1])) b = b.replace(/\n[ \t]*$/, '');
        if (b) words(b, p);
      }
    });
  }
  function blocks(s, parent) {
    for (const seg of s.split(/(<\/?think>)/)) {
      if (seg === '<think>' || seg === '</think>') {
        parent.appendChild(document.createElement('div')).className = 'tag';
        parent.lastChild.textContent = seg === '<think>' ? 'Reasoning' : 'Answer'; continue; }
      const t = seg.replace(/^\n+|\n+$/g, ''); if (!t.trim()) continue;
      const p = parent.appendChild(document.createElement('p')); inline(t, p);
    }
  }
  // A flow shows src[start:end]: finished paragraphs are formatted once, the paragraph being written is plain text.
  function flowSet(f, src, start, end, final) {
    if (f.src !== src || f.start !== start || end < f.end) {
      f.el.textContent = ''; f.src = src; f.start = start; f.end = f.done = start; f.tail = null; }
    if (end === f.end && !(final && f.tail)) return;
    const cut = final ? end : src.lastIndexOf('\n\n', end - 2);
    if (cut >= f.done) {
      if (f.tail) { f.tail.remove(); f.tail = null; }
      for (const s of src.slice(f.done, cut).split(/\n{2,}/)) blocks(s, f.el);
      f.done = final ? end : cut + 2;
    }
    const rest = src.slice(f.done, end).replace(/<\/?think>/g, '');
    if (rest.trim()) {
      if (!f.tail) { f.tail = f.el.appendChild(document.createElement('p')); f.tail.className = 'tail'; }
      f.tail.textContent = rest.replace(/^\n+/, '');
    } else if (f.tail) { f.tail.remove(); f.tail = null; }
    f.end = end;
  }

  // panes
  const grid = $('grid'), panes = {};
  for (const c of ['1', '2', '3', '4']) {
    const el = document.createElement('article'); el.className = 'pane'; el.style.setProperty('--cc', `var(--c${c})`);
    el.innerHTML = `<div class="phead"><span class="pname">${c} · ${COND[c][0]}</span><span class="pdesc">${COND[c][1]}</span></div>
      <div class="pstat"></div><div class="text" tabindex="0"><div></div><span class="stopmark" hidden></span><div class="after"></div></div>`;
    const text = el.querySelector('.text'), [fa, mark, fb] = text.children;
    text.addEventListener('scroll', () => { follow[c] = text.scrollTop + text.clientHeight >= text.scrollHeight - 30; });
    grid.append(el);
    panes[c] = {el, stat: el.querySelector('.pstat'), text, mark, a: {el: fa}, b: {el: fb}, shown: -1, mode: ''};
  }

  const P = () => D.problems[cur];
  function endTok(o) { return useRule && o.loop_stop ? o.loop_stop.tokens_est : o.tokens; }
  function maxTok() { return Math.max(...Object.values(P().outputs).map(endTok)); }

  function select(i) {
    cur = i; clock = 0; playing = false; $('play').textContent = 'Play';
    [...picker.querySelectorAll('.chip')].forEach(b => b.setAttribute('aria-pressed', +b.dataset.i === i));
    const p = P(), box = $('problem');
    box.innerHTML = '';
    const top = document.createElement('div'); top.className = 'meta';
    top.innerHTML = `<span class="label">${TIER[p.tier]}</span><span>${p.id}</span>`;
    const txt = document.createElement('div'); txt.className = 'ptext';
    for (const s of p.problem.split(/\n{2,}/)) blocks(s, txt);
    const m = document.createElement('div'); m.className = 'meta';
    const gold = document.createElement('span'); gold.innerHTML = 'Reference answer: <b></b>'; gold.querySelector('b').textContent = p.gold;
    m.append(gold);
    for (const c of ['1', '2', '3', '4']) { const a = p.accuracy_all_samples[c]; const s = document.createElement('span');
      s.innerHTML = `${COND[c][0]}: <b>${Math.round(a.mean * a.n)}/${a.n}</b> correct`; m.append(s); }
    box.append(top, txt, m);
    for (const c in panes) { panes[c].shown = -1; panes[c].mode = ''; follow[c] = true; panes[c].text.scrollTop = 0; }
    render();
  }

  function badge(cls, text) { return `<span class="badge ${cls}">${text}</span>`; }
  function render() {
    const p = P(), mx = maxTok();
    clock = Math.min(clock, mx);
    $('clock').textContent = `${fmt(clock)} / ${fmt(mx)} tokens`;
    $('scrub').value = mx ? Math.round(1000 * clock / mx) : 0;
    for (const c of ['1', '2', '3', '4']) {
      const o = p.outputs[c], pane = panes[c], end = endTok(o);
      const t = Math.min(clock, end), done = clock >= end;
      const L = o.loop_stop ? o.loop_stop.chars : null;
      let chars = t >= o.tokens ? o.chars : Math.floor(t * o.chars / Math.max(1, o.tokens));
      if (L !== null && t >= o.loop_stop.tokens_est) chars = Math.max(chars, L);  // the token estimate is proportional
      const mode = `${useRule}|${done}`;
      if (chars !== pane.shown || mode !== pane.mode) {
        const split = L !== null && chars >= L;
        pane.mark.hidden = !split;
        if (split) {
          pane.mark.textContent = `Loop rule fires here, about ${fmt(o.loop_stop.tokens_est)} tokens in`
            + (useRule ? '; generation stops.' : '; the output continued below.');
          flowSet(pane.a, o.text, 0, L, true);
          if (useRule) flowSet(pane.b, '', 0, 0, false); else flowSet(pane.b, o.text, L, chars, chars >= o.chars);
        } else { flowSet(pane.a, o.text, 0, chars, chars >= o.chars); flowSet(pane.b, '', 0, 0, false); }
        if (follow[c] !== false) pane.text.scrollTop = pane.text.scrollHeight;
        pane.shown = chars; pane.mode = mode;
        let s = `<span class="tok">${fmt(t)} / ${fmt(o.tokens)} tok</span>`;
        if (!done) s += badge('b-run', 'generating');
        else if (useRule && o.loop_stop) {
          s += badge('b-warn', 'stopped by loop rule');
          s += badge(o.loop_stop.correct_after ? 'b-ok' : 'b-bad', o.loop_stop.correct_after ? 'correct' : 'wrong');
          s += `<span class="tok">saves ${fmt(o.tokens - o.loop_stop.tokens_est)} tok</span>`;
        } else {
          const trunc = o.finish_reason === 'length';
          s += badge(trunc ? 'b-warn' : 'b-run', trunc ? 'hit 32k budget' : 'finished');
          if (c !== '1' && o.think_closed === false) s += badge('b-bad', 'think block open');
          s += badge(o.correct ? 'b-ok' : 'b-bad', o.correct ? 'correct' : 'wrong');
        }
        pane.stat.innerHTML = s;
      }
    }
  }

  function tick(ts) {
    if (!playing) return;
    const dt = last ? Math.min(0.25, (ts - last) / 1000) : 0; last = ts;
    clock += dt * rate;
    if (clock >= maxTok()) { clock = maxTok(); playing = false; $('play').textContent = 'Play'; }
    render();
    if (playing) requestAnimationFrame(tick);
  }
  $('play').onclick = () => { if (!playing && clock >= maxTok()) clock = 0;
    playing = !playing; $('play').textContent = playing ? 'Pause' : 'Play'; last = 0; if (playing) requestAnimationFrame(tick); };
  $('restart').onclick = () => { clock = 0; for (const c in panes) follow[c] = true; render(); };
  $('end').onclick = () => { playing = false; $('play').textContent = 'Play'; clock = maxTok(); render(); };
  $('scrub').oninput = e => { clock = maxTok() * e.target.value / 1000; render(); };
  $('looprule').onchange = e => { useRule = e.target.checked; for (const c in panes) panes[c].shown = -1; render(); };
  let start = 0;
  try { const h = location.hash.slice(1); const k = D.problems.findIndex((p, i) => 'p' + (i + 1) === h); if (k >= 0) start = k; } catch (e) {}
  select(start);
})();
</script>
'''

if __name__ == '__main__':
    main(*sys.argv[1:4])
