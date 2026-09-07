/* ═══════════════════════════════════════════════════════════════
   FishONet — Specimen Ledger
   ═══════════════════════════════════════════════════════════════ */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ══ 1. TAXONOMY CONSTELLATION — hero-only WebGL, motivated: it IS
   the embedding space the whole page is about, not decoration ══ */
(function initConstellation() {
  const canvas = $('#hero-canvas');
  const hero = $('.hero');
  if (reduce || !canvas || !hero || typeof THREE === 'undefined') return;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, hero.clientWidth / hero.clientHeight, 0.1, 100);
  camera.position.set(0, 0, 14);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(hero.clientWidth, hero.clientHeight);

  const N = 180;
  const positions = new Float32Array(N * 3);
  const colors = new Float32Array(N * 3);
  const paper = new THREE.Color(0xf3efe4);
  const accent = new THREE.Color(0xe35b23);

  for (let i = 0; i < N; i++) {
    const r = 4 + Math.random() * 2.5;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta) + 5.5;
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6;
    positions[i * 3 + 2] = r * Math.cos(phi) * 0.7 - 4;

    const c = Math.random() > 0.93 ? accent : paper;
    colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
  }

  const pointGeo = new THREE.BufferGeometry();
  pointGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  pointGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  const points = new THREE.Points(pointGeo, new THREE.PointsMaterial({
    size: 0.06, vertexColors: true, transparent: true, opacity: 0.55, sizeAttenuation: true
  }));
  scene.add(points);

  // sparse nearest-neighbor edges: reads as a taxonomy graph, not confetti
  const linePositions = [];
  for (let i = 0; i < N; i += 3) {
    let best = -1, bestD = Infinity;
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = positions[i * 3] - positions[j * 3];
      const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
      const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) { bestD = d; best = j; }
    }
    if (best >= 0) {
      linePositions.push(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
      linePositions.push(positions[best * 3], positions[best * 3 + 1], positions[best * 3 + 2]);
    }
  }
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(linePositions), 3));
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({
    color: 0xf3efe4, transparent: true, opacity: 0.08
  }));
  scene.add(lines);

  const group = new THREE.Group();
  group.add(points, lines);
  scene.add(group);

  let mx = 0, my = 0;
  window.addEventListener('mousemove', e => {
    mx = (e.clientX / window.innerWidth - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  const ro = new ResizeObserver(() => {
    camera.aspect = hero.clientWidth / hero.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(hero.clientWidth, hero.clientHeight);
  });
  ro.observe(hero);

  const clock = new THREE.Clock();
  let running = true;
  function tick() {
    if (!running) return;
    requestAnimationFrame(tick);
    const dt = clock.getDelta();
    group.rotation.y += dt * 0.05;
    group.rotation.x += (my * 0.15 - group.rotation.x) * 0.4 * dt * 4;
    camera.position.x += (mx * 1.2 - camera.position.x) * 0.4 * dt * 4;
    camera.lookAt(0, 0, -2);
    renderer.render(scene, camera);
  }

  // pause off tab and once the hero scrolls out of view; both are real cost, not micro-opt theater
  new IntersectionObserver(entries => {
    running = entries[0].isIntersecting && !document.hidden;
    if (running) tick();
  }, { threshold: 0 }).observe(hero);
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden && hero.getBoundingClientRect().bottom > 0;
    if (running) tick();
  });

  tick();
})();

/* ══ 2. SCROLL REVEAL — plain IntersectionObserver, no scroll listener ══ */
(function initReveal() {
  const els = $$('.reveal, .reveal-left, .reveal-right, .reveal-scale, .stagger');
  if (reduce) { els.forEach(e => e.classList.add('in')); return; }
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    });
  }, { threshold: 0.12 });
  els.forEach(e => io.observe(e));
})();

/* ══ 2B. PRELOADER — a real delay while fonts/data settle, not theater ══ */
function hidePreloader() {
  const p = $('#preloader');
  if (!p) return;
  p.classList.add('done');
  setTimeout(() => p.remove(), 700);
}

/* replay a .stagger container's reveal right after JS injects its children,
   since the page-load IntersectionObserver may have already fired on the
   (then-empty) container before the data arrived */
function revealNow(el) {
  if (!el) return;
  if (reduce) { el.classList.add('in'); return; }
  el.classList.remove('in');
  void el.offsetWidth;
  requestAnimationFrame(() => el.classList.add('in'));
}

/* ══ 3. COUNT-UP ══════════════════════════════════════════════ */
function countUp(el, to, dp = 0) {
  if (reduce) { el.textContent = to.toFixed(dp); return; }
  const t0 = performance.now(), dur = 1300;
  const tick = t => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4);
    el.textContent = (to * e).toFixed(dp);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function watchCount(el, to, dp = 0) {
  if (!el) return;
  new IntersectionObserver((entries, obs) => {
    if (entries[0].isIntersecting) { countUp(el, to, dp); obs.disconnect(); }
  }, { threshold: 0.4 }).observe(el);
}

// the real numbers are already in the HTML for anything that does not run
// JS; the count-up only replays them once the element is in view
watchCount($('#hero-score'), 53.761, 3);
watchCount($('#res-final'), 53.761, 3);
watchCount($('#res-seen'), 77.00, 2);
watchCount($('#res-novel'), 23.77, 2);

/* ══ 4. PROGRESSION BARS ══════════════════════════════════════ */
(function initBars() {
  const wrap = $('.prog-list');
  if (!wrap) return;
  new IntersectionObserver((entries, obs) => {
    if (!entries[0].isIntersecting) return;
    $$('.bar', wrap).forEach((b, i) => setTimeout(() => b.style.width = b.dataset.w, reduce ? 0 : i * 90));
    obs.disconnect();
  }, { threshold: 0.2 }).observe(wrap);
})();

/* ══ 5. QUOTA SIMULATOR ═══════════════════════════════════════ */
function initSimulator() {
  const slider = $('#sim-slider');
  if (!slider) return;
  function update(f) {
    const df = f - 0.60;
    const tpr = Math.min(96.5, Math.max(76.0, 89.11 + (df / 0.05) * 3.18));
    const tnr = Math.min(92.0, Math.max(54.0, 77.58 - (df / 0.05) * 7.35));
    const a_cond = Math.max(0.80, 0.8702 - (df / 0.05) * 0.0108);
    const b_cond = Math.max(0.24, 0.2953 + (df / 0.05) * 0.0023);
    const overall = (0.5635 * a_cond * (tpr / 100) + 0.4365 * b_cond * (tnr / 100)) * 100;
    const misrouted = Math.round(35665 * (1 - (0.5635 * (tpr / 100) + 0.4365 * (tnr / 100))));

    $('#sim-f-val').textContent = f.toFixed(2);
    $('#sim-tpr-val').textContent = tpr.toFixed(1) + '%';
    $('#sim-tnr-val').textContent = tnr.toFixed(1) + '%';
    $('#sim-score-val').textContent = overall.toFixed(3) + '%';
    $('#sim-loss-val').textContent = misrouted.toLocaleString();
  }
  slider.addEventListener('input', e => update(parseFloat(e.target.value)));
  update(0.60);
}

/* ══ 6. SPECIMEN TILT — one focal interaction, inspecting a photo ══ */
function apply3DTilt(el) {
  if (reduce) return;
  el.addEventListener('mousemove', e => {
    const r = el.getBoundingClientRect();
    const rx = -((e.clientY - r.top - r.height / 2) / (r.height / 2)) * 5;
    const ry = ((e.clientX - r.left - r.width / 2) / (r.width / 2)) * 5;
    el.style.transform = `perspective(900px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg)`;
  });
  el.addEventListener('mouseleave', () => { el.style.transform = ''; });
}

/* ══════════════════════════════════════════════════════════════ */
Promise.all([
  fetch('assets/data/species.json').then(r => r.json()),
  fetch('assets/data/demo.json').then(r => r.json()),
  new Promise(res => setTimeout(res, reduce ? 0 : 550)),
]).then(([data, demo]) => {
  hidePreloader();
  const seen = data.species.filter(s => s.status === 'seen');
  const novel = data.species.filter(s => s.status === 'unseen');

  /* ── problem section: seen mosaic + novel void ─────────────── */
  $('#have').innerHTML = data.mosaic.slice(0, 12)
    .map(m => `<img src="${m.img}" alt="${m.name}" loading="lazy" decoding="async">`).join('');
  $('#havenot').innerHTML = novel.slice(0, 6)
    .map(s => `<div class="void-cell">No training photo<strong class="sci">${s.name}</strong></div>`).join('');
  revealNow($('#have'));
  revealNow($('#havenot'));

  /* ── species gallery ────────────────────────────────────────── */
  let currentFilter = 'all', searchQuery = '';
  const renderCard = s => `
    <article class="spec-card" data-status="${s.status}">
      ${s.img
      ? `<div class="photo"><img src="${s.img}" alt="${s.name}" loading="lazy" decoding="async"></div>`
      : `<div class="no-photo"><span>Zero training photographs.<br>Identified via text only.</span></div>`}
      <div class="body">
        <div class="name sci">${s.name}</div>
        <div class="meta${s.status === 'unseen' ? ' novel' : ''}">${s.status === 'seen' ? 'Seen' : 'Novel'}, ${s.n_train ? s.n_train + ' training photos' : '0 training photos'}</div>
        <p class="desc">${s.desc}</p>
      </div>
    </article>`;

  const grid = $('#cards');
  const draw = () => {
    const q = searchQuery.toLowerCase().trim();
    const filtered = data.species.filter(s => {
      const mf = currentFilter === 'all' || s.status === currentFilter;
      const ms = !q || s.name.toLowerCase().includes(q) || (s.desc && s.desc.toLowerCase().includes(q));
      return mf && ms;
    });
    grid.innerHTML = filtered.length
      ? filtered.map(renderCard).join('')
      : `<div class="empty-note">No matching species for "${searchQuery}"</div>`;
    $$('.spec-card', grid).forEach(apply3DTilt);
    revealNow(grid);
  };
  draw();

  $$('.filter-btn').forEach(b => b.addEventListener('click', () => {
    $$('.filter-btn').forEach(x => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    currentFilter = b.dataset.f;
    draw();
  }));
  $('#species-search').addEventListener('input', e => { searchQuery = e.target.value; draw(); });

  /* ── interactive pipeline ───────────────────────────────────── */
  let subject = demo[0];
  $('#picker').innerHTML = demo.map((d, i) => `
    <button class="pick-btn" aria-pressed="${i === 0}" data-i="${i}">
      <img src="${d.img}" alt="">${d.name}
    </button>`).join('');

  const renderCandidates = (list, gold) => list.map((c, i) => `
    <div class="cand-row ${c.cls === gold ? 'target' : ''}">
      <span>${i + 1}</span><span class="sci">${c.cls}</span><span class="score">${c.score.toFixed(3)}</span>
    </div>`).join('');

  function renderPipeline() {
    const d = subject;
    $$('.specimen-frame img').forEach(i => { i.src = d.img; i.alt = d.name; });
    $$('[data-fill="name"]').forEach(e => e.textContent = d.name);
    $$('[data-fill="ntrain"]').forEach(e => e.textContent = d.n_train);
    $('#seenlist').innerHTML = renderCandidates(d.seen_top, d.name);
    $('#unseenlist').innerHTML = renderCandidates(d.unseen_top, null);
    $('#seenbest').textContent = d.seen_top[0].score.toFixed(3);
    $('#unseenbest').textContent = d.unseen_top[0].score.toFixed(3);

    const pos = Math.min(97, Math.max(62, d.seen_top[0].score * 100));
    $('#gfill').style.width = pos + '%';
    $('#gsim').textContent = d.seen_top[0].score.toFixed(3);

    $('#answer').textContent = d.name;
    $('#wrongans').textContent = d.unseen_top[0].cls;
    const sameGenus = d.unseen_top[0].cls.split(' ')[0] === d.name.split(' ')[0];
    $('#wrongnote').textContent = sameGenus
      ? 'Same genus, close enough to look plausible, but scored as fully wrong'
      : 'A different genus entirely';
  }

  $$('.pick-btn').forEach(b => b.addEventListener('click', () => {
    $$('.pick-btn').forEach(x => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    subject = demo[+b.dataset.i];
    renderPipeline();
  }));

  const panels = $$('.stage-panel'), tabs = $$('.step-btn');
  tabs.forEach((t, i) => t.addEventListener('click', () => {
    tabs.forEach((x, j) => x.setAttribute('aria-selected', String(i === j)));
    panels.forEach((p, j) => p.classList.toggle('active', i === j));
  }));
  renderPipeline();

  initSimulator();
}).catch(err => { hidePreloader(); console.error('data load failed', err); });

/* ══════════════════════════════════════════════════════════════
   7. EVIDENCE — the report's Figure 1 as something you can pick up.
   Every point is a change measured both on the holdout and on the
   leaderboard; the data file carries the HANDOFF line for each one.
   ══════════════════════════════════════════════════════════════ */
const fmtD = v => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(3);

function placeTip(tip, host, x, y) {
  const r = host.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = x - r.left + 14, top = y - r.top - th - 10;
  if (left + tw > r.width - 8) left = x - r.left - tw - 14;
  if (top < 4) top = y - r.top + 16;
  tip.style.left = left + 'px';
  tip.style.top = top + 'px';
}

function drawScatter(host, E) {
  const W = 980, H = 430, ml = 58, mr = 26, mt = 30, mb = 48;
  const LT = 0.1;
  const sl = v => Math.sign(v) * Math.log10(1 + Math.abs(v) / LT);
  const x0 = sl(0.06), x1 = sl(24), y0 = sl(-1.05), y1 = sl(2.7);
  const px = x => ml + (sl(x) - x0) / (x1 - x0) * (W - ml - mr);
  const py = y => mt + (y1 - sl(y)) / (y1 - y0) * (H - mt - mb);

  let s = `<svg class="scatter draw" viewBox="0 0 ${W} ${H}" role="img" aria-label="Holdout promise against leaderboard payment for eighteen measured changes">`;

  // grid + zero
  [-0.5, 0.5, 1, 2].forEach(g => { s += `<line class="grid-line" x1="${ml}" x2="${W - mr}" y1="${py(g)}" y2="${py(g)}"/>`; });
  s += `<line class="zero" x1="${ml}" x2="${W - mr}" y1="${py(0)}" y2="${py(0)}"/>`;

  // anchor band: what the leak-free re-rankers actually paid per promised point
  const xs = [];
  for (let i = 0; i <= 60; i++) xs.push(0.06 * Math.pow(24 / 0.06, i / 60));
  const top = xs.map(x => `${px(x)},${py(E.anchor.hi * x)}`).join(' ');
  const bot = xs.slice().reverse().map(x => `${px(x)},${py(E.anchor.lo * x)}`).join(' ');
  s += `<polygon class="band" points="${top} ${bot}"/>`;

  // the honest line, y = x, until it leaves the frame
  const hx = xs.filter(x => x <= 2.7);
  s += `<polyline class="honest" points="${hx.map(x => `${px(x)},${py(x)}`).join(' ')}"/>`;
  s += `<text class="note" x="${px(2.25)}" y="${py(2.45)}" text-anchor="end">if the holdout were honest</text>`;
  s += `<text class="note good" x="${W - mr}" y="${py(0.36)}" text-anchor="end">leak-free re-rankers paid ${E.anchor.lo.toFixed(3)}–${E.anchor.hi.toFixed(3)} per point</text>`;

  // ticks
  [0.1, 0.3, 1, 3, 10].forEach(t => {
    s += `<text class="tick-label" x="${px(t)}" y="${H - mb + 16}" text-anchor="middle">${t}</text>`;
  });
  [-0.5, 0, 0.5, 1, 2].forEach(t => {
    s += `<text class="tick-label" x="${ml - 10}" y="${py(t) + 3.5}" text-anchor="end">${t < 0 ? '−' + Math.abs(t) : t}</text>`;
  });
  s += `<text class="axis-label" x="${ml + (W - ml - mr) / 2}" y="${H - 8}" text-anchor="middle">what the holdout promised · Δ overall accuracy, points</text>`;
  s += `<text class="axis-label" transform="translate(14 ${mt + (H - mt - mb) / 2}) rotate(-90)" text-anchor="middle">what the leaderboard paid</text>`;

  // callouts, positioned as in the report
  const lead = (x, y, tx, ty) => `<line class="lead" x1="${px(x)}" y1="${py(y)}" x2="${px(tx)}" y2="${py(ty)}"/>`;
  s += lead(13.374, 0.006, 3.2, 0.13) + `<text class="callout" x="${px(3.2)}" y="${py(0.13) - 4}">v81 · +13.374 promised, +0.006 paid</text>`;
  s += lead(8.154, 0.213, 1.2, 0.46) + `<text class="callout" x="${px(1.2)}" y="${py(0.46) - 4}">v82 · same idea, leak-free pool</text>`;
  s += lead(0.523, 1.766, 0.36, 1.28)
     + `<text class="callout strong" x="${px(0.072)}" y="${py(1.22)}">v77 · the learned gate</text>`
     + `<text class="callout strong" x="${px(0.072)}" y="${py(1.04)}">the only change that paid more than it promised</text>`;
  s += lead(9.66, 1.08, 8.5, 1.62) + `<text class="callout" x="${px(8.9)}" y="${py(1.62) - 4}" text-anchor="end">v37 · iNaturalist photo bank</text>`;
  s += lead(5.09, -0.6, 2.4, -0.8) + `<text class="callout bad" x="${px(0.42)}" y="${py(-0.8) + 4}">f = 0.72 · holdout preferred it by 5 points; the leaderboard docked 0.60</text>`;
  s += `<text class="callout bad" x="${px(3.51)}" y="${py(-0.07) + 22}" text-anchor="middle">v34</text>`;

  // points last, so they sit on top
  s += `<circle class="halo" cx="${px(0.523)}" cy="${py(1.766)}" r="12"/>`;
  E.pairs.forEach((p, i) => {
    const cls = p.paid < 0 ? 'flip' : p.paid > p.promised ? 'under' : '';
    s += `<circle class="dot ${cls}" data-i="${i}" cx="${px(p.promised)}" cy="${py(p.paid)}" r="${cls === 'under' ? 7 : 5}" style="transition-delay:${reduce ? 0 : 120 + i * 45}ms" tabindex="0"><title>${p.id}: ${fmtD(p.promised)} promised, ${fmtD(p.paid)} paid</title></circle>`;
  });
  s += `</svg>`;

  host.innerHTML = s + `<div class="tip" role="tooltip"></div>`;
  const svg = $('svg', host), tip = $('.tip', host);

  const show = (dot, x, y) => {
    const p = E.pairs[+dot.dataset.i];
    tip.innerHTML = `<div class="id">${p.id}</div><div class="what">${p.what}</div>
      <div class="row"><span>holdout promised</span><b>${fmtD(p.promised)}</b></div>
      <div class="row"><span>leaderboard paid</span><b>${fmtD(p.paid)}</b></div>
      <div class="src">HANDOFF.md, line ${p.line}</div>`;
    tip.classList.add('show');
    placeTip(tip, host, x, y);
    $$('.dot.on', svg).forEach(d => d.classList.remove('on'));
    dot.classList.add('on');
  };
  const hide = () => { tip.classList.remove('show'); $$('.dot.on', svg).forEach(d => d.classList.remove('on')); };

  $$('.dot', svg).forEach(dot => {
    dot.addEventListener('mouseenter', e => show(dot, e.clientX, e.clientY));
    dot.addEventListener('mousemove', e => placeTip(tip, host, e.clientX, e.clientY));
    dot.addEventListener('mouseleave', hide);
    dot.addEventListener('focus', () => { const r = dot.getBoundingClientRect(); show(dot, r.left + r.width / 2, r.top); });
    dot.addEventListener('blur', hide);
    dot.addEventListener('click', e => { e.stopPropagation(); const r = dot.getBoundingClientRect(); show(dot, r.left + r.width / 2, r.top); });
  });
  host.addEventListener('click', hide);

  // stats under the figure
  const over = E.pairs.filter(p => p.paid >= 0 && p.paid <= p.promised);
  const flips = E.pairs.filter(p => p.paid < 0), under = E.pairs.filter(p => p.paid > p.promised);
  const ratios = over.map(p => p.promised / p.paid).sort((a, b) => a - b);
  // true median: average the two middle values on an even count, as the report does
  const m = ratios.length >> 1;
  const med = ratios.length % 2 ? ratios[m] : (ratios[m - 1] + ratios[m]) / 2;
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  set('#ev-over', over.length); set('#ev-flip', flips.length); set('#ev-under', under.length);
  set('#ev-median', med.toFixed(1) + '×');

  if (reduce) { svg.classList.add('in'); return; }
  new IntersectionObserver((en, obs) => {
    if (en[0].isIntersecting) { svg.classList.add('in'); obs.disconnect(); }
  }, { threshold: 0.3 }).observe(host);
}

/* ══ 8. THE CAMPAIGN — 109 numbered builds, the ones the leaderboard saw ══ */
function drawTimeline(E) {
  const wrap = $('#timeline'), host = $('.timeline-wrap');
  if (!wrap || !host) return;
  const scored = new Map(E.scored.map(b => [b.v, b]));
  const ms = new Set(E.milestones);
  wrap.innerHTML = Array.from({ length: E.builds_total }, (_, i) => {
    const v = i + 1, b = scored.get(v);
    const cls = ['tick', b ? 'scored' : '', b && b.down ? 'down' : '', ms.has(v) && !b?.final ? 'milestone' : '', b?.final ? 'final' : ''].filter(Boolean).join(' ');
    return `<div class="${cls}" data-v="${v}" tabindex="${b ? 0 : -1}" aria-label="build v${v}${b ? ', ' + b.score.toFixed(2) + '%' : ''}"></div>`;
  }).join('');
  wrap.classList.add('draw');
  const set = (id, v) => { const el = $(id); if (el) el.textContent = v; };
  set('#tl-total', E.builds_total); set('#tl-scored', E.scored.length);

  const tip = $('.tip', host);
  const show = (t, x, y) => {
    const v = +t.dataset.v, b = scored.get(v);
    tip.innerHTML = b
      ? `<div class="id">v${v} · leaderboard</div><div class="what">${b.score.toFixed(2)}%</div><div class="row"><span>${b.what}</span></div>`
      : `<div class="id">v${v}</div><div class="what">resolved offline</div><div class="row"><span>no submission spent</span></div>`;
    tip.classList.add('show'); placeTip(tip, host, x, y);
    $$('.tick.on', wrap).forEach(d => d.classList.remove('on')); t.classList.add('on');
  };
  const hide = () => { tip.classList.remove('show'); $$('.tick.on', wrap).forEach(d => d.classList.remove('on')); };
  $$('.tick', wrap).forEach(t => {
    t.addEventListener('mouseenter', e => show(t, e.clientX, e.clientY));
    t.addEventListener('mousemove', e => placeTip(tip, host, e.clientX, e.clientY));
    t.addEventListener('mouseleave', hide);
    t.addEventListener('focus', () => { const r = t.getBoundingClientRect(); show(t, r.left, r.top); });
    t.addEventListener('blur', hide);
  });

  if (reduce) { wrap.classList.add('in'); return; }
  new IntersectionObserver((en, obs) => {
    if (!en[0].isIntersecting) return;
    $$('.tick', wrap).forEach((t, i) => t.style.transitionDelay = (i * 9) + 'ms');
    wrap.classList.add('in'); obs.disconnect();
  }, { threshold: 0.3 }).observe(wrap);
}

/* ══ 9. THE INSTRUMENT — what a holdout point is worth, in real points ══ */
function initCalculator(anchor) {
  const inp = $('#calc-in'), out = $('#calc-out'), verdict = $('#calc-verdict');
  if (!inp || !out) return;
  const run = () => {
    const v = Math.max(0, parseFloat(inp.value) || 0);
    const lo = v * anchor.lo, hi = v * anchor.hi;
    out.innerHTML = `${fmtD(lo)} to ${fmtD(hi)}<small>real points</small>`;
    if (v === 0) { verdict.className = 'verdict'; verdict.textContent = 'Type a holdout gain.'; return; }
    if (v < anchor.sign_unreliable_below) {
      verdict.className = 'verdict warn';
      verdict.textContent = `Below about ${anchor.sign_unreliable_below} proxy points the sign itself is unreliable: v84 promised +0.345 and paid −0.039. We stopped shipping changes with only this much support.`;
    } else if (hi < 0.3) {
      verdict.className = 'verdict';
      verdict.textContent = `Worth under a third of a point on the leaderboard. This is where most of our proposals were killed without spending a submission.`;
    } else {
      verdict.className = 'verdict';
      verdict.textContent = `Worth a submission. For scale, a full real point needs about +${Math.ceil(1 / anchor.hi)} proxy points, and nothing we had came close.`;
    }
  };
  inp.addEventListener('input', run);
  run();
}

/* ══ 10. STRICTNESS — how much a per-image build changes, as a switch ══ */
function initStrict(ladder) {
  const btns = $$('.strict-btn'), n = $('#strict-n'), pct = $('#strict-pct'), note = $('#strict-note');
  if (!btns.length || !n) return;
  const byKey = Object.fromEntries(ladder.map(l => [l.key, l]));
  let current = 0;
  const go = key => {
    const l = byKey[key]; if (!l) return;
    btns.forEach(b => b.setAttribute('aria-pressed', String(b.dataset.key === key)));
    const from = current, to = l.changed, t0 = performance.now(), dur = reduce ? 0 : 700;
    const tick = t => {
      const p = dur ? Math.min(1, (t - t0) / dur) : 1, e = 1 - Math.pow(1 - p, 3);
      const v = Math.round(from + (to - from) * e);
      n.textContent = v.toLocaleString();
      pct.textContent = (v / 35665 * 100).toFixed(2) + '%';
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    current = to;
    note.textContent = l.note;
  };
  btns.forEach(b => b.addEventListener('click', () => go(b.dataset.key)));
  go('shipped');
}

/* ══ 11. COMPLIANCE — the checks settle in, the gauges fill ═══════ */
(function initCompliance() {
  const sec = $('#compliance');
  if (!sec) return;
  const fill = () => {
    $$('.gauge .fill', sec).forEach(f => f.style.width = f.dataset.w);
    $$('.check-row', sec).forEach((r, i) => setTimeout(() => r.classList.add('in'), reduce ? 0 : 140 + i * 90));
  };
  if (reduce) { fill(); return; }
  new IntersectionObserver((en, obs) => { if (en[0].isIntersecting) { fill(); obs.disconnect(); } }, { threshold: 0.2 }).observe(sec);
})();

/* ══ 12. BOOT the evidence-driven pieces ═══════════════════════════ */
(function initEvidence() {
  const host = $('#scatter-host');
  if (!host && !$('#timeline') && !$('#calc-in')) return;
  fetch('assets/data/evidence.json').then(r => r.json()).then(E => {
    if (host) drawScatter(host, E);
    drawTimeline(E);
    initCalculator(E.anchor);
    initStrict(E.ladder);
  }).catch(err => console.error('evidence load failed', err));
})();
