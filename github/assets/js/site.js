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

watchCount($('#hero-score'), 53.697, 3);
watchCount($('#res-final'), 53.697, 3);
watchCount($('#res-seen'), 77.54, 2);
watchCount($('#res-novel'), 22.91, 2);

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
