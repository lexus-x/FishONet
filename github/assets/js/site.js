/* FishONet — page behaviour. No libraries.
   All species names, photographs, descriptions and head scores are REAL:
   loaded from assets/data/*.json, generated from the competition training
   set and the project's own cached BioCLIP embeddings. */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ── reveal on scroll ─────────────────────────────────── */
const revealer = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealer.unobserve(e.target); } });
}, { threshold: 0.12, rootMargin: '0px 0px -40px' });
const watch = el => revealer.observe(el);

/* ── count-up ─────────────────────────────────────────── */
function countUp(el) {
  const to = parseFloat(el.dataset.to), dp = +(el.dataset.dp || 0);
  if (reduce) { el.textContent = to.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp }); return; }
  const t0 = performance.now(), dur = 1300;
  const tick = t => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (to * e).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
const counter = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) { countUp(e.target); counter.unobserve(e.target); } });
}, { threshold: 0.6 });

/* ── bars grow when seen ──────────────────────────────── */
const barObs = new IntersectionObserver((es) => {
  es.forEach(e => {
    if (!e.isIntersecting) return;
    $$('.bf', e.target).forEach((b, i) => setTimeout(() => b.style.width = b.dataset.w, reduce ? 0 : i * 70));
    barObs.unobserve(e.target);
  });
}, { threshold: 0.35 });

/* ══════════════════════════════════════════════════════ */
Promise.all([
  fetch('assets/data/species.json').then(r => r.json()),
  fetch('assets/data/demo.json').then(r => r.json()),
]).then(([data, demo]) => {
  const seen  = data.species.filter(s => s.status === 'seen');
  const novel = data.species.filter(s => s.status === 'unseen');

  /* ── hero photo cluster ─────────────────────────────── */
  const layout = ['tall', '', '', 'wide', '', '', 'tall', ''];
  $('#cluster').innerHTML = seen.slice(0, 8).map((s, i) => `
    <figure class="${layout[i] || ''}">
      <img src="${s.img}" alt="${s.name}" loading="${i < 4 ? 'eager' : 'lazy'}" decoding="async">
      <figcaption>${s.name}</figcaption>
    </figure>`).join('');

  /* ── marquee ────────────────────────────────────────── */
  const strip = data.mosaic.map(m => `<img src="${m.img}" alt="${m.name}" loading="lazy" decoding="async">`).join('');
  $('#mqtrack').innerHTML = strip + strip;

  /* ── split: what we have vs what we don't ───────────── */
  $('#have').innerHTML = data.mosaic.slice(0, 12)
    .map(m => `<img src="${m.img}" alt="${m.name}" loading="lazy" decoding="async">`).join('');
  $('#havenot').innerHTML = novel.slice(0, 8).concat(novel.slice(0, 4))
    .map(s => `<div class="ghost-card">no photo<br>${s.name.split(' ')[0]}</div>`).join('');

  /* ── gallery ────────────────────────────────────────── */
  const card = s => `
    <article class="card rv" data-status="${s.status}">
      ${s.img
        ? `<div class="ph"><img src="${s.img}" alt="${s.name}" loading="lazy" decoding="async"></div>`
        : `<div class="noph"><span>NO TRAINING IMAGE<br>EXISTS &nbsp;·&nbsp; TEXT ONLY</span></div>`}
      <div class="bd">
        <div class="nm">${s.name}</div>
        <div class="meta">
          <span class="badge ${s.status === 'seen' ? 'seen' : 'novel'}">${s.status === 'seen' ? 'seen' : 'novel'}</span>
          &nbsp;${s.n_train ? s.n_train + ' training photos' : '0 training photos'}
        </div>
        <p class="ds">${s.desc}</p>
      </div>
    </article>`;
  const grid = $('#cards');
  const draw = f => {
    grid.innerHTML = data.species.filter(s => f === 'all' || s.status === f).map(card).join('');
    $$('.card', grid).forEach((c, i) => { c.style.transitionDelay = (i % 8) * 45 + 'ms'; watch(c); });
  };
  draw('all');
  $$('.fbtn').forEach(b => b.addEventListener('click', () => {
    $$('.fbtn').forEach(x => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    draw(b.dataset.f);
  }));

  /* ══ INTERACTIVE PIPELINE ═════════════════════════════ */
  let subject = demo[0];

  $('#picker').innerHTML = demo.map((d, i) => `
    <button class="pick" role="button" aria-pressed="${i === 0}" data-i="${i}">
      <img src="${d.img}" alt=""><span>${d.name}</span>
    </button>`).join('');

  const bars = (list, gold, cls) => list.map((c, i) => {
    const w = Math.max(4, (c.score / list[0].score) * 100);
    return `<div class="cand ${c.cls === gold ? 'gold' : ''}">
      <span class="r">${i + 1}</span>
      <span class="nm">${c.cls}</span>
      <span class="bar" style="width:${w * 0.42}px;background:${cls}"></span>
      <span class="sc">${c.score.toFixed(3)}</span>
    </div>`;
  }).join('');

  function render() {
    const d = subject;
    $$('.specimen img').forEach(i => { i.src = d.img; i.alt = d.name; });
    $$('[data-fill="name"]').forEach(e => e.textContent = d.name);
    $$('[data-fill="ntrain"]').forEach(e => e.textContent = d.n_train);

    $('#seenlist').innerHTML  = bars(d.seen_top, d.name, 'var(--c1)');
    $('#unseenlist').innerHTML = bars(d.unseen_top, null, 'var(--c2)');
    $('#seenbest').textContent   = d.seen_top[0].score.toFixed(3);
    $('#unseenbest').textContent = d.unseen_top[0].score.toFixed(3);

    /* Gate position from a REAL feature the gate actually reads (seen-head max
       similarity). Not the full 12-feature score — labelled as such in the UI. */
    const pos = Math.min(97, Math.max(62, d.seen_top[0].score * 100));
    $('#gyou').style.left = pos + '%';
    $('#gfill').style.width = pos + '%';
    $('#gsim').textContent = d.seen_top[0].score.toFixed(3);

    $('#answer').textContent = d.name;
    $('#wrongans').textContent = d.unseen_top[0].cls;
    const sameGenus = d.unseen_top[0].cls.split(' ')[0] === d.name.split(' ')[0];
    $('#wrongnote').textContent = sameGenus
      ? 'the same genus — close enough to look right, still scored as wrong'
      : 'a different genus entirely';
  }

  $$('.pick').forEach(b => b.addEventListener('click', () => {
    $$('.pick').forEach(x => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    subject = demo[+b.dataset.i];
    render();
    if (!reduce) $$('.enc').forEach((e, i) => {
      e.classList.remove('lit'); setTimeout(() => e.classList.add('lit'), 90 * i);
    });
  }));

  /* stage switching */
  const panels = $$('.panel'), tabs = $$('.step');
  const go = n => {
    tabs.forEach((t, i) => t.setAttribute('aria-selected', String(i === n)));
    panels.forEach((p, i) => p.classList.toggle('on', i === n));
    if (n === 1 && !reduce) $$('.enc').forEach((e, i) => {
      e.classList.remove('lit'); setTimeout(() => e.classList.add('lit'), 100 * i);
    });
  };
  tabs.forEach((t, i) => t.addEventListener('click', () => go(i)));
  go(0);
  render();

  $$('.rv').forEach(watch);
}).catch(err => {
  console.error('data load failed', err);
  const g = document.querySelector('#cards');
  if (g) g.innerHTML = '<p>Could not load species data. Serve this folder over HTTP (e.g. <code>python -m http.server</code>) rather than opening the file directly.</p>';
});

/* observers that do not depend on fetched data */
$$('.rv').forEach(watch);
$$('[data-to]').forEach(e => counter.observe(e));
$$('.bars').forEach(e => barObs.observe(e));
