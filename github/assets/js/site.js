/* FishONet — Dynamic Behavioral & Interactive Script
   Includes:
   - Ambient Aquatic Particle Physics Canvas
   - Dynamic Quota Routing Sandbox Simulator
   - 3D Interactive Card Tilting
   - Pipeline Flow & Encoder Glow Micro-interactions
   - Live Count-Up Metrics & Smooth Scrollspy
*/

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ══ 1. AMBIENT AQUATIC PARTICLE CANVAS ═══════════════ */
(function initAquaticCanvas() {
  if (reduce) return;
  const canvas = document.createElement('canvas');
  canvas.id = 'bg-canvas';
  document.body.prepend(canvas);
  const ctx = canvas.getContext('2d');

  let w = (canvas.width = window.innerWidth);
  let h = (canvas.height = window.innerHeight);

  const particles = [];
  const count = Math.min(45, Math.floor(w / 35));

  let mouse = { x: w / 2, y: h / 2, active: false };

  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * w,
      y: Math.random() * h,
      r: Math.random() * 2.2 + 0.8,
      vx: (Math.random() - 0.5) * 0.45,
      vy: -Math.random() * 0.5 - 0.25, // upward gentle drift
      alpha: Math.random() * 0.5 + 0.2,
      pulse: Math.random() * Math.PI,
      color: Math.random() > 0.4 ? 'rgba(0, 212, 198, ' : 'rgba(224, 150, 0, '
    });
  }

  window.addEventListener('resize', () => {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  });

  window.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY + window.scrollY;
    mouse.active = true;
  });

  function draw() {
    ctx.clearRect(0, 0, w, h);

    particles.forEach(p => {
      p.pulse += 0.025;
      const currentAlpha = p.alpha + Math.sin(p.pulse) * 0.15;

      // Mouse repulsion
      if (mouse.active) {
        const dx = p.x - mouse.x;
        const dy = p.y - (mouse.y - window.scrollY);
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 120) {
          const force = (120 - dist) / 120;
          p.x += (dx / dist) * force * 2.5;
          p.y += (dy / dist) * force * 2.5;
        }
      }

      p.x += p.vx;
      p.y += p.vy;

      if (p.y < -10) p.y = h + 10;
      if (p.x < -10) p.x = w + 10;
      if (p.x > w + 10) p.x = -10;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color + Math.max(0, currentAlpha) + ')';
      ctx.shadowBlur = 10;
      ctx.shadowColor = p.color + '0.5)';
      ctx.fill();
    });

    requestAnimationFrame(draw);
  }
  draw();
})();

/* ══ 2. REVEAL & SCROLLSPY ═════════════════════════════ */
const revealer = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); revealer.unobserve(e.target); } });
}, { threshold: 0.1, rootMargin: '0px 0px -40px' });
const watch = el => revealer.observe(el);

// Scrollspy for navigation
const navLinks = $$('.nav a.lnk');
const sections = $$('section, header');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(sec => {
    const top = sec.offsetTop - 120;
    if (window.scrollY >= top) current = sec.getAttribute('id');
  });
  navLinks.forEach(link => {
    link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
  });
  $('.nav').classList.toggle('scrolled', window.scrollY > 20);
}, { passive: true });

/* ══ 3. COUNT-UP METRICS ══════════════════════════════ */
function countUp(el) {
  const to = parseFloat(el.dataset.to), dp = +(el.dataset.dp || 0);
  if (reduce) { el.textContent = to.toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp }); return; }
  const t0 = performance.now(), dur = 1400;
  const tick = t => {
    const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 4); // Quartic ease out
    el.textContent = (to * e).toLocaleString('en-US', { minimumFractionDigits: dp, maximumFractionDigits: dp });
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
const counter = new IntersectionObserver((es) => {
  es.forEach(e => { if (e.isIntersecting) { countUp(e.target); counter.unobserve(e.target); } });
}, { threshold: 0.5 });

/* ── bars grow when visible ─────────────────────────── */
const barObs = new IntersectionObserver((es) => {
  es.forEach(e => {
    if (!e.isIntersecting) return;
    $$('.bf', e.target).forEach((b, i) => setTimeout(() => b.style.width = b.dataset.w, reduce ? 0 : i * 80));
    barObs.unobserve(e.target);
  });
}, { threshold: 0.3 });

/* ══ 4. 3D TILT EFFECT FOR CARDS ═══════════════════════ */
function apply3DTilt(element) {
  if (reduce) return;
  element.addEventListener('mousemove', e => {
    const rect = element.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    const rx = -(y / (rect.height / 2)) * 6; // max 6 deg
    const ry = (x / (rect.width / 2)) * 6;
    element.style.transform = `perspective(1000px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-5px)`;
  });
  element.addEventListener('mouseleave', () => {
    element.style.transform = '';
  });
}

/* ══ 5. INTERACTIVE QUOTA ROUTING SIMULATOR ════════════ */
function initSimulator() {
  const slider = $('#sim-slider');
  if (!slider) return;

  const fVal = $('#sim-f-val');
  const tprVal = $('#sim-tpr-val');
  const tnrVal = $('#sim-tnr-val');
  const scoreVal = $('#sim-score-val');
  const lossVal = $('#sim-loss-val');
  const simMarker = $('#sim-marker');

  function update(f) {
    // Exact calibrated empirical transfer curve:
    // f=0.60 is peak: TPR ~ 89.11%, TNR ~ 77.58%, overall = 53.697%
    // Delta per 0.05 f: dTPR +3.18, dTNR -7.35, dScore -0.18
    const df = f - 0.60;
    const tpr = Math.min(96.5, Math.max(76.0, 89.11 + (df / 0.05) * 3.18));
    const tnr = Math.min(92.0, Math.max(54.0, 77.58 - (df / 0.05) * 7.35));
    
    // Overall = 0.5635 * a_cond * TPR + 0.4365 * b_cond * TNR
    // At f=0.60: a_cond=0.8702, b_cond=0.2953
    const a_cond = Math.max(0.80, 0.8702 - (df / 0.05) * 0.0108);
    const b_cond = Math.max(0.24, 0.2953 + (df / 0.05) * 0.0023);
    const overall = (0.5635 * a_cond * (tpr / 100) + 0.4365 * b_cond * (tnr / 100)) * 100;
    const misrouted = Math.round(35665 * (1 - (0.5635 * (tpr / 100) + 0.4365 * (tnr / 100))));

    fVal.textContent = f.toFixed(2);
    tprVal.textContent = tpr.toFixed(1) + '%';
    tnrVal.textContent = tnr.toFixed(1) + '%';
    scoreVal.textContent = overall.toFixed(3) + '%';
    lossVal.textContent = misrouted.toLocaleString();

    if (simMarker) {
      simMarker.style.left = (f * 100) + '%';
    }
  }

  slider.addEventListener('input', e => {
    update(parseFloat(e.target.value));
  });
  update(0.60);
}

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

  $$('#cluster figure').forEach(apply3DTilt);

  /* ── marquee ────────────────────────────────────────── */
  const strip = data.mosaic.map(m => `<img src="${m.img}" alt="${m.name}" loading="lazy" decoding="async">`).join('');
  $('#mqtrack').innerHTML = strip + strip;

  /* ── split: what we have vs what we don't ───────────── */
  $('#have').innerHTML = data.mosaic.slice(0, 12)
    .map(m => `<img src="${m.img}" alt="${m.name}" loading="lazy" decoding="async">`).join('');
  $('#havenot').innerHTML = novel.slice(0, 8).concat(novel.slice(0, 4))
    .map(s => `<div class="ghost-card">no photo<br><strong>${s.name.split(' ')[0]}</strong></div>`).join('');

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
    $$('.card', grid).forEach((c, i) => {
      c.style.transitionDelay = (i % 8) * 40 + 'ms';
      watch(c);
      apply3DTilt(c);
    });
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
    const w = Math.max(6, (c.score / list[0].score) * 100);
    return `<div class="cand ${c.cls === gold ? 'gold' : ''}">
      <span class="r">${i + 1}</span>
      <span class="nm">${c.cls}</span>
      <span class="bar" style="width:${w * 0.44}px;background:${cls}"></span>
      <span class="sc">${c.score.toFixed(3)}</span>
    </div>`;
  }).join('');

  function render() {
    const d = subject;
    $$('.specimen img').forEach(i => { i.src = d.img; i.alt = d.name; });
    $$('[data-fill="name"]').forEach(e => e.textContent = d.name);
    $$('[data-fill="ntrain"]').forEach(e => e.textContent = d.n_train);

    $('#seenlist').innerHTML   = bars(d.seen_top, d.name, 'var(--c1)');
    $('#unseenlist').innerHTML = bars(d.unseen_top, null, 'var(--c2)');
    $('#seenbest').textContent   = d.seen_top[0].score.toFixed(3);
    $('#unseenbest').textContent = d.unseen_top[0].score.toFixed(3);

    const pos = Math.min(97, Math.max(62, d.seen_top[0].score * 100));
    $('#gyou').style.left = pos + '%';
    $('#gfill').style.width = pos + '%';
    $('#gsim').textContent = d.seen_top[0].score.toFixed(3);

    $('#answer').textContent = d.name;
    $('#wrongans').textContent = d.unseen_top[0].cls;
    const sameGenus = d.unseen_top[0].cls.split(' ')[0] === d.name.split(' ')[0];
    $('#wrongnote').textContent = sameGenus
      ? 'the same genus — close enough to look plausible, but scored as 0% wrong'
      : 'a different genus entirely';
  }

  $$('.pick').forEach(b => b.addEventListener('click', () => {
    $$('.pick').forEach(x => x.setAttribute('aria-pressed', 'false'));
    b.setAttribute('aria-pressed', 'true');
    subject = demo[+b.dataset.i];
    render();
    if (!reduce) $$('.enc').forEach((e, i) => {
      e.classList.remove('lit'); setTimeout(() => e.classList.add('lit'), 80 * i);
    });
  }));

  /* Stage switching */
  const panels = $$('.panel'), tabs = $$('.step');
  const go = n => {
    tabs.forEach((t, i) => t.setAttribute('aria-selected', String(i === n)));
    panels.forEach((p, i) => p.classList.toggle('on', i === n));
    if (n === 1 && !reduce) $$('.enc').forEach((e, i) => {
      e.classList.remove('lit'); setTimeout(() => e.classList.add('lit'), 80 * i);
    });
  };
  tabs.forEach((t, i) => t.addEventListener('click', () => go(i)));
  go(0);
  render();

  initSimulator();
  $$('.rv').forEach(watch);
}).catch(err => {
  console.error('data load failed', err);
});

/* Observers that do not depend on fetched data */
$$('.rv').forEach(watch);
$$('[data-to]').forEach(e => counter.observe(e));
$$('.bars').forEach(e => barObs.observe(e));
