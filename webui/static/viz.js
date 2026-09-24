/* ==========================================================================
   MiniMind 可视化基础库（无依赖，纯 SVG）
   --------------------------------------------------------------------------
   配套 `viz.css` 里的设计令牌。要点：

   - **配色可计算**：分类色序固定（--s1..--s8），不循环、不按排名重排；
     顺序色（--seq-*）用于量值，发散色（--div-*）用于「相对基线的正负」。
     这四类色各司其职，因此同一图里不会出现「颜色既表示分类又表示大小」。
   - **一种图型服务一件事**：折线看趋势、条形看量值、热力图看矩阵、
     发散条看增减、环图看构成、散点看权衡、管线图看流程。
     同一个页面里刻意混用不同图型，避免「全是折线」。
   - **交互默认带上**：折线有十字准星 + 数值浮层；条/格/点有逐标记浮层。
   - **两条硬约束**：绝不双 Y 轴；分类色不循环（超过 8 条就折叠或分面）。
   ========================================================================== */
(function () {
'use strict';

const SVG_NS = 'http://www.w3.org/2000/svg';
const PALETTE_N = 8;

/** 读 CSS 变量（主题切换后颜色自动跟着变，不需要在 JS 里硬编码两套色）。 */
function token(name, fallback = '#888') {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
/** 第 i 个分类色（i 从 0 起）。**按固定顺序取，不循环**——超过 8 条由调用方折叠。 */
function seriesColor(i) {
  return token(`--s${(i % PALETTE_N) + 1}`);
}
/** 顺序色：t ∈ [0,1] → 该色阶上的一个颜色。 */
function seqColor(t) {
  const steps = 8;
  const k = Math.max(0, Math.min(steps - 1, Math.round((Number.isFinite(t) ? t : 0) * (steps - 1))));
  return token(`--seq-${k + 1}`);
}
/** 发散色：v ∈ [-1,1]，0 为中性灰。用于「相对基线」。 */
function divColor(v) {
  const x = Math.max(-1, Math.min(1, Number.isFinite(v) ? v : 0));
  if (Math.abs(x) < 0.08) return token('--div-0');
  const k = Math.max(1, Math.min(4, Math.ceil(Math.abs(x) * 4)));
  return token(x > 0 ? `--div-p${k}` : `--div-n${k}`);
}

const svgEl = (tag, attrs = {}) => {
  const n = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
  return n;
};
const mkEl = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};

/* ------------------------------ 格式化 ---------------------------------- */
const fmtNum = (v, nd = 4) => (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(nd);
const fmtInt = (v) => (v === null || v === undefined) ? '—' : Math.round(v).toLocaleString('en-US');
const fmtSigned = (v, nd = 4) => (v === null || v === undefined) ? '—'
  : `${v > 0 ? '+' : v < 0 ? '−' : '±'}${Math.abs(Number(v)).toFixed(nd)}`;
const fmtPct = (v, nd = 1) => (v === null || v === undefined) ? '—' : `${(Number(v) * 100).toFixed(nd)}%`;
const fmtK = (v) => {
  if (v === null || v === undefined) return '—';
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}G`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)}k`;
  return String(Math.round(v * 1000) / 1000);
};
const fmtDur = (sec) => {
  if (sec === null || sec === undefined) return '—';
  const s = Math.round(sec);
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, '0')}s`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
};
const fmtBytes = (b) => (b === null || b === undefined) ? '—'
  : b >= 1 << 30 ? `${(b / (1 << 30)).toFixed(2)} GB` : b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : `${(b / 1024).toFixed(0)} kB`;
const escapeHtml = (s) => String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** 好看的刻度：在 [lo, hi] 上取 3~6 个「整数位」刻度。 */
function niceTicks(lo, hi, count = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.abs(raw) || 1)));
  const norm = raw / mag;
  const step = (norm >= 7.5 ? 10 : norm >= 3.5 ? 5 : norm >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const out = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) out.push(Number(v.toPrecision(12)));
  return out;
}

/* ------------------------------ 浮层 ------------------------------------ */
let _tipEl = null;
function tipEl() {
  if (!_tipEl) {
    _tipEl = mkEl('div', 'viz-tip');
    _tipEl.dataset.show = 'false';
    document.body.appendChild(_tipEl);
  }
  return _tipEl;
}
function showTip(html, ev) {
  const t = tipEl();
  t.innerHTML = html;
  t.dataset.show = 'true';
  const r = t.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY - r.height - 12;
  if (x + r.width > window.innerWidth - 8) x = Math.max(8, ev.clientX - r.width - 14);
  if (y < 8) y = ev.clientY + 18;
  t.style.left = `${x}px`;
  t.style.top = `${y}px`;
}
function hideTip() { if (_tipEl) _tipEl.dataset.show = 'false'; }
document.addEventListener('scroll', hideTip, true);

function tipRow(color, name, value) {
  return `<div class="viz-row">`
    + (color ? `<span class="viz-sw" style="background:${color}"></span>` : '<span class="viz-sw viz-sw-none"></span>')
    + `<span class="viz-k">${escapeHtml(name)}</span><span class="viz-v">${escapeHtml(value)}</span></div>`;
}
function tipHead(text) { return `<div class="viz-head">${escapeHtml(text)}</div>`; }

/* ======================================================================== */
/*  图例                                                                     */
/* ======================================================================== */
/**
 * 图例。`onToggle(key)` 时切换曲线可见性。
 * 两条以上曲线必须有图例（身份不能只靠颜色）。
 */
function legend(host, series, hidden, onToggle) {
  host.innerHTML = '';
  for (const s of series) {
    const b = mkEl('button', 'viz-leg', `<span class="viz-sw" style="background:${s.color}"></span>${escapeHtml(s.name)}`);
    b.type = 'button';
    const off = hidden && hidden.has(s.key);
    b.setAttribute('aria-pressed', off ? 'false' : 'true');
    if (off) b.classList.add('off');
    b.addEventListener('click', () => onToggle(s.key));
    host.appendChild(b);
  }
}

/* ======================================================================== */
/*  折线 / 面积图                                                            */
/* ======================================================================== */
/**
 * 折线图。series: [{key, name, color, dashed, points:[{x,y}], unit}]
 * opts: {height, xLabel, yLabel, xTickFmt, yTickFmt, yTickNd, zeroLine, yPad,
 *        area, unit, markY:{y,label}, bands:[{from,to,label}]}
 */
function lineChart(host, series, opts = {}) {
  host.innerHTML = '';
  const W = Math.max(280, host.clientWidth || 640);
  const H = opts.height || 220;
  const m = { t: opts.title ? 22 : 12, r: 14, b: 32, l: opts.labelW || 52 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;

  const live = series.filter((s) => s.points && s.points.length);
  if (!live.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }

  let xLo = Infinity, xHi = -Infinity, yLo = Infinity, yHi = -Infinity;
  for (const s of live) for (const p of s.points) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    if (p.x < xLo) xLo = p.x; if (p.x > xHi) xHi = p.x;
    if (p.y < yLo) yLo = p.y; if (p.y > yHi) yHi = p.y;
  }
  if (!Number.isFinite(xLo)) { host.appendChild(mkEl('div', 'viz-empty', '无有效数据')); return; }
  if (!(xHi > xLo)) xHi = xLo + 1;
  const span = (yHi - yLo) || Math.abs(yHi) * 0.2 || 1;
  const pad = span * (opts.yPad === undefined ? 0.12 : opts.yPad);
  yLo -= pad; yHi += pad;
  if (opts.zeroLine) { yLo = Math.min(yLo, 0); yHi = Math.max(yHi, 0); }

  const sx = (x) => m.l + ((x - xLo) / (xHi - xLo)) * iw;
  const sy = (y) => m.t + ih - ((y - yLo) / (yHi - yLo)) * ih;

  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H,
                             preserveAspectRatio: 'none' });

  if (opts.bands) for (const b of opts.bands) {
    const y1 = sy(Math.max(b.from, yLo)), y2 = sy(Math.min(b.to, yHi));
    svg.appendChild(svgEl('rect', { class: 'viz-band', x: m.l, y: Math.min(y1, y2),
                                    width: iw, height: Math.abs(y2 - y1) }));
    if (b.label) {
      const t = svgEl('text', { class: 'viz-bandlabel', x: m.l + iw - 4, y: Math.min(y1, y2) + 11, 'text-anchor': 'end' });
      t.textContent = b.label;
      svg.appendChild(t);
    }
  }

  const xt = niceTicks(xLo, xHi, Math.max(3, Math.floor(iw / 96)));
  const yt = niceTicks(yLo, yHi, Math.max(3, Math.floor(ih / 46)));
  for (const t of yt) {
    svg.appendChild(svgEl('line', { class: 'viz-grid', x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }));
    const lb = svgEl('text', { class: 'viz-tick', x: m.l - 7, y: sy(t) + 3.5, 'text-anchor': 'end' });
    lb.textContent = (opts.yTickFmt || fmtNum)(t, opts.yTickNd === undefined ? 3 : opts.yTickNd);
    svg.appendChild(lb);
  }
  for (const t of xt) {
    const lb = svgEl('text', { class: 'viz-tick', x: sx(t), y: m.t + ih + 15, 'text-anchor': 'middle' });
    lb.textContent = (opts.xTickFmt || fmtK)(t);
    svg.appendChild(lb);
  }
  svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
  if (opts.zeroLine) svg.appendChild(svgEl('line', { class: 'viz-zero', x1: m.l, x2: m.l + iw, y1: sy(0), y2: sy(0) }));

  // 面积只在单条曲线时垫底（多条会互相遮挡，反而更难读）
  if (live.length === 1 && opts.area !== false) {
    const pts = live[0].points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    const base = sy(Math.max(yLo, Math.min(yHi, opts.zeroLine ? 0 : yLo)));
    const d = `M${sx(pts[0].x)},${base}` + pts.map((p) => `L${sx(p.x)},${sy(p.y)}`).join('')
      + `L${sx(pts[pts.length - 1].x)},${base}Z`;
    const grad = svgEl('linearGradient', { id: `g${Math.random().toString(36).slice(2, 9)}`, x1: 0, y1: 0, x2: 0, y2: 1 });
    grad.appendChild(svgEl('stop', { offset: '0%', 'stop-color': live[0].color, 'stop-opacity': '.22' }));
    grad.appendChild(svgEl('stop', { offset: '100%', 'stop-color': live[0].color, 'stop-opacity': '0' }));
    const defs = svgEl('defs'); defs.appendChild(grad); svg.appendChild(defs);
    svg.appendChild(svgEl('path', { d, fill: `url(#${grad.id})`, stroke: 'none' }));
  }

  for (const s of live) {
    const pts = s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!pts.length) continue;
    const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join('');
    svg.appendChild(svgEl('path', { class: 'viz-line', d, stroke: s.color,
                                    'stroke-width': live.length > 4 ? 1.6 : 2,
                                    'stroke-dasharray': s.dashed ? '5 4' : undefined }));
    // 直接标注末点：4 条以内直接写在端点上，身份不只靠颜色。
    // 曲线收在右边界时标签改挂左侧（text-anchor:end）—— 否则 start 锚点会
    // 让整段文字探出 viewBox 右沿被裁掉。
    if (live.length <= 4 && opts.endLabels !== false) {
      const last = pts[pts.length - 1];
      const xe = sx(last.x);
      const flip = xe > m.l + iw - 58;
      const t = svgEl('text', { class: 'viz-endlabel',
                                x: flip ? xe - 5 : Math.min(xe + 4, m.l + iw - 2),
                                y: Math.max(m.t + 9, sy(last.y) - 4),
                                'text-anchor': flip ? 'end' : 'start' });
      t.textContent = s.shortName || s.name;
      svg.appendChild(t);
    }
  }

  if (opts.markY !== undefined && Number.isFinite(opts.markY)) {
    svg.appendChild(svgEl('line', { class: 'viz-mark', x1: m.l, x2: m.l + iw, y1: sy(opts.markY), y2: sy(opts.markY) }));
    if (opts.markYLabel) {
      const t = svgEl('text', { class: 'viz-marklabel', x: m.l + 4, y: sy(opts.markY) - 4 });
      t.textContent = opts.markYLabel;
      svg.appendChild(t);
    }
  }

  // 十字准星 + 最近点浮层
  const hit = svgEl('rect', { class: 'viz-hit', x: m.l, y: m.t, width: iw, height: ih });
  svg.appendChild(hit);
  const cross = svgEl('line', { class: 'viz-cross', y1: m.t, y2: m.t + ih, x1: -9, x2: -9 });
  svg.appendChild(cross);
  const dots = live.map((s) => {
    const c = svgEl('circle', { class: 'viz-dot', r: 3.4, fill: s.color });
    svg.appendChild(c);
    return c;
  });
  const xs = [];
  for (const s of live) for (const p of s.points) if (Number.isFinite(p.x)) xs.push(p.x);
  xs.sort((a, b) => a - b);
  const uniqX = xs.filter((v, i) => i === 0 || v !== xs[i - 1]);
  const bisect = (v) => {           // 最近公共 x（各系列 x 网格可能不同，先对齐到公共列）
    let lo = 0, hi = uniqX.length - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (uniqX[mid] < v) lo = mid + 1; else hi = mid; }
    if (lo > 0 && Math.abs(uniqX[lo - 1] - v) <= Math.abs(uniqX[lo] - v)) lo -= 1;
    return uniqX[lo];
  };

  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    if (px < m.l - 4 || px > m.l + iw + 4) { leave(); return; }
    const xv = xLo + ((px - m.l) / iw) * (xHi - xLo);
    const x0 = bisect(xv);
    cross.setAttribute('x1', sx(x0));
    cross.setAttribute('x2', sx(x0));
    cross.setAttribute('class', 'viz-cross on');
    const rows = [];
    live.forEach((s, i) => {
      let best = null, bd = Infinity;
      for (const p of s.points) {
        const d = Math.abs(p.x - x0);
        if (d < bd) { bd = d; best = p; }
      }
      if (best && bd <= (xHi - xLo) * 0.02 + 1e-9) {
        dots[i].setAttribute('cx', sx(best.x));
        dots[i].setAttribute('cy', sy(best.y));
        dots[i].setAttribute('opacity', 1);
        rows.push({ s, p: best });
      } else {
        dots[i].setAttribute('opacity', 0);
      }
    });
    if (!rows.length) { hideTip(); return; }
    rows.sort((a, b) => b.p.y - a.p.y);
    const head = `${(opts.xTickFmt || fmtK)(x0)}${opts.xUnit ? ` ${opts.xUnit}` : ''}`;
    showTip(tipHead(head) + rows.map((r) => tipRow(r.s.color, r.s.name,
      `${(opts.yTickFmt || fmtNum)(r.p.y, opts.tipNd === undefined ? 4 : opts.tipNd)}${opts.unit ? ` ${opts.unit}` : ''}`
      + (r.p.extra ? `  ·  ${r.p.extra}` : ''))).join(''), ev);
  };
  const leave = () => {
    hideTip();
    cross.setAttribute('class', 'viz-cross');
    dots.forEach((d) => d.setAttribute('opacity', 0));
  };
  hit.addEventListener('mousemove', move);
  hit.addEventListener('mouseleave', leave);
  svg.addEventListener('mouseleave', leave);

  if (opts.xLabel || opts.yLabel) {
    const g = svgEl('text', { class: 'viz-axlabel', x: m.l + iw, y: H - 2, 'text-anchor': 'end' });
    g.textContent = [opts.yLabel, opts.xLabel].filter(Boolean).join('  ·  ');
    svg.appendChild(g);
  }
  if (opts.title) {
    const t = svgEl('text', { class: 'viz-title', x: m.l, y: 13 });
    t.textContent = opts.title;
    svg.appendChild(t);
  }
  host.appendChild(svg);
}

/* ======================================================================== */
/*  条形图（横向）与发散条                                                    */
/* ======================================================================== */
/**
 * 横向条形图。items: [{label, value, color, sub, text, dim}]
 * opts: {labelW, rowH, valueLabel, symmetric, showZero, tickFmt, max}
 * `symmetric` 时以中线为 0，向左为负、向右为正 —— 用于「相对基线的增减」。
 */
function barChart(host, items, opts = {}) {
  host.innerHTML = '';
  if (!items || !items.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }
  const W = Math.max(280, host.clientWidth || 640);
  const rowH = opts.rowH || 22, gap = opts.gap === undefined ? 6 : opts.gap;
  const m = { t: 4, r: opts.valueW || 74, b: 20, l: opts.labelW || 150 };
  const H = m.t + m.b + items.length * (rowH + gap);
  const iw = Math.max(40, W - m.l - m.r);
  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H });

  const sym = !!opts.symmetric;
  let hi = opts.max !== undefined ? opts.max : Math.max(...items.map((i) => Math.abs(i.value) || 0), 1e-9);
  if (opts.minRef !== undefined) hi = Math.max(hi, Math.abs(opts.minRef));
  const zero = sym ? m.l + iw / 2 : m.l;
  const scale = sym ? iw / 2 : iw;

  for (const t of niceTicks(0, hi, Math.max(2, Math.floor(iw / 110)))) {
    const x = zero + (sym ? (t / hi) * scale : (t / hi) * scale);
    svg.appendChild(svgEl('line', { class: 'viz-grid', x1: x, x2: x, y1: m.t, y2: H - m.b }));
    const lb = svgEl('text', { class: 'viz-tick', x, y: H - 5, 'text-anchor': 'middle' });
    lb.textContent = (opts.tickFmt || fmtK)(t);
    svg.appendChild(lb);
    if (sym && t > 0) {
      const lb2 = svgEl('text', { class: 'viz-tick', x: zero - (t / hi) * scale, y: H - 5, 'text-anchor': 'middle' });
      lb2.textContent = `−${(opts.tickFmt || fmtK)(t)}`;
      svg.appendChild(lb2);
    }
  }
  svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l + iw, y1: H - m.b, y2: H - m.b }));
  svg.appendChild(svgEl('line', { class: sym ? 'viz-zero' : 'viz-axis', x1: zero, x2: zero, y1: m.t, y2: H - m.b }));

  items.forEach((it, i) => {
    const y = m.t + i * (rowH + gap);
    const lb = svgEl('text', { class: 'viz-tick viz-tick-strong', x: m.l - 8, y: y + rowH / 2 + 3.6, 'text-anchor': 'end' });
    lb.textContent = it.label;
    svg.appendChild(lb);

    const v = Number.isFinite(it.value) ? it.value : 0;
    const w = Math.max(1.5, (Math.abs(v) / hi) * scale);
    const neg = sym && v < 0;
    const x = sym ? (neg ? zero - w : zero) : m.l;
    const fill = it.color || (sym ? divColor(v / (hi || 1)) : token('--accent'));
    const r = svgEl('rect', { class: 'viz-bar', x, y, width: w, height: rowH,
                              rx: Math.min(4, rowH / 4), fill, opacity: it.dim ? 0.35 : 1 });
    r.addEventListener('mousemove', (ev) => showTip(
      tipHead(it.label)
      + tipRow(fill, opts.valueLabel || '值', it.text ?? String(it.value))
      + (it.sub ? `<div class="viz-row"><span class="viz-k">${escapeHtml(it.sub)}</span></div>` : ''), ev));
    r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);

    const vt = svgEl('text', {
      class: 'viz-val',
      x: sym ? (neg ? x - 6 : x + w + 6) : m.l + w + 6,
      y: y + rowH / 2 + 3.6,
      'text-anchor': neg ? 'end' : 'start',
    });
    vt.textContent = it.text ?? String(it.value);
    svg.appendChild(vt);
  });
  host.appendChild(svg);
}

/* ======================================================================== */
/*  柱状图（纵向，用于分面 / 小倍数）                                          */
/* ======================================================================== */
/**
 * 纵向柱状图。items: [{label, value, color, text}]，逐个 hover 出浮层。
 * opts: {height, yTickFmt, yTickNd, labelRotate, zeroLine, valueLabel}
 */
function columnChart(host, items, opts = {}) {
  host.innerHTML = '';
  if (!items || !items.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }
  const W = Math.max(220, host.clientWidth || 420);
  const H = opts.height || 170;
  const m = { t: 14, r: 8, b: opts.labelRotate ? 40 : 26, l: opts.labelW || 44 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = items.map((i) => Number(i.value) || 0);
  let lo = Math.min(0, ...vals), hi = Math.max(...vals, 1e-9);
  const pad = (hi - lo) * 0.1 || 1;
  hi += pad; if (lo < 0) lo -= pad;
  const sy = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const bw = Math.max(2, iw / items.length * 0.72);
  const step = iw / items.length;

  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H });
  for (const t of niceTicks(lo, hi, 3)) {
    svg.appendChild(svgEl('line', { class: 'viz-grid', x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }));
    const lb = svgEl('text', { class: 'viz-tick', x: m.l - 6, y: sy(t) + 3.5, 'text-anchor': 'end' });
    lb.textContent = (opts.yTickFmt || fmtK)(t);
    svg.appendChild(lb);
  }
  if (opts.zeroLine !== false) {
    svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l + iw, y1: sy(0), y2: sy(0) }));
  }
  items.forEach((it, i) => {
    const v = Number(it.value) || 0;
    const x = m.l + i * step + (step - bw) / 2;
    const y0 = sy(0), y1 = sy(v);
    const r = svgEl('rect', { class: 'viz-bar', x, y: Math.min(y0, y1), width: bw,
                              height: Math.max(1.2, Math.abs(y1 - y0)), rx: 2.5,
                              fill: it.color || token('--accent') });
    r.addEventListener('mousemove', (ev) => showTip(
      tipHead(it.full || it.label)
      + tipRow(it.color || token('--accent'), opts.valueLabel || '值', it.text ?? String(it.value))
      + (it.sub ? `<div class="viz-row"><span class="viz-k">${escapeHtml(it.sub)}</span></div>` : ''), ev));
    r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);
    const lb = svgEl('text', { class: 'viz-tick', x: x + bw / 2, y: H - (opts.labelRotate ? 26 : 6),
                               'text-anchor': opts.labelRotate ? 'end' : 'middle' });
    if (opts.labelRotate) lb.setAttribute('transform', `rotate(-40 ${x + bw / 2} ${H - 26})`);
    lb.textContent = it.short || it.label;
    svg.appendChild(lb);
  });
  host.appendChild(svg);
}

/* ======================================================================== */
/*  堆叠面积图（构成随时间的演化）                                            */
/* ======================================================================== */
/**
 * 堆叠面积图。用于「总量由哪几项构成、各项占比怎么变」——折线叠画会互相遮挡、
 * 看不出占比，柱状堆叠读不出趋势，只有堆叠面积能同时读出「总量」与「构成」。
 *
 * series: [{key, name, color, points:[{x,y}]}]。**各分量必须同量纲且可加**
 *   （reward = 长度分 + 思考分 + 闭合分 − 重复罚 + RM 分），否则堆出来的高度没有意义。
 * opts: {height, xLabel, yLabel, xTickFmt, yTickFmt, yTickNd, unit, labelW,
 *        totalLine:{name,color,points,shortName}, xUnit, tipNd}
 *
 * 混合正负的处理：**正分量从零线向上堆、负分量向下堆**，零线画在中间。
 * 把负数也顺着堆上去会让「总量」变成一个没有意义的中间值 —— 这是堆叠图最常见的错误。
 */
function stackedArea(host, series, opts = {}) {
  host.innerHTML = '';
  const W = Math.max(280, host.clientWidth || 640);
  const H = opts.height || 240;
  const m = { t: 12, r: 14, b: 32, l: opts.labelW || 56 };
  const iw = W - m.l - m.r;
  const ih = H - m.t - m.b;

  const live = (series || []).filter((s) => s.points && s.points.length);
  if (!live.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }

  // 逐点相加的前提是各分量落在同一组 x 上。这里取**交集**、绝不插值 ——
  // 插值会凭空造出原始 CSV 里没有的值，而这个图正是用来看「谁在涨」的。
  const xsets = live.map((s) => new Set(s.points.map((p) => p.x)));
  const xs = [...xsets[0]].filter((x) => xsets.every((st) => st.has(x))).sort((a, b) => a - b);
  if (xs.length < 3) {
    // 交集太小：说明这些序列的横轴口径不一致，不能堆叠。退化成各自的折线，不假装可加。
    lineChart(host, series, opts);
    return;
  }
  const getters = live.map((s) => {
    const map = new Map(s.points.map((p) => [p.x, p.y]));
    return (x) => map.get(x);
  });

  // 正负分层：posCum[i][k] = 第 i 条在 xs[k] 处的「正向上界」；neg 同理向下
  const posCum = live.map(() => new Array(xs.length).fill(0));
  const negCum = live.map(() => new Array(xs.length).fill(0));
  for (let k = 0; k < xs.length; k++) {
    let p = 0, q = 0;
    for (let i = 0; i < live.length; i++) {
      const v = Number(getters[i](xs[k]));
      if (!Number.isFinite(v)) continue;   // 该点缺值 = 这一项此刻为 0，不是「整条断掉」
      if (v >= 0) p += v; else q += v;
      posCum[i][k] = p;
      negCum[i][k] = q;
    }
  }

  let yLo = 0, yHi = 0;
  for (let k = 0; k < xs.length; k++) {
    yHi = Math.max(yHi, posCum.length ? posCum[posCum.length - 1][k] : 0);
    yLo = Math.min(yLo, negCum.length ? negCum[negCum.length - 1][k] : 0);
  }
  const pad = (yHi - yLo) * 0.08 || 1;
  yHi += pad; yLo -= pad;
  const sx = (x) => m.l + ((x - xs[0]) / ((xs[xs.length - 1] - xs[0]) || 1)) * iw;
  const sy = (y) => m.t + ih - ((y - yLo) / ((yHi - yLo) || 1)) * ih;

  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H });

  const yt = niceTicks(yLo, yHi, Math.max(3, Math.floor(ih / 46)));
  for (const t of yt) {
    svg.appendChild(svgEl('line', { class: 'viz-grid', x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }));
    const lb = svgEl('text', { class: 'viz-tick', x: m.l - 7, y: sy(t) + 3.5, 'text-anchor': 'end' });
    lb.textContent = (opts.yTickFmt || fmtNum)(t, opts.yTickNd === undefined ? 3 : opts.yTickNd);
    svg.appendChild(lb);
  }
  for (const t of niceTicks(xs[0], xs[xs.length - 1], Math.max(3, Math.floor(iw / 96)))) {
    const lb = svgEl('text', { class: 'viz-tick', x: sx(t), y: m.t + ih + 15, 'text-anchor': 'middle' });
    lb.textContent = (opts.xTickFmt || fmtK)(t);
    svg.appendChild(lb);
  }
  svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
  svg.appendChild(svgEl('line', { class: 'viz-zero', x1: m.l, x2: m.l + iw, y1: sy(0), y2: sy(0) }));

  // 各分量的带（band）。同号的分量互不重叠，因此绘制顺序不影响可读性。
  // 每个带用底色描边 1.5px，形成规范要求的 2px 分段缝隙。
  live.forEach((s, i) => {
    const cum = posCum[i], prev = i === 0 ? null : posCum[i - 1];
    const ncum = negCum[i], nprev = i === 0 ? null : negCum[i - 1];
    const band = (arr, parr) => {
      const up = xs.map((x, k) => `${sx(x)},${sy(parr ? parr[k] : 0)}`).join('L');
      const down = [...xs.keys()].reverse().map((k) => `${sx(xs[k])},${sy(arr[k])}`).join('L');
      return `M${up}L${down}Z`;
    };
    const hasPos = cum.some((v) => v > 0), hasNeg = ncum.some((v) => v < 0);
    if (hasPos) svg.appendChild(svgEl('path', { class: 'viz-area-seg', d: band(cum, prev), fill: s.color }));
    if (hasNeg) svg.appendChild(svgEl('path', { class: 'viz-area-seg', d: band(ncum, nprev), fill: s.color }));
  });

  // 总量线：分量之和。它可能来自另一条点数不同的曲线，所以独立传进来、不参与堆叠。
  const tl = opts.totalLine;
  if (tl && tl.points && tl.points.length) {
    const map = new Map(tl.points.map((p) => [p.x, p.y]));
    const pts = xs.filter((x) => map.has(x)).map((x) => ({ x, y: map.get(x) }));
    if (pts.length > 1) {
      const d = pts.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join('');
      svg.appendChild(svgEl('path', { class: 'viz-line', d, stroke: tl.color || token('--ink'),
                                      'stroke-width': 2.2 }));
      const last = pts[pts.length - 1];
      const t = svgEl('text', { class: 'viz-endlabel', x: Math.min(sx(last.x) + 4, m.l + iw - 2),
                                y: Math.max(m.t + 9, sy(last.y) - 5) });
      t.textContent = tl.shortName || tl.name || '合计';
      svg.appendChild(t);
    }
  }

  // 十字准星 + 该 step 上的完整构成（含合计与占比）——堆叠图没有浮层就只剩形状
  const hit = svgEl('rect', { class: 'viz-hit', x: m.l, y: m.t, width: iw, height: ih });
  svg.appendChild(hit);
  const cross = svgEl('line', { class: 'viz-cross', y1: m.t, y2: m.t + ih, x1: -9, x2: -9 });
  svg.appendChild(cross);
  const nearest = (px) => {
    const xv = xs[0] + ((px - m.l) / iw) * (xs[xs.length - 1] - xs[0]);
    let best = 0, bd = Infinity;
    xs.forEach((x, k) => { const d = Math.abs(x - xv); if (d < bd) { bd = d; best = k; } });
    return best;
  };
  const move = (ev) => {
    const box = svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * W;
    if (px < m.l - 4 || px > m.l + iw + 4) { leave(); return; }
    const k = nearest(px);
    cross.setAttribute('x1', sx(xs[k]));
    cross.setAttribute('x2', sx(xs[k]));
    cross.setAttribute('class', 'viz-cross on');
    const nd = opts.tipNd === undefined ? 4 : opts.tipNd;
    const rows = [];
    let total = 0;
    live.forEach((s, i) => {
      const v = Number(getters[i](xs[k]));
      if (!Number.isFinite(v)) return;
      total += v;
      rows.push({ s, v });
    });
    rows.sort((a, b) => b.v - a.v);
    showTip(tipHead(`${(opts.xTickFmt || fmtK)(xs[k])}${opts.xUnit ? ` ${opts.xUnit}` : ''}`)
      + rows.map((r) => tipRow(r.s.color, r.s.name, `${(opts.yTickFmt || fmtNum)(r.v, nd)}`
        + (opts.unit ? ` ${opts.unit}` : '')
        + (total ? `  ·  ${fmtPct(r.v / total, 0)}` : ''))).join('')
      + (rows.length > 1 ? tipRow(null, '合计', `${(opts.yTickFmt || fmtNum)(total, nd)}${opts.unit ? ` ${opts.unit}` : ''}`) : ''), ev);
  };
  const leave = () => {
    hideTip();
    cross.setAttribute('class', 'viz-cross');
  };
  hit.addEventListener('mousemove', move);
  hit.addEventListener('mouseleave', leave);
  svg.addEventListener('mouseleave', leave);

  if (opts.xLabel || opts.yLabel) {
    const g = svgEl('text', { class: 'viz-axlabel', x: m.l + iw, y: H - 2, 'text-anchor': 'end' });
    g.textContent = [opts.yLabel, opts.xLabel].filter(Boolean).join('  ·  ');
    svg.appendChild(g);
  }
  host.appendChild(svg);
}

/* ======================================================================== */
/*  热力图                                                                   */
/* ======================================================================== */
/**
 * 矩阵热力图 —— 一次看清「行 × 列」的二维结构（深度 × 层、权重 × 任务、配置 × 组件）。
 * rows/cols 是标签数组，values[r][c] 为数值（null 留空）。
 * opts: {mode:'seq'|'div', lo, hi, height, cellFmt, colW, colWMax, rowH, labelW}
 *   colW    列宽（不填则按宿主宽度自适应）—— 调用方通常按列数与标签长度算好自然宽度
 *   colWMax 自适应时的列宽上限（不填 = 46）
 *   labelW  左侧行标签栏宽度（不填 = 108）
 * 矩阵按自己的自然宽度绘制、左对齐；只有不传 colW 时才去铺宿主宽度。
 */
function heatmap(host, rows, cols, values, opts = {}) {
  host.innerHTML = '';
  if (!rows.length || !cols.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }
  const rowW = opts.labelW || 108;
  // 列宽只看「矩阵自身多少列、格子多宽」，不跟宿主宽度纠缠：
  //   给了 colW   → 就用它（调用方已按列数与标签长度算好自然宽度）；
  //   没给 colW   → 按宿主宽度自适应，夹在 [14, colWMax || 46] 的可读区间。
  // 关键在下面 svg 的 width：它跟 viewBox 同宽，元素不再被 width:100% 拉满再
  // 等比居中。旧写法（width:'100%' + 窄 viewBox + 默认 preserveAspectRatio）
  // 会让矩阵缩成卡片正中的一条窄带、左右各留一大片空白。
  const cw = opts.colW
    ? opts.colW
    : Math.max(14, Math.min(opts.colWMax || 46, ((host.clientWidth || 640) - rowW - 10) / cols.length));
  const rh = opts.rowH || 20;
  const W = rowW + cols.length * cw + 6;
  const headH = opts.colLabelH || (cols.length > 14 ? 34 : 18);
  const H = headH + rows.length * rh + (opts.footer || 0) + 4;

  // 值域：默认按数据自适应；div 模式以 0 为中心对称
  let lo = opts.lo, hi = opts.hi;
  if (lo === undefined || hi === undefined) {
    const flat = [];
    for (const row of values) for (const v of row) if (Number.isFinite(v)) flat.push(v);
    if (!flat.length) { host.appendChild(mkEl('div', 'viz-empty', '无有效数值')); return; }
    if (opts.mode === 'div') {
      const a = Math.max(Math.abs(Math.min(...flat)), Math.abs(Math.max(...flat))) || 1;
      lo = -a; hi = a;
    } else { lo = Math.min(...flat); hi = Math.max(...flat); }
  }
  if (!(hi > lo)) hi = lo + 1e-9;

  // width 跟 viewBox 同宽（不是 100%）：矩阵按自己的自然宽度绘制、左对齐，
  // 不被拉满卡片再居中 —— 同页的横向条/散点也是这个样子。
  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');

  cols.forEach((c, j) => {
    const t = svgEl('text', { class: 'viz-tick', x: rowW + j * cw + cw / 2, y: headH - 6, 'text-anchor': 'middle' });
    if (cols.length > 14) {
      t.setAttribute('transform', `rotate(-55 ${rowW + j * cw + cw / 2} ${headH - 6})`);
      t.setAttribute('text-anchor', 'end');
    }
    t.textContent = c;
    svg.appendChild(t);
  });

  const cellFmt = opts.cellFmt || ((v) => fmtNum(v, 2));
  // `values` 决定颜色；`raw` 决定显示与浮层。
  // 需要「按列归一化配色、但读数仍是原值」时（如各评测任务量纲不同）用这一对。
  const raw = opts.raw || values;
  rows.forEach((r, i) => {
    const y = headH + i * rh;
    const lb = svgEl('text', { class: 'viz-tick viz-tick-strong', x: rowW - 8, y: y + rh / 2 + 3.6, 'text-anchor': 'end' });
    lb.textContent = r;
    lb.addEventListener('mousemove', (ev) => showTip(tipHead(r), ev));
    lb.addEventListener('mouseleave', hideTip);
    svg.appendChild(lb);

    cols.forEach((c, j) => {
      const v = values[i] ? values[i][j] : null;
      const x = rowW + j * cw;
      const cell = svgEl('rect', {
        class: 'viz-cell', x: x + 1, y: y + 1, width: Math.max(1, cw - 2), height: Math.max(1, rh - 2),
        rx: 2.5,
        fill: !Number.isFinite(v) ? token('--cell-empty')
          : (opts.mode === 'div' ? divColor(((v - (lo + hi) / 2) / ((hi - lo) / 2)))
            : seqColor((v - lo) / (hi - lo))),
      });
      const rv = raw[i] ? raw[i][j] : null;
      cell.addEventListener('mousemove', (ev) => showTip(
        opts.tip ? opts.tip(i, j, rv)
          : (tipHead(`${r} · ${c}`)
            + tipRow(null, opts.valueLabel || '值', Number.isFinite(rv) ? (opts.rawFmt || cellFmt)(rv) : '—')
            + (opts.hint ? `<div class="viz-row"><span class="viz-k">${escapeHtml(opts.hint)}</span></div>` : '')), ev));
      cell.addEventListener('mouseleave', hideTip);
      svg.appendChild(cell);
      if (opts.showValues && cw >= 26 && Number.isFinite(rv)) {
        const t = svgEl('text', { class: 'viz-celltext', x: x + cw / 2, y: y + rh / 2 + 3.2, 'text-anchor': 'middle' });
        t.textContent = (opts.rawFmt || cellFmt)(rv);
        svg.appendChild(t);
      }
    });
  });

  // 色标：说明「深 = 大」，读者不必猜
  if (opts.legend !== false) {
    const ly = H - 12, lw = Math.min(160, iw0(W, rowW));
    const lx = rowW;
    const grad = svgEl('linearGradient', { id: `lg${Math.random().toString(36).slice(2, 9)}`, x1: 0, y1: 0, x2: 1, y2: 0 });
    const n = 8;
    for (let k = 0; k < n; k++) {
      grad.appendChild(svgEl('stop', {
        offset: `${(k / (n - 1)) * 100}%`,
        'stop-color': opts.mode === 'div' ? divColor(k / (n - 1) * 2 - 1) : seqColor(k / (n - 1)),
      }));
    }
    const defs = svgEl('defs'); defs.appendChild(grad); svg.appendChild(defs);
    svg.appendChild(svgEl('rect', { x: lx, y: ly - 7, width: lw, height: 7, rx: 3, fill: `url(#${grad.id})` }));
    const a = svgEl('text', { class: 'viz-tick', x: lx - 4, y: ly, 'text-anchor': 'end' });
    a.textContent = cellFmt(lo);
    const b = svgEl('text', { class: 'viz-tick', x: lx + lw + 4, y: ly });
    b.textContent = cellFmt(hi);
    svg.appendChild(a); svg.appendChild(b);
  }
  host.appendChild(svg);
}
function iw0(W, rowW) { return Math.max(60, W - rowW - 90); }

/* ======================================================================== */
/*  散点图（权衡 / 帕累托）                                                    */
/* ======================================================================== */
/** points: [{x, y, label, color, r, sub}]。用于 loss-吞吐这类二维权衡。 */
function scatterChart(host, points, opts = {}) {
  host.innerHTML = '';
  if (!points || !points.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }
  const W = Math.max(280, host.clientWidth || 520);
  const H = opts.height || 260;
  const m = { t: 12, r: 16, b: 34, l: 54 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = points.map((p) => p.x).filter(Number.isFinite);
  const ys = points.map((p) => p.y).filter(Number.isFinite);
  if (!xs.length || !ys.length) { host.appendChild(mkEl('div', 'viz-empty', '无有效数据')); return; }
  let xLo = Math.min(...xs), xHi = Math.max(...xs), yLo = Math.min(...ys), yHi = Math.max(...ys);
  const xp = (xHi - xLo) * 0.08 || 1, yp = (yHi - yLo) * 0.1 || 1;
  xLo -= xp; xHi += xp; yLo -= yp; yHi += yp;
  if (opts.yFromZero) yLo = Math.min(0, yLo);
  const sx = (x) => m.l + ((x - xLo) / (xHi - xLo)) * iw;
  const sy = (y) => m.t + ih - ((y - yLo) / (yHi - yLo)) * ih;

  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${W} ${H}`, width: '100%', height: H });
  for (const t of niceTicks(yLo, yHi, 4)) {
    svg.appendChild(svgEl('line', { class: 'viz-grid', x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t) }));
    const lb = svgEl('text', { class: 'viz-tick', x: m.l - 7, y: sy(t) + 3.5, 'text-anchor': 'end' });
    lb.textContent = (opts.yTickFmt || fmtK)(t);
    svg.appendChild(lb);
  }
  for (const t of niceTicks(xLo, xHi, 4)) {
    const lb = svgEl('text', { class: 'viz-tick', x: sx(t), y: m.t + ih + 16, 'text-anchor': 'middle' });
    lb.textContent = (opts.xTickFmt || fmtK)(t);
    svg.appendChild(lb);
  }
  svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih }));
  svg.appendChild(svgEl('line', { class: 'viz-axis', x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih }));

  // 帕累托前缘：左上角（loss 更小 & 吞吐更大）为优
  if (opts.pareto) {
    const sorted = [...points].filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => b.x - a.x);
    const front = [];
    let best = Infinity;
    for (const p of sorted) if (p.y < best) { best = p.y; front.push(p); }
    if (front.length > 1) {
      const d = front.map((p, i) => `${i ? 'L' : 'M'}${sx(p.x)},${sy(p.y)}`).join('');
      svg.appendChild(svgEl('path', { class: 'viz-front', d }));
    }
  }

  points.forEach((p, i) => {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
    const col = p.color || seriesColor(p.ci || 0);
    const c = svgEl('circle', { class: 'viz-pt', cx: sx(p.x), cy: sy(p.y), r: p.r || 4.5,
                                fill: col, opacity: p.dim ? 0.35 : 0.95 });
    c.addEventListener('mousemove', (ev) => showTip(
      tipHead(p.label)
      + tipRow(col, opts.xName || 'x', (opts.xTickFmt || fmtNum)(p.x, opts.nd || 2))
      + tipRow(col, opts.yName || 'y', (opts.yTickFmt || fmtNum)(p.y, opts.nd || 4))
      + (p.sub ? `<div class="viz-row"><span class="viz-k">${escapeHtml(p.sub)}</span></div>` : ''), ev));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
  });
  if (opts.xLabel || opts.yLabel) {
    const t = svgEl('text', { class: 'viz-axlabel', x: m.l + iw, y: H - 2, 'text-anchor': 'end' });
    t.textContent = [opts.yLabel, opts.xLabel].filter(Boolean).join('  ·  ');
    svg.appendChild(t);
  }
  host.appendChild(svg);
}

/* ======================================================================== */
/*  环形图                                                                   */
/* ======================================================================== */
/** slices: [{label, value, color}]。用于「构成」——总数为 1 个维度时最好读。 */
function donut(host, slices, opts = {}) {
  host.innerHTML = '';
  const items = (slices || []).filter((s) => s.value > 0);
  if (!items.length) { host.appendChild(mkEl('div', 'viz-empty', '无数据')); return; }
  const size = opts.size || 150;
  const R = size / 2, r = R * 0.62;
  const total = items.reduce((a, s) => a + s.value, 0) || 1;
  const svg = svgEl('svg', { class: 'viz-svg', viewBox: `0 0 ${size} ${size}`, width: size, height: size });
  let ang = -Math.PI / 2;
  items.forEach((s) => {
    const sweep = (s.value / total) * Math.PI * 2;
    const a0 = ang, a1 = ang + sweep;
    ang = a1;
    const big = sweep > Math.PI ? 1 : 0;
    const p = (rad, a) => [R + rad * Math.cos(a), R + rad * Math.sin(a)];
    const [x0, y0] = p(R - 1, a0), [x1, y1] = p(R - 1, a1);
    const [x2, y2] = p(r, a1), [x3, y3] = p(r, a0);
    const path = svgEl('path', {
      class: 'viz-slice',
      d: `M${x0},${y0}A${R - 1},${R - 1} 0 ${big} 1 ${x1},${y1}L${x2},${y2}A${r},${r} 0 ${big} 0 ${x3},${y3}Z`,
      fill: s.color || token('--accent'),
    });
    path.addEventListener('mousemove', (ev) => showTip(
      tipHead(s.label) + tipRow(s.color, opts.valueLabel || '数量', `${s.value}`)
      + tipRow(null, '占比', fmtPct(s.value / total)), ev));
    path.addEventListener('mouseleave', hideTip);
    svg.appendChild(path);
  });
  host.appendChild(svg);
  if (opts.center) {
    const c = mkEl('div', 'viz-donut-center', `<strong>${escapeHtml(opts.center.value)}</strong><span>${escapeHtml(opts.center.label)}</span>`);
    host.appendChild(c);
  }
}

/* ======================================================================== */
/*  迷你趋势线（嵌在表格/卡片里）                                              */
/* ======================================================================== */
function sparkline(host, values, opts = {}) {
  host.innerHTML = '';
  const vals = (values || []).filter(Number.isFinite);
  if (vals.length < 2) { host.appendChild(mkEl('span', 'viz-empty viz-empty-inline', '')); return; }
  const W = opts.width || 96, H = opts.height || 24;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const sx = (i) => (i / (vals.length - 1)) * (W - 2) + 1;
  const sy = (v) => H - 2 - ((v - lo) / ((hi - lo) || 1)) * (H - 4);
  const col = opts.color || token('--accent');
  const svg = svgEl('svg', { class: 'viz-svg viz-spark', viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  const d = vals.map((v, i) => `${i ? 'L' : 'M'}${sx(i)},${sy(v)}`).join('');
  svg.appendChild(svgEl('path', { class: 'viz-line', d, stroke: col, 'stroke-width': 1.5 }));
  const lx = sx(vals.length - 1), ly = sy(vals[vals.length - 1]);
  svg.appendChild(svgEl('circle', { cx: lx, cy: ly, r: 2, fill: col }));
  host.appendChild(svg);
}

/* ======================================================================== */
/*  流程管线图                                                                */
/* ======================================================================== */
/**
 * 全流程管线：节点横向排列，可分组。用于「从 tokenizer 到部署」的一图总览。
 * nodes: [{id, label, sub, state:'done'|'partial'|'todo'|'active', badge, group}]
 */
function pipeline(host, nodes, opts = {}) {
  host.innerHTML = '';
  host.style.overflowX = '';
  const n = nodes.length;
  if (!n) { host.appendChild(mkEl('div', 'viz-empty', '无节点')); return; }
  const rowH = opts.rowH || 74;
  const groups = [];
  for (const nd of nodes) {
    const last = groups[groups.length - 1];
    if (last && last.name === nd.group) last.items.push(nd);
    else groups.push({ name: nd.group, items: [nd] });
  }
  const maxItems = Math.max(1, ...groups.map((g) => g.items.length));

  // 尺寸：节点是定宽盒子，所以宽度只能「由列数与列宽推出来」，不能反过来把
  // viewBox 压到宿主宽度再靠 width:100% 拉伸 —— 那样 viewBox 比内容还窄，节点会
  // 一路溢出卡片右缘（窄屏下再被 body 的 overflow:hidden 直接裁掉，连横向滚动都
  // 没有）。这里先按自然尺寸画，宿主放不下时先收间距、再收列宽，收到下限仍放不下
  // 就退回自然宽度 + 宿主横向滚动。同 heatmap 的取法。
  let colW = opts.colW || 132;
  let gap = opts.gap || 26;
  // colWMin 112 不是随手取的：节点标签从盒左缘 24px 起排（11.5px 圆点后），
  // 右缘再留 6px，最长的一个标签「GRPO / CISPO」实测 77px ⇒ 24+77+6 = 107 ≤ 112。
  // 再窄标签就会顶穿节点盒、蹭到下一个节点上。
  const colWMin = opts.colWMin || 112, gapMin = opts.gapMin || 14;
  const extent = () => 8 + maxItems * (colW + gap) - gap;
  const avail = (host.clientWidth || 0) - 4;
  if (avail > 0 && extent() > avail) {
    gap = gapMin;
    if (extent() > avail) colW = Math.max(colWMin, Math.floor((avail - 8 - (maxItems - 1) * gap) / maxItems));
  }
  const W = Math.max(320, extent() + 4);
  const rows = groups.length || 1;
  const H = rows * rowH + rows * 10 + 6;
  const svg = svgEl('svg', { class: 'viz-svg viz-pipe', viewBox: `0 0 ${W} ${H}`, width: W, height: H });
  svg.setAttribute('preserveAspectRatio', 'xMinYMin meet');
  if (avail > 0 && W > avail) host.style.overflowX = 'auto';

  const STATE = {
    done: token('--ok'), active: token('--accent'), partial: token('--warn'),
    todo: token('--ink-3'), fail: token('--danger'),
  };
  groups.forEach((g, gi) => {
    const y = 6 + gi * (rowH + 10);
    if (g.name) {
      const t = svgEl('text', { class: 'viz-group', x: 2, y: y + 10 });
      t.textContent = g.name;
      svg.appendChild(t);
    }
    g.items.forEach((nd, i) => {
      const x = 8 + i * (colW + gap);
      const yy = y + (g.name ? 16 : 0);
      const col = STATE[nd.state] || STATE.todo;
      const box = svgEl('rect', { class: 'viz-node', x, y: yy, width: colW, height: rowH - 20,
                                  rx: 9, stroke: col, opacity: nd.state === 'todo' ? 0.55 : 1 });
      box.addEventListener('mousemove', (ev) => showTip(
        tipHead(nd.label) + (nd.sub ? `<div class="viz-row"><span class="viz-k">${escapeHtml(nd.sub)}</span></div>` : '')
        + (nd.detail || []).map((d) => tipRow(null, d[0], d[1])).join(''), ev));
      box.addEventListener('mouseleave', hideTip);
      svg.appendChild(box);
      const dot = svgEl('circle', { cx: x + 13, cy: yy + 16, r: 4, fill: col });
      svg.appendChild(dot);
      const lb = svgEl('text', { class: 'viz-nodelabel', x: x + 24, y: yy + 20 });
      lb.textContent = nd.label;
      svg.appendChild(lb);
      if (nd.badge) {
        const b = svgEl('text', { class: 'viz-nodebadge', x: x + 11, y: yy + 38 });
        b.textContent = nd.badge;
        svg.appendChild(b);
      }
      if (nd.sub) {
        const s = svgEl('text', { class: 'viz-nodesub', x: x + 11, y: yy + 38 });
        s.textContent = nd.sub;
        svg.appendChild(s);
      }
      if (i < g.items.length - 1) {
        const ax = x + colW, ay = yy + (rowH - 20) / 2;
        svg.appendChild(svgEl('path', { class: 'viz-arrow', d: `M${ax + 4},${ay}L${ax + gap - 6},${ay}` }));
        svg.appendChild(svgEl('path', { class: 'viz-arrow', d: `M${ax + gap - 10},${ay - 3.5}L${ax + gap - 5},${ay}L${ax + gap - 10},${ay + 3.5}Z`, fill: token('--ink-3'), stroke: 'none' }));
      }
    });
  });
  host.appendChild(svg);
}

/* ======================================================================== */
/*  分段进度条（epoch / 预算的完成度）                                        */
/* ======================================================================== */
/** segments: [{label, value, color}]；用于「预算里已经用掉多少」。 */
function meterBar(host, segments, opts = {}) {
  host.innerHTML = '';
  const total = segments.reduce((a, s) => a + (Number(s.value) || 0), 0) || 1;
  const bar = mkEl('div', 'viz-meter');
  const legend = mkEl('div', 'viz-meter-legend');
  for (const s of segments) {
    const w = ((Number(s.value) || 0) / total) * 100;
    if (w <= 0) continue;
    const seg = mkEl('div', 'viz-meter-seg');
    seg.style.width = `${w}%`;
    seg.style.background = s.color || token('--accent');
    seg.title = `${s.label}: ${s.value}`;
    seg.addEventListener('mousemove', (ev) => showTip(
      tipHead(s.label) + tipRow(s.color, opts.valueLabel || '数量', String(s.value))
      + tipRow(null, '占比', fmtPct(s.value / total)), ev));
    seg.addEventListener('mouseleave', hideTip);
    bar.appendChild(seg);
    legend.appendChild(mkEl('span', 'viz-meter-item',
      `<span class="viz-sw" style="background:${s.color || token('--accent')}"></span>${escapeHtml(s.label)} <b>${fmtK(s.value)}</b>`));
  }
  host.appendChild(bar);
  if (opts.legend !== false) host.appendChild(legend);
}

/* ------------------------------ 导出 ------------------------------------ */
// 整个文件包在一个 IIFE 里：只有 window.Viz 是全局的，内部符号不会和页面脚本撞名
// （lab.js / train.js 都会从 Viz 解构出自己需要的助手，且都声明为 const）。
window.Viz = {
  token, seriesColor, seqColor, divColor, PALETTE_N,
  fmtNum, fmtInt, fmtSigned, fmtPct, fmtK, fmtDur, fmtBytes, escapeHtml, niceTicks,
  mkEl, svgEl, showTip, hideTip, tipHead, tipRow, legend, lineChart, stackedArea, barChart,
  columnChart, heatmap, scatterChart, donut, sparkline, pipeline, meterBar,
};
})();
